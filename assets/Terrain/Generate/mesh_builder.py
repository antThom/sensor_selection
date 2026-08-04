from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate
from shapely.geometry.polygon import orient

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomVertexReader,
    NodePath,
    Texture, 
    TexturePool, 
    TextureStage
)


ROAD_WIDTHS = {
    "motorway": 22.0,
    "trunk": 18.0,
    "primary": 14.0,
    "secondary": 10.0,
    "tertiary": 8.0,
    "residential": 6.0,
    "service": 4.0,
    "unclassified": 5.0,
    "living_street": 4.5,
    "default": 5.0,
}

DEFAULT_BUILDING_HEIGHT = 8.0
DEFAULT_ROAD_THICKNESS = 0.25
DEFAULT_WATER_THICKNESS = 0.08
DEFAULT_PARK_THICKNESS = 0.05
BRIDGE_CLEARANCE = 5.0
EPS = 1e-8


def _first_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        for v in value:
            return v
        return default
    return value


def _normalize_highway(value) -> str:
    value = _first_scalar(value, "default")
    if value is None:
        return "default"
    return str(value).split(";")[0].strip().lower()


def _parse_height_m(value) -> Optional[float]:
    if value is None:
        return None
    try:
        s = str(value).strip().lower()
        s = s.replace("meters", "").replace("meter", "").replace("m", "").strip()
        return float(s)
    except Exception:
        return None


def _representative_xy(geom: BaseGeometry) -> tuple[float, float]:
    if geom is None or geom.is_empty:
        return 0.0, 0.0
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        pt = geom.representative_point()
    elif geom.geom_type in ("LineString", "MultiLineString"):
        pt = geom.interpolate(0.5, normalized=True)
    elif geom.geom_type == "Point":
        pt = geom
    else:
        pt = geom.centroid
    return float(pt.x), float(pt.y)


def _ensure_single_polygon(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def _buffer_road_geometry(geom: BaseGeometry, width: float):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.buffer(width * 0.5, cap_style=2, join_style=2)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    return None


def _geom_to_xy_polygon(geom: BaseGeometry):
    poly = _ensure_single_polygon(geom)
    if poly is not None:
        return poly
    return None


def _polygon_valid_tris(poly: Polygon):
    tris = triangulate(poly)
    valid = []
    for tri in tris:
        rp = tri.representative_point()
        if poly.contains(rp) or poly.touches(rp):
            valid.append(tri)
    return valid


def _face_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / norm

def _repair_geom(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return None
    try:
        geom = geom.buffer(0)
    except Exception:
        return None
    if geom.is_empty:
        return None
    return geom

def _iter_polygon_parts(geom: BaseGeometry):
    """
    Yield Polygon objects from Polygon, MultiPolygon, or GeometryCollection.
    """
    geom = _repair_geom(geom)
    if geom is None:
        return

    if geom.geom_type == "Polygon":
        yield orient(geom, sign=1.0)

    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            part = _repair_geom(part)
            if part is not None and part.geom_type == "Polygon":
                yield orient(part, sign=1.0)

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygon_parts(part)

def _iter_rings(poly: Polygon):
    yield poly.exterior
    for hole in poly.interiors:
        yield hole
            
@dataclass
class MeshAsset:
    name: str
    nodepath: NodePath
    kind: str
    source_index: int
    metadata: dict


class CityMeshBuilder:
    def __init__(
        self,
        terrain_sampler: Callable[[float, float], float],
        output_dir: str = "city_output",
    ):
        self.terrain_sampler = terrain_sampler
        self.output_dir = Path(output_dir)
        self.assets: list[MeshAsset] = []
        self.buildings: list[BuildingAsset] = []
        self.tile_size = 500.0
        self.tiles = {}
        self.root = NodePath("city_root")

    def sample_ground_z(self, geom: BaseGeometry) -> float:
        x, y = _representative_xy(geom)
        return float(self.terrain_sampler(x, y))

    def _tile_key(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.tile_size)), int(np.floor(y / self.tile_size)))
    
    def _get_tile_node(self, tile_key: tuple[int, int]) -> NodePath:
        if tile_key not in self.tiles:
            ix, iy = tile_key
            tile_node = self.root.attachNewNode(f"tile_{ix}_{iy}")
            self.tiles[tile_key] = tile_node
        return self.tiles[tile_key]
    
    def _make_geomnode_from_extrusion(self, geom: BaseGeometry, height: float, name: str, color: Optional[tuple[float, float, float, float]] = None,) -> NodePath:
        """
        Create a Panda3D GeomNode from an extruded polygonal geometry.

        Supports:
        - Polygon
        - MultiPolygon
        - GeometryCollection containing polygonal parts
        """
        geom = _repair_geom(geom)
        if geom is None:
            return NodePath(name)

        # Position + normal + texture coordinates
        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData(name, fmt, Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        nwriter = GeomVertexWriter(vdata, "normal")
        twriter = GeomVertexWriter(vdata, "texcoord")
        prim = GeomTriangles(Geom.UHStatic)

        # Texture scaling controls
        cap_uv_scale = 0.02   # flat surfaces
        side_u_scale = 0.02   # along wall length
        side_v_scale = 0.10   # vertical repeat

        def add_tri(p0, p1, p2, normal, uv0, uv1, uv2):
            start_index = vdata.getNumRows()
            for p, uv in ((p0, uv0), (p1, uv1), (p2, uv2)):
                vwriter.addData3f(float(p[0]), float(p[1]), float(p[2]))
                nwriter.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
                twriter.addData2f(float(uv[0]), float(uv[1]))
            prim.addVertices(start_index, start_index + 1, start_index + 2)
            prim.closePrimitive()

        def ring_segments(coords):
            pts = list(coords)
            if len(pts) < 2:
                return []
            if np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if len(pts) < 2:
                return []
            return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

        def cap_uv(x: float, y: float) -> tuple[float, float]:
            return (x * cap_uv_scale, y * cap_uv_scale)

        def add_polygon(poly: Polygon):
            poly = orient(poly, sign=1.0)

            # Top / bottom caps
            cap_tris = _polygon_valid_tris(poly)
            for tri in cap_tris:
                coords = list(tri.exterior.coords)[:3]
                p0 = np.array([coords[0][0], coords[0][1], 0.0], dtype=float)
                p1 = np.array([coords[1][0], coords[1][1], 0.0], dtype=float)
                p2 = np.array([coords[2][0], coords[2][1], 0.0], dtype=float)

                uv0 = cap_uv(coords[0][0], coords[0][1])
                uv1 = cap_uv(coords[1][0], coords[1][1])
                uv2 = cap_uv(coords[2][0], coords[2][1])

                # bottom
                add_tri(p0, p2, p1,
                        np.array([0.0, 0.0, -1.0], dtype=float),
                        uv0, uv2, uv1)

                # top
                p0t = p0.copy(); p0t[2] = height
                p1t = p1.copy(); p1t[2] = height
                p2t = p2.copy(); p2t[2] = height
                add_tri(p0t, p1t, p2t,
                        np.array([0.0, 0.0, 1.0], dtype=float),
                        uv0, uv1, uv2)

            # Side walls for exterior + holes
            for ring in _iter_rings(poly):
                coords = list(ring.coords)
                ccw = ring.is_ccw

                for p0_xy, p1_xy in ring_segments(coords):
                    x0, y0 = float(p0_xy[0]), float(p0_xy[1])
                    x1, y1 = float(p1_xy[0]), float(p1_xy[1])

                    b0 = np.array([x0, y0, 0.0], dtype=float)
                    b1 = np.array([x1, y1, 0.0], dtype=float)
                    t0 = np.array([x0, y0, height], dtype=float)
                    t1 = np.array([x1, y1, height], dtype=float)

                    edge_len = float(np.linalg.norm(np.array([x1 - x0, y1 - y0], dtype=float)))
                    u0 = 0.0
                    u1 = edge_len * side_u_scale
                    v0 = 0.0
                    v1 = height * side_v_scale

                    edge = np.array([x1 - x0, y1 - y0, 0.0], dtype=float)
                    up = np.array([0.0, 0.0, 1.0], dtype=float)
                    normal = _face_normal(b0, b1, t1)
                    if np.linalg.norm(normal) < EPS:
                        normal = np.cross(edge, up)
                        nrm = np.linalg.norm(normal)
                        if nrm < EPS:
                            normal = np.array([0.0, 1.0, 0.0], dtype=float)
                        else:
                            normal = normal / nrm

                    # Exterior and hole rings need opposite winding
                    if ccw:
                        add_tri(b0, b1, t1, normal, (u0, v0), (u1, v0), (u1, v1))
                        add_tri(b0, t1, t0, normal, (u0, v0), (u1, v1), (u0, v1))
                    else:
                        add_tri(b0, t1, b1, normal, (u0, v0), (u1, v1), (u1, v0))
                        add_tri(b0, t0, t1, normal, (u0, v0), (u0, v1), (u1, v1))

        for poly in _iter_polygon_parts(geom):
            add_polygon(poly)

        node = GeomNode(name)
        g = Geom(vdata)
        g.addPrimitive(prim)
        node.addGeom(g)
        return NodePath(node)   
     
    def _add_asset(self, name: str, nodepath: NodePath, kind: str, source_index: int, metadata: dict, tile_key):
        parent = self._get_tile_node(tile_key)
        nodepath.reparentTo(parent)
        self.assets.append(
            MeshAsset(
                name=name,
                nodepath=nodepath,
                kind=kind,
                source_index=source_index,
                metadata=metadata,
            )
        )

    def add_buildings(self, buildings: gpd.GeoDataFrame):
        if buildings is None or buildings.empty:
            return

        # Optional small cache if many buildings hit the same point
        ground_cache = {}

        asset_start = len(self.assets)

        for asset_idx, row in enumerate(buildings.itertuples(index=False)):
            geom = getattr(row, "geometry", None)
            poly = _geom_to_xy_polygon(geom)
            if poly is None:
                continue

            # Use precomputed metadata if present
            height = getattr(row, "height_m", None)
            if height is None or np.isnan(height):
                height = _parse_height_m(getattr(row, "height", None))
            if height is None or np.isnan(height):
                levels = getattr(row, "building_levels", None)
                if levels is None:
                    levels = getattr(row, "_4", None)  # only if your dataframe layout is odd; otherwise remove
                try:
                    if levels is not None and not pd.isna(levels):
                        height = float(levels) * 3.0
                except Exception:
                    pass
            if height is None or np.isnan(height):
                height = DEFAULT_BUILDING_HEIGHT

            x, y = _representative_xy(poly)
            tile_key = self._tile_key(x, y)

            # Cache ground sampling by tile center-ish location
            cache_key = (round(x, 2), round(y, 2))
            if cache_key in ground_cache:
                base_z = ground_cache[cache_key]
            else:
                base_z = self.sample_ground_z(poly)
                ground_cache[cache_key] = base_z

            node = self._make_geomnode_from_extrusion(
                poly, height, f"building_{asset_start + asset_idx}"
            )
            node.setZ(base_z)

            landuse_tag = getattr(row, "landuse_tag", None)

            building = BuildingAsset(
                name=f"building_{asset_start + asset_idx}",
                nodepath=node,
                source_index=asset_idx,
                height_m=float(height),
                base_z=float(base_z),
                landuse_tag=landuse_tag,
                building_tag=getattr(row, "building", None),
                metadata={
                    "height_m": float(height),
                    "base_z": float(base_z),
                    "landuse_tag": landuse_tag,
                    "building": getattr(row, "building", None),
                },
            )

            parent = self._get_tile_node(tile_key)
            node.reparentTo(parent)

            self.buildings.append(building)
            self.assets.append(
                MeshAsset(
                    name=building.name,
                    nodepath=building.nodepath,
                    kind="building",
                    source_index=asset_idx,
                    metadata=building.metadata,
                )
            )
        
    def add_roads(self, roads: gpd.GeoDataFrame):
        if roads is None or roads.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(roads.iterrows()):
            geom = row.geometry
            highway = _normalize_highway(row.get("highway", "default"))
            width = float(ROAD_WIDTHS.get(highway, ROAD_WIDTHS["default"]))
            road_poly = _buffer_road_geometry(geom, width)
            if road_poly is None:
                continue

            base_z = self.sample_ground_z(geom)
            is_bridge = False
            if "bridge" in roads.columns:
                is_bridge = str(row.get("bridge", "")).strip().lower() in ("yes", "true", "1")
            if "layer" in roads.columns:
                try:
                    is_bridge = is_bridge or (float(row.get("layer", 0)) > 0)
                except Exception:
                    pass
            if is_bridge:
                base_z += BRIDGE_CLEARANCE
            x, y = _representative_xy(geom)
            tile_key = self._tile_key(x, y)
            node = self._make_geomnode_from_extrusion(
                road_poly,
                DEFAULT_ROAD_THICKNESS,
                f"road_{len(self.assets)}",
            )
            node.setZ(base_z)

            self._add_asset(
                name=f"road_{len(self.assets)}",
                nodepath=node,
                kind="bridge" if is_bridge else "road",
                source_index=asset_idx,
                metadata={
                    "highway": highway,
                    "width_m": width,
                    "base_z": float(base_z),
                    "bridge": bool(is_bridge),
                },
                tile_key=tile_key,
            )

    def add_bridges(self, roads: gpd.GeoDataFrame):
        if roads is None or roads.empty:
            return

        if "bridge" not in roads.columns and "layer" not in roads.columns:
            return

        bridge_mask = np.zeros(len(roads), dtype=bool)

        if "bridge" in roads.columns:
            bridge_mask |= roads["bridge"].fillna("").astype(str).str.lower().isin(["yes", "true", "1"]).to_numpy()

        if "layer" in roads.columns:
            try:
                bridge_mask |= roads["layer"].fillna(0).astype(float).to_numpy() > 0
            except Exception:
                pass

        bridge_roads = roads.loc[bridge_mask]
        if bridge_roads.empty:
            return

        for i, row in bridge_roads.iterrows():
            geom = row.geometry
            highway = _normalize_highway(row.get("highway", "default"))
            width = float(ROAD_WIDTHS.get(highway, ROAD_WIDTHS["default"]))
            road_poly = _buffer_road_geometry(geom, width)
            if road_poly is None:
                continue

            base_z = self.sample_ground_z(geom) + BRIDGE_CLEARANCE
            x, y = _representative_xy(geom)
            tile_key = self._tile_key(x, y)
            node = self._make_geomnode_from_extrusion(
                road_poly,
                DEFAULT_ROAD_THICKNESS,
                f"bridge_{len(self.assets)}",
            )
            node.setZ(base_z)

            self._add_asset(
                name=f"bridge_{len(self.assets)}",
                nodepath=node,
                kind="bridge",
                source_index=int(i),
                metadata={
                    "highway": highway,
                    "width_m": width,
                    "base_z": float(base_z),
                    "bridge": True,
                },
                tile_key=tile_key,
            )

    def add_water(self, water: gpd.GeoDataFrame):
        if water is None or water.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(water.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
            x, y = _representative_xy(poly)
            tile_key = self._tile_key(x, y)
            node = self._make_geomnode_from_extrusion(
                poly,
                DEFAULT_WATER_THICKNESS,
                f"water_{len(self.assets)}",
            )
            node.setZ(z)
            self._add_asset(
                name=f"water_{len(self.assets)}",
                nodepath=node,
                kind="water",
                source_index=asset_idx,
                metadata={"surface_z": float(z)},
                tile_key=tile_key,
            )

    def add_parks(self, parks: gpd.GeoDataFrame):
        if parks is None or parks.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(parks.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
            x, y = _representative_xy(poly)
            tile_key = self._tile_key(x, y)
            node = self._make_geomnode_from_extrusion(
                poly,
                DEFAULT_PARK_THICKNESS,
                f"park_{len(self.assets)}",
            )
            node.setZ(z)
            self._add_asset(
                name=f"park_{len(self.assets)}",
                nodepath=node,
                kind="park",
                source_index=asset_idx,
                metadata={"surface_z": float(z)},
                tile_key=tile_key,
            )
            
    def _write_obj_from_nodepath(self, nodepath: NodePath, fh, object_name: str, vertex_offset: int) -> int:
        node = nodepath.node()
        if not hasattr(node, "getNumGeoms"):
            return vertex_offset

        fh.write(f"o {object_name}\n")
        fh.write(f"g {object_name}\n")

        for geom_index in range(node.getNumGeoms()):
            geom = node.getGeom(geom_index)
            vdata = geom.getVertexData()
            vreader = GeomVertexReader(vdata, "vertex")

            nreader = None
            try:
                nreader = GeomVertexReader(vdata, "normal")
            except Exception:
                nreader = None

            local_vertices = []
            local_normals = []

            while not vreader.isAtEnd():
                v = vreader.getData3f()
                local_vertices.append((float(v[0]), float(v[1]), float(v[2])))

                if nreader is not None and not nreader.isAtEnd():
                    n = nreader.getData3f()
                    local_normals.append((float(n[0]), float(n[1]), float(n[2])))

            for vx, vy, vz in local_vertices:
                fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for nx, ny, nz in local_normals:
                fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

            for prim_index in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(prim_index)
                for p in range(prim.getNumPrimitives()):
                    s = prim.getPrimitiveStart(p)
                    e = prim.getPrimitiveEnd(p)
                    if e - s != 3:
                        continue
                    i0 = prim.getVertex(s + 0) + 1 + vertex_offset
                    i1 = prim.getVertex(s + 1) + 1 + vertex_offset
                    i2 = prim.getVertex(s + 2) + 1 + vertex_offset
                    fh.write(f"f {i0} {i1} {i2}\n")

            vertex_offset += len(local_vertices)

        return vertex_offset
    
    def _iter_polygon_rings(self, geom):
        if geom is None or geom.is_empty:
            return

        if geom.geom_type == "Polygon":
            yield geom

        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                yield part

        elif geom.geom_type == "GeometryCollection":
            for part in geom.geoms:
                yield from self._iter_polygon_rings(part)
                
    def export_bam(self, filename: str = "city.bam", flatten: bool = False) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(self.output_dir, filename)

        # if flatten:
        #     self.root.flattenStrong()
        self.root.writeBamFile(str(out_path))
        return out_path
    
    def export_obj(self, filename: str = "city.obj") -> Path:
        """Export the city as a Wavefront OBJ file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("# City mesh export\n")
            fh.write(f"# assets={len(self.assets)}\n")
            fh.write("mtllib city.mtl\n")

            vertex_offset = 0
            for asset in self.assets:
                vertex_offset = self._write_obj_from_nodepath(
                    asset.nodepath,
                    fh,
                    asset.name,
                    vertex_offset,
                )

        mtl_path = self.output_dir / "city.mtl"
        with mtl_path.open("w", encoding="utf-8") as fh:
            fh.write("newmtl default\n")
            fh.write("Kd 0.8 0.8 0.8\n")

        return out_path

    def _load_texture(self, texture_path: str) -> Texture:
        path_cwd = Path.cwd()
        texture_path = Path(path_cwd,texture_path)
        tex = TexturePool.loadTexture(texture_path)
        if tex is None:
            raise FileNotFoundError(f"Could not load texture: {texture_path}")
        return tex

    def _building_texture_for_height(
        self,
        height_m: float,
        landuse: str | None,
        texture_rules: dict,
    ):
        """
        Select a building texture.

        Parameters
        ----------
        height_m : float
            Building height in meters.

        landuse : str
            OSM landuse tag associated with the building
            (e.g. residential, commercial, retail, industrial).

        texture_rules : dict
            Example:

            {
                "residential": "textures/buildings/house.jpg",
                "commercial":  "textures/buildings/commercial.jpg",
                "retail":      "textures/buildings/retail.jpg",
                "industrial":  "textures/buildings/industrial.jpg",
                "default": [
                    (10, "textures/buildings/lowrise.jpg"),
                    (25, "textures/buildings/midrise.jpg"),
                    (1e9, "textures/buildings/highrise.jpg"),
                ]
            }
        """

        # First try the landuse texture
        if landuse:
            landuse = landuse.lower()
            if landuse in texture_rules:
                return texture_rules[landuse]

        # Fall back to height-based texture selection
        for height_max, tex_path in texture_rules["default"]:
            if height_m <= height_max:
                return tex_path

        return texture_rules["default"][-1][1]

    def apply_textures(
        self,
        building_rules,
        road_texture_path: str,
        park_texture_path: str,
        water_texture_path: str,
    ):
        road_tex = self._load_texture(str(road_texture_path))
        park_tex = self._load_texture(str(park_texture_path))
        water_tex = self._load_texture(str(water_texture_path))
        tex_stage = TextureStage("tex")

        texture_library = BuildingTextureLibrary(
            landuse_textures={k.lower(): Path(v) for k, v in building_rules},
            default_height_rules=[
                (10.0, Path("assets/textures/building_materials/bricks/Bricks101_1K-JPG/Bricks101_1K-JPG_Color.jpg")),
                (25.0, Path("assets/textures/building_materials/bricks/Bricks097_1K-JPG/Bricks097_1K-JPG_Color.jpg")),
                (1e9, Path("assets/textures/building_materials/office/window-pattern-textures-building.jpg")),
            ],
        )

        for building in self.buildings:
            tex_path = texture_library.select(building)
            tex = self._load_texture(str(tex_path))
            building.nodepath.setTexture(tex_stage, tex, 1)

        for asset in self.assets:
            if asset.kind == "road":
                asset.nodepath.setTexture(tex_stage, road_tex, 1)
            elif asset.kind == "park":
                asset.nodepath.setTexture(tex_stage, park_tex, 1)
            elif asset.kind == "water":
                asset.nodepath.setTexture(tex_stage, water_tex, 1)
            elif asset.kind == "bridge":
                asset.nodepath.setTexture(tex_stage, road_tex, 1)
            
    def export_manifest_csv(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for asset in self.assets:
            rec = {"name": asset.name, "kind": asset.kind, "source_index": asset.source_index}
            rec.update(asset.metadata)
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        out_path = self.output_dir / "city_manifest.csv"
        df.to_csv(out_path, index=False)
        return out_path
    
@dataclass
class BuildingAsset:
    name: str
    nodepath: NodePath
    source_index: int
    height_m: float
    base_z: float
    landuse_tag: Optional[str] = None
    building_tag: Optional[str] = None
    building_levels: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, name: str, nodepath: NodePath, source_index: int, row, height_m: float, base_z: float):
        def _first(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple, set)):
                return next(iter(v), None)
            return v

        landuse_tag = _first(row.get("landuse_tag", None))
        building_tag = _first(row.get("building", None))
        levels = row.get("building:levels", None)

        try:
            if levels is not None and pd.isna(levels):
                levels = None
        except Exception:
            pass

        meta = dict(row.drop(labels=["geometry"], errors="ignore"))
        meta["height_m"] = float(height_m)
        meta["base_z"] = float(base_z)
        meta["landuse_tag"] = landuse_tag
        meta["building_tag"] = building_tag
        meta["building_levels"] = levels

        return cls(
            name=name,
            nodepath=nodepath,
            source_index=source_index,
            height_m=float(height_m),
            base_z=float(base_z),
            landuse_tag=landuse_tag,
            building_tag=building_tag,
            building_levels=levels,
            metadata=meta,
        )

    @property
    def texture_key(self) -> str:
        if self.landuse_tag:
            return str(self.landuse_tag).strip().lower()
        return "default"
    
@dataclass
class BuildingTextureLibrary:
    landuse_textures: dict[str, Path]
    default_height_rules: list[tuple[float, Path]]

    def select(self, building: BuildingAsset) -> Path:
        landuse = (building.landuse_tag or "").strip().lower()
        if landuse in self.landuse_textures:
            return self.landuse_textures[landuse]

        for height_max, tex_path in self.default_height_rules:
            if building.height_m <= height_max:
                return tex_path

        return self.default_height_rules[-1][1]
    
