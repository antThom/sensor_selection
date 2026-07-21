from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
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
        self.root = NodePath("city_root")

    def sample_ground_z(self, geom: BaseGeometry) -> float:
        x, y = _representative_xy(geom)
        return float(self.terrain_sampler(x, y))

    def _make_geomnode_from_extrusion(
        self,
        poly: Polygon,
        height: float,
        name: str,
        color: Optional[tuple[float, float, float, float]] = None,
    ) -> NodePath:
        """Create a Panda3D GeomNode from an extruded polygon.

        The extrusion is built explicitly so the result can be exported directly
        to BAM without an intermediate mesh format.
        """
        if poly is None or poly.is_empty:
            return NodePath(name)

        fmt = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData(name, fmt, Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        nwriter = GeomVertexWriter(vdata, "normal")
        prim = GeomTriangles(Geom.UHStatic)

        def add_tri(p0, p1, p2, normal):
            start_index = vdata.getNumRows()
            for p in (p0, p1, p2):
                vwriter.addData3f(float(p[0]), float(p[1]), float(p[2]))
                nwriter.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
            prim.addVertices(start_index, start_index + 1, start_index + 2)
            prim.closePrimitive()

        def ring_segments(coords):
            pts = list(coords)
            if len(pts) < 2:
                return []
            if np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

        # Top / bottom caps via triangulation, filtered to the polygon interior.
        cap_tris = _polygon_valid_tris(poly)
        for tri in cap_tris:
            coords = list(tri.exterior.coords)[:3]
            p0 = np.array([coords[0][0], coords[0][1], 0.0])
            p1 = np.array([coords[1][0], coords[1][1], 0.0])
            p2 = np.array([coords[2][0], coords[2][1], 0.0])
            # Bottom face
            add_tri(p0, p2, p1, np.array([0.0, 0.0, -1.0]))
            # Top face
            p0t = p0.copy(); p0t[2] = height
            p1t = p1.copy(); p1t[2] = height
            p2t = p2.copy(); p2t[2] = height
            add_tri(p0t, p1t, p2t, np.array([0.0, 0.0, 1.0]))

        # Side walls for exterior + holes.
        for ring in [poly.exterior, *list(poly.interiors)]:
            for p0_xy, p1_xy in ring_segments(ring.coords):
                x0, y0 = float(p0_xy[0]), float(p0_xy[1])
                x1, y1 = float(p1_xy[0]), float(p1_xy[1])

                b0 = np.array([x0, y0, 0.0])
                b1 = np.array([x1, y1, 0.0])
                t0 = np.array([x0, y0, height])
                t1 = np.array([x1, y1, height])

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

                # Use ring order to keep winding consistent.
                add_tri(b0, b1, t1, normal)
                add_tri(b0, t1, t0, normal)

        node = GeomNode(name)
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node.addGeom(geom)
        return NodePath(node)

    def _add_asset(self, name: str, nodepath: NodePath, kind: str, source_index: int, metadata: dict):
        nodepath.reparentTo(self.root)
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
        for asset_idx, (gdf_idx, row) in enumerate(buildings.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue

            height = _parse_height_m(row.get("height", None))
            if height is None or np.isnan(height):
                levels = row.get("building:levels", None)
                try:
                    if levels is not None and not pd.isna(levels):
                        height = float(levels) * 3.0
                except Exception:
                    pass
            if height is None or np.isnan(height):
                height = DEFAULT_BUILDING_HEIGHT

            base_z = self.sample_ground_z(poly)
            node = self._make_geomnode_from_extrusion(poly, height, f"building_{len(self.assets)}")
            node.setZ(base_z)

            self._add_asset(
                name=f"building_{len(self.assets)}",
                nodepath=node,
                kind="building",
                source_index=asset_idx,
                metadata={"height_m": float(height), "base_z": float(base_z)},
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
            )

    def add_water(self, water: gpd.GeoDataFrame):
        if water is None or water.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(water.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
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
            )

    def add_parks(self, parks: gpd.GeoDataFrame):
        if parks is None or parks.empty:
            return
        for asset_idx, (gdf_idx, row) in enumerate(parks.iterrows()):
            poly = _geom_to_xy_polygon(row.geometry)
            if poly is None:
                continue
            z = self.sample_ground_z(poly)
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
            )

    def export_bam(self, filename: str = "city.bam", flatten: bool = True) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename

        if flatten:
            self.root.flattenStrong()
        self.root.writeBamFile(str(out_path))
        return out_path

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