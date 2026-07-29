from __future__ import annotations

import argparse
from pathlib import Path
import geopandas as gpd
import math
from shapely.geometry import box

from osm import OSMDownloader
from terrain import ConstantTerrainSampler, RasterTerrainSampler
from mesh_builder import CityMeshBuilder


def _parse_point(value: str):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point must be formatted as 'lat,lon'")
    return float(parts[0].strip()), float(parts[1].strip())

def load_cached_gdf(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    if path.suffix.lower() in (".geojson", ".json"):
        return gpd.read_file(path)
    raise ValueError(f"Unsupported cache format: {path}")

def save_gdf(gdf: gpd.GeoDataFrame, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or gdf.empty:
        return None
    try:
        gdf.to_parquet(path.with_suffix(".parquet"))
        return path.with_suffix(".parquet")
    except Exception:
        gdf.to_file(path.with_suffix(".geojson"), driver="GeoJSON")
        return path.with_suffix(".geojson")
    
def load_or_query_buildings(osm: OSMDownloader, place: str | None, point, cache_dir: Path, use_cache: bool, add_elevation: bool):
    if place:
        stem = osm._gdf_cache_stem("buildings", place)
        if use_cache:
            cached = osm._load_gdf(stem)
            if cached is not None:
                return cached, stem
        gdf = osm.buildings_from_place(place, add_elevation=add_elevation, use_cache=False, save_cache=False)
        return gdf, stem

    stem = osm._gdf_cache_stem("buildings", f"point_{point[0]}_{point[1]}")
    if use_cache:
        cached = osm._load_gdf(stem)
        if cached is not None:
            return cached, stem
    gdf = osm.buildings_from_point(point, add_elevation=add_elevation, use_cache=False, save_cache=False, cache_name=f"point_{point[0]}_{point[1]}")
    return gdf, stem

def load_or_query_roads(osm: OSMDownloader, place: str | None, point, network_type: str, use_cache: bool, add_elevation: bool):
    if place:
        stem = osm._gdf_cache_stem(f"roads_{network_type}", place)
        if use_cache:
            nodes, edges = osm._load_roads(stem)
            if nodes is not None and edges is not None:
                return nodes, edges, stem
        nodes, edges = osm.roads_from_place(place, network_type=network_type, add_elevation=add_elevation, use_cache=False, save_cache=False)
        return nodes, edges, stem

    stem = osm._gdf_cache_stem(f"roads_{network_type}", f"point_{point[0]}_{point[1]}")
    if use_cache:
        nodes, edges = osm._load_roads(stem)
        if nodes is not None and edges is not None:
            return nodes, edges, stem
    nodes, edges = osm.roads_from_point(point, network_type=network_type, add_elevation=add_elevation, use_cache=False, save_cache=False, cache_name=f"point_{point[0]}_{point[1]}")
    return nodes, edges, stem

def load_or_query_feature(method_name: str, osm: OSMDownloader, place: str | None, point, use_cache: bool, add_elevation: bool, cache_key: str):
    if place:
        stem = osm._gdf_cache_stem(method_name, place)
        if use_cache:
            cached = osm._load_gdf(stem)
            if cached is not None:
                return cached, stem
        query = getattr(osm, f"{method_name}_from_place")
        gdf = query(place, add_elevation=add_elevation, use_cache=False, save_cache=False)
        return gdf, stem

    stem = osm._gdf_cache_stem(method_name, cache_key)
    if use_cache:
        cached = osm._load_gdf(stem)
        if cached is not None:
            return cached, stem
    query = getattr(osm, f"{method_name}_from_point")
    gdf = query(point, add_elevation=add_elevation, use_cache=False, save_cache=False, cache_name=cache_key)
    return gdf, stem

def _iter_tiles(bounds, tile_size: float, overlap: float = 0.0):
    xmin, ymin, xmax, ymax = bounds
    step = max(tile_size - overlap, 1e-6)

    nx = max(1, math.ceil((xmax - xmin) / step))
    ny = max(1, math.ceil((ymax - ymin) / step))

    for ix in range(nx):
        for iy in range(ny):
            x0 = xmin + ix * step
            y0 = ymin + iy * step
            x1 = min(x0 + tile_size, xmax)
            y1 = min(y0 + tile_size, ymax)
            if x1 <= x0 or y1 <= y0:
                continue
            yield ix, iy, box(x0, y0, x1, y1)

def _safe_bounds(*gdfs: gpd.GeoDataFrame):
    bounds = []
    for gdf in gdfs:
        if gdf is not None and not gdf.empty:
            bounds.append(gdf.total_bounds)

    if not bounds:
        return None

    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)
    return xmin, ymin, xmax, ymax

def _clip_gdf_to_tile(gdf: gpd.GeoDataFrame | None, tile_geom):
    if gdf is None or gdf.empty:
        return gdf
    try:
        clipped = gpd.clip(gdf, tile_geom)
        if clipped is not None and not clipped.empty:
            return clipped
    except Exception:
        pass

    try:
        mask = gdf.intersects(tile_geom)
        return gdf.loc[mask].copy()
    except Exception:
        return gdf

def _export_scene(
    builder_output_dir: Path,
    output_name: str,
    terrain,
    buildings,
    roads,
    water,
    parks,
):
    builder = CityMeshBuilder(
        terrain_sampler=terrain.height_at,
        output_dir=str(builder_output_dir),
    )

    builder.add_buildings(buildings)
    builder.add_roads(roads)
    builder.add_water(water)
    # builder.add_parks(parks)

    obj_path = builder.export_obj(output_name.replace(".bam", ".obj"))
    bam_path = builder.export_bam(output_name)
    manifest_path = builder.export_manifest_csv()

    return bam_path, obj_path, manifest_path, builder
    
def main():
    parser = argparse.ArgumentParser(description="Generate a city mesh from OSM data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--place", type=str, help="OSM place name, e.g. 'Atlanta, Georgia, USA'")
    group.add_argument("--point", type=_parse_point, help="Latitude,longitude point, e.g. '33.7490,-84.3880'")

    parser.add_argument("--dist", type=float, default=1000, help="Search radius for point queries in meters")
    parser.add_argument("--network_type", type=str, default="drive", help="OSM network type for roads")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--use_cache", action="store_true", help="Load cached GeoDataFrames if available")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cache and re-download")
    parser.add_argument("--save_each", action="store_true", help="Save each GeoDataFrame as it is fetched")
    parser.add_argument("--output_dir", type=str, default="city_output", help="Output directory")
    parser.add_argument("--output_name", type=str, default="city.glb", help="Output GLB filename")
    parser.add_argument("--dem", type=str, default=None, help="Optional DEM raster path for elevation sampling")
    parser.add_argument("--add_elevation", action="store_true", help="Attach elevation metadata from terrain/DEM")
    parser.add_argument(
        "--tile_size",
        type=float,
        default=None,
        help="Tile size in projected map units. If omitted, export a single full-city BAM.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        default=10.0,
        help="Overlap between adjacent tiles in projected map units.",
    )
    parser.add_argument(
        "--tile_prefix",
        type=str,
        default="tile",
        help="Prefix used when naming tile output folders and files.",
    )
    
    args = parser.parse_args()

    if args.dem:
        terrain = RasterTerrainSampler(args.dem)
    else:
        terrain = ConstantTerrainSampler()

    osm = OSMDownloader(
        dist=args.dist,
        elevation_sampler=terrain.height_at,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )

    place = args.place
    point = args.point
    use_cache = args.use_cache and not args.overwrite

    # output_dir = Path(Path.cwd(),"assets","Terrain","Generate",str(args.output_dir))
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if place:
        buildings, bstem = load_or_query_buildings(osm, place, point, Path(args.cache_dir), use_cache, args.add_elevation)
        nodes, roads, rstem = load_or_query_roads(osm, place, point, args.network_type, use_cache, args.add_elevation)
        water, wstem = load_or_query_feature("water", osm, place, point, use_cache, args.add_elevation, "water")
        parks, pstem = load_or_query_feature("parks", osm, place, point, use_cache, args.add_elevation, "parks")
    else:
        buildings, bstem = load_or_query_buildings(osm, None, point, Path(args.cache_dir), use_cache, args.add_elevation)
        nodes, roads, rstem = load_or_query_roads(osm, None, point, args.network_type, use_cache, args.add_elevation)
        water, wstem = load_or_query_feature("water", osm, None, point, use_cache, args.add_elevation, f"point_{point[0]}_{point[1]}")
        parks, pstem = load_or_query_feature("parks", osm, None, point, use_cache, args.add_elevation, f"point_{point[0]}_{point[1]}")

    if args.save_each:
        save_gdf(buildings, Path(output_dir,f"{bstem}.parquet"))
        save_gdf(nodes, Path(output_dir,f"{rstem}__nodes.parquet"))
        save_gdf(roads, Path(output_dir,f"{rstem}__edges.parquet"))
        save_gdf(water, Path(output_dir,f"{wstem}.parquet"))
        save_gdf(parks, Path(output_dir,f"{pstem}.parquet"))

    print("Buildings:", 0 if buildings is None else len(buildings))
    print("Road nodes:", 0 if nodes is None else len(nodes))
    print("Road edges:", 0 if roads is None else len(roads))
    print("Water:", 0 if water is None else len(water))
    print("Parks:", 0 if parks is None else len(parks))
    
    if args.tile_size is None or args.tile_size <= 0:
        builder = CityMeshBuilder(
            terrain_sampler=terrain.height_at,
            output_dir=str(output_dir),
        )

        builder.add_buildings(buildings)
        builder.add_roads(roads)
        builder.add_water(water)
        builder.add_parks(parks)
        
        builder.apply_textures(
            building_rules=[
                ("residential", Path("assets//textures//building_materials//bricks//Bricks097_1K-JPG//Bricks097_1K-JPG_Color.jpg")),
                ("retail", Path("assets//textures//building_materials//bricks//Bricks101_1K-JPG//Bricks101_1K-JPG_Color.jpg")),
                ("commercial", Path("assets//textures//building_materials//office//window-pattern-textures-building.jpg")),
            ],
            road_texture_path=Path("assets//textures//path//cement//Road007_1K-JPG//Road007.png"),
            park_texture_path=Path("assets//textures//path//grass//Grass001_1K-JPG//Grass001.png")
            water_texture_path=Path("assets//textures//water//GPT_muted_green_water.png"),
        )
        

        obj_path = builder.export_obj(Path(args.output_name).with_suffix(".obj").name)
        glb_path = builder.export_bam(args.output_name)
        manifest_path = builder.export_manifest_csv()

        print(f"Exported scene: {glb_path} and {obj_path}")
        print(f"Exported manifest: {manifest_path}")
        print(f"Generated assets: {len(builder.assets)}")
        return
    
    bounds = _safe_bounds(buildings, roads, water, parks)
    if bounds is None:
        print("No geometry found; nothing to tile.")
        return

    tile_count = 0
    total_assets = 0

    for ix, iy, tile_geom in _iter_tiles(bounds, args.tile_size, args.tile_overlap):
        tile_buildings = _clip_gdf_to_tile(buildings, tile_geom)
        tile_roads = _clip_gdf_to_tile(roads, tile_geom)
        tile_water = _clip_gdf_to_tile(water, tile_geom)
        tile_parks = _clip_gdf_to_tile(parks, tile_geom)

        has_data = any(
            gdf is not None and not gdf.empty
            for gdf in (tile_buildings, tile_roads, tile_water, tile_parks)
        )
        if not has_data:
            continue

        tile_name = f"{args.tile_prefix}_{ix:03d}_{iy:03d}"
        tile_dir = output_dir / tile_name
        tile_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"{tile_name}: "
            f"buildings={0 if tile_buildings is None else len(tile_buildings)}, "
            f"roads={0 if tile_roads is None else len(tile_roads)}, "
            f"water={0 if tile_water is None else len(tile_water)}, "
            f"parks={0 if tile_parks is None else len(tile_parks)}"
        )

        bam_path, obj_path, manifest_path, builder = _export_scene(
            builder_output_dir=tile_dir,
            output_name=f"{tile_name}.bam",
            terrain=terrain,
            buildings=tile_buildings,
            roads=tile_roads,
            water=tile_water,
            parks=tile_parks,
        )

        print(f"  Exported: {bam_path}")
        print(f"  Exported: {obj_path}")
        print(f"  Exported manifest: {manifest_path}")
        print(f"  Generated assets: {len(builder.assets)}")

        tile_count += 1
        total_assets += len(builder.assets)

    print(f"Finished exporting {tile_count} tiles.")
    print(f"Total generated assets: {total_assets}")


if __name__ == "__main__":
    main()
    """
    Examples to run:
    1. Full city:
       python main.py --place "Baltimore, Maryland, USA" --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation --use_cache

    2. Tiled export:
       python main.py --place "Baltimore, Maryland, USA" --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation --use_cache --tile_size 500

    3. Point-based location:
       python main.py --point 33.7490,-84.3880 --dist 1500 --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation

    4. With DEM data:
       python main.py --place "Atlanta, Georgia, USA" --dem atlanta_dem.tif --output_dir assets/Terrain/Generate/baltimore --output_name baltimore.bam --add_elevation
    """