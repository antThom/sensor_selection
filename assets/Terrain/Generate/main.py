from __future__ import annotations

import argparse
from pathlib import Path
import geopandas as gpd

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

    output_dir = Path(Path.cwd(),"assets","Terrain","Generate",str(args.output_dir))
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
    
    builder = CityMeshBuilder(
        terrain_sampler=terrain.height_at,
        output_dir=args.output_dir,
    )

    builder.add_buildings(buildings)
    builder.add_roads(roads)
    builder.add_water(water)
    # builder.add_parks(parks)

    glb_path = builder.export_bam(args.output_name)
    manifest_path = builder.export_manifest_csv()

    print(f"Exported scene: {glb_path}")
    print(f"Exported manifest: {manifest_path}")
    print(f"Generated assets: {len(builder.assets)}")


if __name__ == "__main__":
    main()
    """ Examples to run:
    1. Place-based location: python -m citygen.main --place "Baltimore, Maryland, USA" --output_dir baltimore --output_name baltimore.bam --add_elevation --use_cache --save_each --overwrite
    2. Point-based location: python -m citygen.main --point 33.7490,-84.3880 --dist 1500 --output_dir atlanta_city --output_name atlanta.glb --add_elevation
    3. with DEM data:        python -m citygen.main --place "Atlanta, Georgia, USA" --dem atlanta_dem.tif --output_dir atlanta_city --output_name atlanta.glb --add_elevation
    
    """