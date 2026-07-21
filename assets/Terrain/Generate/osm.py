from __future__ import annotations

"""osm.py

Utilities for downloading OpenStreetMap data using OSMnx.
Includes place->point fallback and lightweight on-disk caching for GeoDataFrames.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import re

import geopandas as gpd
import osmnx as ox
import pandas as pd


# ---------------------------------------------------------------------
# OSMnx settings
# ---------------------------------------------------------------------

ox.settings.use_cache = True
ox.settings.log_console = True


FEATURE_TAGS = {
    "buildings": {"building": True},
    "water": {"natural": "water", "waterway": True},
    "parks": {"leisure": "park"},
    "landuse": {"landuse": True},
    "railways": {"railway": True},
    "bridges": {"bridge": True},
    "trees": {"natural": "tree"},
}


@dataclass
class OSMCityBundle:
    buildings: gpd.GeoDataFrame
    road_nodes: gpd.GeoDataFrame
    road_edges: gpd.GeoDataFrame
    bridges: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    parks: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame
    railways: gpd.GeoDataFrame
    trees: gpd.GeoDataFrame


class OSMDownloader:
    """Download and cache OSM features in projected CRS (meters).

    The downloader tries a place query first and falls back to a point query
    if the place-based request fails. Results can optionally be cached to disk
    so later runs can skip the OSM request entirely.
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        dist: float = 1000.0,
        overwrite: bool = False,
        clean_geometry: bool = True,
        minimum_area: float = 4.0,
        elevation_sampler: Optional[Callable[[float, float], float]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dist = float(dist)
        self.overwrite = bool(overwrite)
        self.clean_geometry = bool(clean_geometry)
        self.minimum_area = float(minimum_area)
        self.elevation_sampler = elevation_sampler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "query"

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty or gdf.crs is None:
            return gdf
        return gdf.to_crs(gdf.estimate_utm_crs())

    def _clean(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty or not self.clean_geometry:
            return gdf
        out = gdf.copy()
        out = out[~out.geometry.isna()]
        out = out[~out.geometry.is_empty]
        out = out[out.geometry.is_valid]
        return out

    def _filter_min_area(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty:
            return gdf
        out = gdf.copy()
        out["geom_area"] = out.geometry.area
        return out[out["geom_area"] >= self.minimum_area]

    @staticmethod
    def _parse_height(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            s = str(value).strip().lower()
            s = s.replace("meters", "").replace("meter", "").replace("m", "").strip()
            return float(s)
        except Exception:
            return None

    def _geometry_sample_point(self, geom):
        if geom is None or geom.is_empty:
            return None
        gtype = geom.geom_type
        if gtype in ("Polygon", "MultiPolygon"):
            return geom.representative_point()
        if gtype in ("LineString", "MultiLineString"):
            return geom.interpolate(0.5, normalized=True)
        if gtype == "Point":
            return geom
        return geom.centroid

    def _sample_elevation(self, x: float, y: float) -> Optional[float]:
        if self.elevation_sampler is None:
            return None
        try:
            return float(self.elevation_sampler(x, y))
        except Exception:
            return None

    def _add_elevation_column(self, gdf: gpd.GeoDataFrame, col_name: str = "ground_z") -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty:
            return gdf
        out = gdf.copy()
        zs = []
        for geom in out.geometry:
            pt = self._geometry_sample_point(geom)
            if pt is None:
                zs.append(None)
            else:
                zs.append(self._sample_elevation(pt.x, pt.y))
        out[col_name] = zs
        return out

    def _building_height_from_row(self, row: pd.Series) -> Optional[float]:
        h = self._parse_height(row.get("height", None))
        if h is not None:
            return h
        levels = row.get("building:levels", None)
        try:
            if levels is not None and not pd.isna(levels):
                return float(levels) * 3.0
        except Exception:
            pass
        return None

    def _add_building_metadata(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty:
            return gdf
        out = gdf.copy()
        if "geom_area" not in out.columns:
            out["geom_area"] = out.geometry.area
        out["height_m"] = out.apply(self._building_height_from_row, axis=1)
        out["height_m"] = out["height_m"].fillna(8.0)
        return out

    def _cache_paths(self, stem: str, suffix: str) -> Path:
        return self.cache_dir / f"{stem}{suffix}"

    def _gdf_cache_stem(self, kind: str, name: str) -> str:
        return f"{kind}__{self._slugify(name)}"

    def _save_gdf(self, gdf: gpd.GeoDataFrame, stem: str) -> Optional[Path]:
        if gdf is None:
            return None
        if gdf.empty:
            return None

        # Prefer parquet, fall back to geojson if parquet dependencies are unavailable.
        pq_path = self._cache_paths(stem, ".parquet")
        gj_path = self._cache_paths(stem, ".geojson")

        try:
            gdf.to_parquet(pq_path)
            return pq_path
        except Exception:
            try:
                gdf.to_file(gj_path, driver="GeoJSON")
                return gj_path
            except Exception:
                return None

    def _load_gdf(self, stem: str) -> Optional[gpd.GeoDataFrame]:
        pq_path = self._cache_paths(stem, ".parquet")
        gj_path = self._cache_paths(stem, ".geojson")

        if pq_path.exists():
            try:
                return gpd.read_parquet(pq_path)
            except Exception:
                pass

        if gj_path.exists():
            try:
                return gpd.read_file(gj_path)
            except Exception:
                pass

        return None

    def _save_roads(self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, stem: str) -> None:
        self._save_gdf(nodes, f"{stem}__nodes")
        self._save_gdf(edges, f"{stem}__edges")

    def _load_roads(self, stem: str):
        nodes = self._load_gdf(f"{stem}__nodes")
        edges = self._load_gdf(f"{stem}__edges")
        if nodes is None or edges is None:
            return None, None
        return nodes, edges

    def _query_features_place_or_point(
        self,
        place: str,
        tags: dict,
        dist: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        try:
            return ox.features_from_place(place, tags)
        except Exception as e:
            print(f"[WARN] features_from_place failed for '{place}': {e}")
            try:
                point = ox.geocode(place)
                return ox.features_from_point(point, tags, dist=self.dist if dist is None else dist)
            except Exception as e2:
                print(f"[ERROR] features_from_point fallback failed for '{place}': {e2}")
                return gpd.GeoDataFrame()

    def _query_graph_place_or_point(self, place: str, network_type: str = "drive"):
        try:
            return ox.graph_from_place(place, network_type=network_type, simplify=True)
        except Exception as e:
            print(f"[WARN] graph_from_place failed for '{place}': {e}")
            try:
                point = ox.geocode(place)
                return ox.graph_from_point(point, dist=self.dist, network_type=network_type, simplify=True)
            except Exception as e2:
                print(f"[ERROR] graph_from_point fallback failed for '{place}': {e2}")
                return None

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------

    def buildings_from_place(
        self,
        place: str,
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("buildings", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"building": True})
        gdf = self._project(self._clean(gdf))
        gdf = self._filter_min_area(gdf)
        gdf = self._add_building_metadata(gdf)
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def buildings_from_point(
        self,
        point: Tuple[float, float],
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("buildings", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = ox.features_from_point(point, tags={"building": True}, dist=self.dist)
        gdf = self._project(self._clean(gdf))
        gdf = self._filter_min_area(gdf)
        gdf = self._add_building_metadata(gdf)
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Roads
    # ------------------------------------------------------------------

    def roads_from_place(
        self,
        place: str,
        network_type: str = "drive",
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
    ):
        stem = self._gdf_cache_stem(f"roads_{network_type}", place)
        if use_cache and not self.overwrite:
            nodes, edges = self._load_roads(stem)
            if nodes is not None and edges is not None:
                return nodes, edges

        G = self._query_graph_place_or_point(place, network_type=network_type)
        if G is None:
            empty = gpd.GeoDataFrame()
            return empty, empty

        nodes, edges = ox.graph_to_gdfs(G)
        nodes = self._project(self._clean(nodes))
        edges = self._project(self._clean(edges))
        if add_elevation:
            nodes = self._add_elevation_column(nodes, col_name="ground_z")
            edges = self._add_elevation_column(edges, col_name="ground_z")
        if save_cache:
            self._save_roads(nodes, edges, stem)
        return nodes, edges

    def roads_from_point(
        self,
        point: Tuple[float, float],
        network_type: str = "drive",
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ):
        stem = self._gdf_cache_stem(f"roads_{network_type}", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            nodes, edges = self._load_roads(stem)
            if nodes is not None and edges is not None:
                return nodes, edges

        try:
            G = ox.graph_from_point(point, dist=self.dist, network_type=network_type, simplify=True)
        except Exception as e:
            print(f"[ERROR] graph_from_point failed for {point}: {e}")
            empty = gpd.GeoDataFrame()
            return empty, empty

        nodes, edges = ox.graph_to_gdfs(G)
        nodes = self._project(self._clean(nodes))
        edges = self._project(self._clean(edges))
        if add_elevation:
            nodes = self._add_elevation_column(nodes, col_name="ground_z")
            edges = self._add_elevation_column(edges, col_name="ground_z")
        if save_cache:
            self._save_roads(nodes, edges, stem)
        return nodes, edges

    # ------------------------------------------------------------------
    # Bridges
    # ------------------------------------------------------------------

    def bridges_from_place(
        self,
        place: str,
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("bridges", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"bridge": True})
        gdf = self._project(self._clean(gdf))
        if gdf is None or gdf.empty:
            return gdf
        if "bridge" in gdf.columns:
            mask = gdf["bridge"].fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
            gdf = gdf[mask]
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def bridges_from_point(
        self,
        point: Tuple[float, float],
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("bridges", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, tags={"bridge": True}, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] bridges_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if gdf is None or gdf.empty:
            return gdf
        if "bridge" in gdf.columns:
            mask = gdf["bridge"].fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
            gdf = gdf[mask]
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Water
    # ------------------------------------------------------------------

    def water_from_place(
        self,
        place: str,
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
    ) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("water", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"natural": "water", "waterway": True})
        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def water_from_point(
        self,
        point,
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ):
        stem = self._gdf_cache_stem("water", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, {"natural": "water", "waterway": True}, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] water_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Parks
    # ------------------------------------------------------------------

    def parks_from_place(self, place, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True):
        stem = self._gdf_cache_stem("parks", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"leisure": "park"})
        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def parks_from_point(self, point, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True, cache_name: Optional[str] = None):
        stem = self._gdf_cache_stem("parks", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, {"leisure": "park"}, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] parks_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Railways
    # ------------------------------------------------------------------

    def railways_from_place(self, place, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True):
        stem = self._gdf_cache_stem("railways", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"railway": True})
        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def railways_from_point(self, point, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True, cache_name: Optional[str] = None):
        stem = self._gdf_cache_stem("railways", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, {"railway": True}, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] railways_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Trees
    # ------------------------------------------------------------------

    def trees_from_place(self, place, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True):
        stem = self._gdf_cache_stem("trees", place)
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, {"natural": "tree"})
        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def trees_from_point(self, point, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True, cache_name: Optional[str] = None):
        stem = self._gdf_cache_stem("trees", cache_name or f"point_{point[0]}_{point[1]}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, {"natural": "tree"}, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] trees_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # Generic feature query
    # ------------------------------------------------------------------

    def features_from_place(self, place: str, tags: dict, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True, cache_name: Optional[str] = None) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("features", cache_name or f"{place}_{sorted(tags.items())}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        gdf = self._query_features_place_or_point(place, tags)
        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    def features_from_point(self, point, tags: dict, add_elevation: bool = False, use_cache: bool = True, save_cache: bool = True, cache_name: Optional[str] = None) -> gpd.GeoDataFrame:
        stem = self._gdf_cache_stem("features", cache_name or f"point_{point[0]}_{point[1]}_{sorted(tags.items())}")
        if use_cache and not self.overwrite:
            cached = self._load_gdf(stem)
            if cached is not None:
                return cached

        try:
            gdf = ox.features_from_point(point, tags, dist=self.dist)
        except Exception as e:
            print(f"[ERROR] features_from_point failed for {point}: {e}")
            return gpd.GeoDataFrame()

        gdf = self._project(self._clean(gdf))
        if add_elevation:
            gdf = self._add_elevation_column(gdf, col_name="ground_z")
        if save_cache:
            self._save_gdf(gdf, stem)
        return gdf

    # ------------------------------------------------------------------
    # City bundle helpers
    # ------------------------------------------------------------------

    def city_bundle_from_place(
        self,
        place: str,
        network_type: str = "drive",
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
    ) -> OSMCityBundle:
        buildings = self.buildings_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        road_nodes, road_edges = self.roads_from_place(place, network_type=network_type, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        bridges = self.bridges_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        water = self.water_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        parks = self.parks_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        landuse = self.features_from_place(place, tags={"landuse": True}, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=f"landuse__{place}")
        railways = self.railways_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)
        trees = self.trees_from_place(place, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache)

        return OSMCityBundle(
            buildings=buildings,
            road_nodes=road_nodes,
            road_edges=road_edges,
            bridges=bridges,
            water=water,
            parks=parks,
            landuse=landuse,
            railways=railways,
            trees=trees,
        )

    def city_bundle_from_point(
        self,
        point: Tuple[float, float],
        network_type: str = "drive",
        add_elevation: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_name: Optional[str] = None,
    ) -> OSMCityBundle:
        suffix = cache_name or f"point_{point[0]}_{point[1]}"
        buildings = self.buildings_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        road_nodes, road_edges = self.roads_from_point(point, network_type=network_type, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        bridges = self.bridges_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        water = self.water_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        parks = self.parks_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        landuse = self.features_from_point(point, tags={"landuse": True}, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=f"landuse__{suffix}")
        railways = self.railways_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)
        trees = self.trees_from_point(point, add_elevation=add_elevation, use_cache=use_cache, save_cache=save_cache, cache_name=suffix)

        return OSMCityBundle(
            buildings=buildings,
            road_nodes=road_nodes,
            road_edges=road_edges,
            bridges=bridges,
            water=water,
            parks=parks,
            landuse=landuse,
            railways=railways,
            trees=trees,
        )