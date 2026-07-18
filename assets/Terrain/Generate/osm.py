"""
osm.py

Utilities for downloading OpenStreetMap data using OSMnx.

Author: Anthony Thompson
"""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import osmnx as ox

# ---------------------------------------------------------------------
# OSMnx settings
# ---------------------------------------------------------------------

ox.settings.use_cache = True
ox.settings.log_console = True

FEATURE_TAGS = {

    "buildings": {
        "building": True,
    },

    "water": {
        "natural": "water",
        "waterway": True,
    },

    "parks": {
        "leisure": "park",
    },

    "landuse": {
        "landuse": True,
    },

    "railways": {
        "railway": True,
    },

    "bridges": {
        "bridge": True,
    },

    "trees": {
        "natural": "tree",
    }

}


class OSMDownloader:
    """
    Downloads OpenStreetMap features and converts them into a projected CRS
    suitable for simulation (meters instead of latitude/longitude).
    """

    def __init__(
        self,
        cache_dir="cache",
        dist=1000,
        overwrite=False,
        clean_geometry=True,
        minimum_area=4.0,
    ):
        """
        Parameters
        ----------
        dist : float
            Search radius (meters) for point queries.
        """

        self.dist = dist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Project GeoDataFrame into a local UTM coordinate system."""

        if gdf.empty:
            return gdf

        return gdf.to_crs(gdf.estimate_utm_crs())

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------

    def buildings_from_place(self, place: str) -> gpd.GeoDataFrame:
        """Download building footprints for a named location."""

        gdf = ox.features_from_place(
            place,
            tags={"building": True},
        )

        return self._project(gdf)

    def buildings_from_point(
        self,
        point: Tuple[float, float],
    ) -> gpd.GeoDataFrame:
        """Download building footprints around a point."""

        gdf = ox.features_from_point(
            point,
            tags={"building": True},
            dist=self.dist,
        )

        return self._project(gdf)

    # ------------------------------------------------------------------
    # Roads
    # ------------------------------------------------------------------

    def roads_from_place(
        self,
        place: str,
        network_type: str = "drive",
    ):
        """Download road network."""

        G = ox.graph_from_place(
            place,
            network_type=network_type,
        )

        nodes, edges = ox.graph_to_gdfs(G)

        return (
            self._project(nodes),
            self._project(edges),
        )

    def roads_from_point(
        self,
        point: Tuple[float, float],
        network_type: str = "drive",
    ):
        """Download road network."""

        G = ox.graph_from_point(
            point,
            dist=self.dist,
            network_type=network_type,
        )

        nodes, edges = ox.graph_to_gdfs(G)

        return (
            self._project(nodes),
            self._project(edges),
        )

    # ------------------------------------------------------------------
    # Water
    # ------------------------------------------------------------------

    def water_from_place(self, place: str):
        tags = {
            "natural": "water",
            "waterway": True,
        }

        gdf = ox.features_from_place(place, tags)

        return self._project(gdf)

    def water_from_point(self, point):

        tags = {
            "natural": "water",
            "waterway": True,
        }

        gdf = ox.features_from_point(
            point,
            tags,
            dist=self.dist,
        )

        return self._project(gdf)

    # ------------------------------------------------------------------
    # Parks
    # ------------------------------------------------------------------

    def parks_from_place(self, place):

        tags = {
            "leisure": "park",
        }

        gdf = ox.features_from_place(place, tags)

        return self._project(gdf)

    def parks_from_point(self, point):

        tags = {
            "leisure": "park",
        }

        gdf = ox.features_from_point(
            point,
            tags,
            dist=self.dist,
        )

        return self._project(gdf)

    # ------------------------------------------------------------------
    # Generic feature query
    # ------------------------------------------------------------------

    def features_from_place(
        self,
        place: str,
        tags: dict,
    ) -> gpd.GeoDataFrame:

        gdf = ox.features_from_place(place, tags)

        return self._project(gdf)

    def features_from_point(
        self,
        point,
        tags: dict,
    ) -> gpd.GeoDataFrame:

        gdf = ox.features_from_point(
            point,
            tags,
            dist=self.dist,
        )

        return self._project(gdf)