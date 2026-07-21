from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ConstantTerrainSampler:
    def height_at(self, x: float, y: float) -> float:
        return 0.0


class RasterTerrainSampler:
    def __init__(self, raster_path: str):
        import rasterio
        self.ds = rasterio.open(raster_path)

    def height_at(self, x: float, y: float) -> float:
        row, col = self.ds.index(x, y)
        val = self.ds.read(1)[row, col]
        if np.isnan(val):
            return 0.0
        return float(val)