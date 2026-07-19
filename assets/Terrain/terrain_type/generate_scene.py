import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import py3dep
from shapely.geometry import Polygon
import numpy as np
import pyvista as pv
import rioxarray
from pathlib import Path

ox.settings.log_console = True
ox.settings.use_cache = True

def plot_place(place, network_type="drive", default_width=4):
    img_folder = Path("City")
    extension = "png"
    dpi = 40
    fp = Path(img_folder,f"{place}.{extension}")
    G = ox.graph.graph_from_address(
        place,
        dist=1000,
        network_type=network_type,
        truncate_by_edge=True,
    )
    fig, ax = ox.plot.plot_figure_ground(
        G=G,
        dist=805,
        default_width=default_width,
        filepath=fp,
        dpi=dpi,
        save=True,
        show=False,
        close=True,
    )
    return fp

def plot_point(place, point, network_type="drive", default_width=4):
    img_folder = Path("City",f"{place}")
    extension = "png"
    dpi = 40
    fp = Path(img_folder,f"{place}.{extension}")
    G = ox.graph.graph_from_point(
        point,
        dist=1000,
        network_type=network_type,
        truncate_by_edge=True,
    )
    fig, ax = ox.plot.plot_figure_ground(
        G=G,
        dist=805,
        default_width=default_width,
        filepath=fp,
        dpi=dpi,
        save=True,
        show=False,
        close=True,
    )
    return fp

fp = plot_place("Atlanta, Georgia, USA")