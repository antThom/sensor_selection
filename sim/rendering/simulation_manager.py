"""
File for the class `SimulationManager`,
the handler for interfacing and adding objects to panad3d
"""

from typing import Any

from direct.showbase.ShowBase import ShowBase
from sim.rendering.renderable_object import RenderableObjectBuilder


class SimulationManager:
    """Handler for simulation utilities. Modifies groups of nodes for tasks"""

    def __init__(self, show_base: ShowBase):
        self.world = show_base
        self.renderable_builder = RenderableObjectBuilder(self.world)
        
    # Empty for now.
    # Initalization of objects were moved to the `RenderableObjectBuilder()`
