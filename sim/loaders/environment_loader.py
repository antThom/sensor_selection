"""
Utilities and loader for loading terrain.
"""

from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
)

from sim.environment.terrain.atmosphere import ATMOSPHERE
from sim.environment.terrain.terrain import TERRAIN


class EnvironmentLoader:
    """
    Environment loader and manager for the simulation. Loads up the 3D models of
    the sun, sky, terrain, and other static objects into the simulation. This, so far, does
    not set up the special physics such as heat onto these objects.
    """

    def __init__(self, config_file, world):
        """Instantiates the environment loader"""
        # Boilerplate in order to derive and set up from these files
        self.config_file = config_file
        self.world = world

    def load_environment(self):
        """Actually loads the environment and attaches objects world."""

        # Set up our environment
        self.sky = ATMOSPHERE(
            loader=self.world.loader,
            config=self.config_file["atmosphere"],
            render=self.world.render,
        )
        self.terrain = TERRAIN(self.world.loader, self.config_file["terrain"])
        self.terrain.object.reparentTo(self.world.render)

    def get_terrain_height(self, position):
        """Getter for finding terrain height at a location"""

        self.cTrav = CollisionTraverser()
        self.rayQueue = CollisionHandlerQueue()

        ray = CollisionRay()
        rayNode = CollisionNode("treeRay")
        rayNode.addSolid(ray)
        rayNode.setFromCollideMask(BitMask32.bit(1))
        rayNode.setIntoCollideMask(BitMask32.allOff())
        self.rayNP = self.world.render.attachNewNode(rayNode)
        self.cTrav.addCollider(self.rayNP, self.rayQueue)

        z = self.terrain.terrain_height_at(
            x=position[0],
            y=position[1],
            ray=ray,
            render=self.world.render,
            cTrav=self.cTrav,
            rayQueue=self.rayQueue,
        )
        return z
