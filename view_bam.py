"""
view_bam.py

Simple viewer for Panda3D .bam files.

Usage:
    python view_bam.py city.bam
"""

import argparse

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Vec4,
    Vec3,
    NodePath,
)


class BamViewer(ShowBase):

    def __init__(self, bam_file):
        ShowBase.__init__(self)

        # self.disableMouse()

        self.scene = self.loader.loadModel(bam_file)
        self.scene.setTwoSided(True)
        self.scene.reparentTo(self.render)
        # subset = NodePath("subset")
        # xmin, xmax = 0, 500
        # ymin, ymax = 0, 500

        # for np in self.scene.findAllMatches("**/building_*"):
        #     p = np.getPos(self.scene)
        #     if xmin <= p.x <= xmax and ymin <= p.y <= ymax:
        #         np.reparentTo(subset)
        # subset.reparentTo(self.render)
                
        # for np in self.scene.findAllMatches("**/road_*"):
        #             p = np.getPos(self.scene)
        #             if xmin <= p.x <= xmax and ymin <= p.y <= ymax:
        #                 np.reparentTo(subset)

        # subset.reparentTo(self.render)

        

        # Center model
        center = self.scene.getBounds().getCenter()
        radius = self.scene.getBounds().getRadius()

        self.scene.setPos(-center)

        self.camera.setPos(
            0,
            -radius * 2.5,
            3*radius,
        )
        self.camera.lookAt(0, 0, radius * 0.25)
        self.render.setTwoSided(True)

        #
        # Lighting
        #

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.45, 0.45, 0.45, 1))
        self.render.setLight(
            self.render.attachNewNode(ambient)
        )

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1, 1, 1, 1))

        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -45, 0)

        self.render.setLight(sun_np)

        #
        # Mouse orbit
        #

        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)

    def zoom_in(self):
        self.camera.setY(self.camera, 10)

    def zoom_out(self):
        self.camera.setY(self.camera, -10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bam",
        help="bam file",
        default="assets//Terrain//Generate//baltimore//tile_097_217//tile_097_217.bam"
    )

    args = parser.parse_args()

    app = BamViewer(args.bam)
    app.run()