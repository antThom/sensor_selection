"""
view_bam.py

Simple viewer for Panda3D .bam files.

Usage:
    python view_bam.py city.bam
"""

import argparse
from pathlib import Path
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Vec4,
    Vec3,
    NodePath,
    CardMaker,
    Filename,
)
from math import pi, sin, cos, radians
from direct.task import Task



class BamViewer(ShowBase):

    def __init__(self, bam_file):
        ShowBase.__init__(self)

        # self.disableMouse()
        self.setBackgroundColor(0.2, 0.2, 0.2, 1)
        
        # Lighting
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.6, 0.6, 0.6, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1, 1, 1, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -60, 0)
        self.render.setLight(sun_np)
        
        bam_file = Path(bam_file).resolve()
        print(f"Loading: {bam_file}")
        print(f"Exists: {bam_file.exists()}")
        if not bam_file.exists():
            raise FileNotFoundError(bam_file)
        panda_path = Filename.fromOsSpecific(str(bam_file))
        print(type(panda_path), panda_path)
        with open(r"assets\Terrain\Generate\baltimore\tile_008_006\tile_008_006.bam", "rb") as f:
            print(f.read(16))
        city = self.loader.loadModel(panda_path)
        if city.isEmpty():
            raise RuntimeError(f"Failed to load model: {bam_file}")
        city.setTwoSided(True)
        city.reparentTo(self.render)        
        city.setScale(0.50)
        city.setPos(0, 0, -50)
        self.render.setTwoSided(True)
        
        # Ground
        # cm = CardMaker("ground")
        # cm.setFrame(-100, 100, -100, 100)
        # ground = self.render.attachNewNode(cm.generate())
        # ground.setP(-90)
        # ground.setColor(0.35, 0.35, 0.35, 1)
        # ground.setTwoSided(True)

        # Simple object
        # cube = self.loader.loadModel("models/misc/rgbCube")
        # cube.reparentTo(self.render)
        # cube.setScale(5)
        # cube.setPos(0, 0, 5)

        # Camera
        self.camera.setPos(100, 0, 1200)
        self.camera.lookAt(0, 0, 0)
        
        
        
        

        # self.accept("wheel_up", self.zoom_in)
        # self.accept("wheel_down", self.zoom_out)
        # Add the spinCameraTask procedure to the task manager.
        # Frame the scene with a camera above it looking down
        # self._frame_scene(
        #     heading_deg=45.0,   # rotate around the scene
        #     pitch_deg=55.0,     # how steeply the camera looks down
        #     distance_scale=1.8, # farther = more of the city visible
        # )

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.scene.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 30)
        self.scene.setHpr(angleDegrees, 0, 0)
        return Task.cont
    
    def _add_lights(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.65, 0.65, 0.65, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.0, 1.0, 1.0, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(0, 90, 0)  # direction of the light
        self.render.setLight(sun_np)
        
    def _frame_scene(self, heading_deg=45.0, pitch_deg=55.0, distance_scale=1.8):
        bounds = self.scene.getTightBounds()
        if not bounds:
            # Fallback if bounds are unavailable
            center = Vec3(0, 0, 0)
            radius = 1000.0
        else:
            min_b, max_b = bounds
            center = (min_b + max_b) * 0.5
            radius = (max_b - min_b).length() * 0.5
            radius = max(radius, 1.0)

        # Distance chosen from model size
        dist = radius * distance_scale

        # Convert to radians
        heading = radians(heading_deg)
        pitch = radians(pitch_deg)

        # Camera position: above and offset from the scene center
        x = center.x + dist * cos(heading) * cos(pitch)
        y = center.y + dist * sin(heading) * cos(pitch)
        z = center.z + dist * sin(pitch)

        self.camera.setPos(x, y, z)
        self.camera.lookAt(center)

        # Increase clipping range for large city models
        self.camLens.setNearFar(1.0, max(100000.0, dist * 20.0))

    def zoom_in(self):
        self.camera.setY(self.camera, 10)

    def zoom_out(self):
        self.camera.setY(self.camera, -10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bam",
        help="bam file",
        default="assets//Terrain//Generate//baltimore//baltimore.bam"
    )

    args = parser.parse_args()

    app = BamViewer(args.bam)
    app.run()