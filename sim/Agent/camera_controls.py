from direct.showbase import DirectObject
from direct.task import Task
import numpy as np
from scipy.spatial.transform import Rotation


class CameraControls(DirectObject.DirectObject):
    #     """
    #     Class that deals with handling controls for the camera
    #     """
    pass
    #     # All arrays and vectors are in format xyz.

    def __init__(self, world):
        #         self.accept("w-repeat", self.zoom_in)
        #         self.accept("a-repeat", self.pan_left)
        #         self.accept("s-repeat", self.pan_right)
        #         self.accept("d-repeat", self.zoom_out)
        pass


#         self.world = world
#         self.world.disableMouse()  # Disables default panda3D mouse controls

#         self.mouse_x = world.mouseWatcherNode.getMouseX()
#         self.mouse_y = world.mouseWatcherNode.getMouseY()

#         self.keyboard_multiplier = 0.001
#         self.mouse_multiplier = 0.01

#         self.camera_perspective = Rotation.from_euler(
#             "xyz",
#             [self.world.camera.getH, self.world.camera.getP, self.world.camera.getR],
#             degrees=True,
#         )
#         self.location = np.array(
#             [self.world.camera.getX, self.world.camera.getY, self.world.camera.getZ]
#         )

#     def register_controls(self):
#         """
#         For regestering tasks into the thingy
#         """

#         taskMgr.add(self.change_camera_angle)

#     def change_camera_angle(self):
#         """_summary_
#         Internal task called that adjusts the camera angle based on mouse inputs.
#         Returns:
#             _type_: _description_
#         """

#         # Panda3D cannot give difference, manually calculate it
#         delta_x = self.mouse_x - self.world.mouseWatcherNode.getMouseX()
#         delta_y = self.mouse_x - self.world.mouseWatcherNode.getMouseY()

#         delta_x *= self.mouse_multiplier
#         delta_y *= self.mouse_multiplier

#         change_orentation = Rotation.from_euler("yz", [delta_x, delta_y])

#         self.world.camera.setHpr()

#         return Task.cont

#     def update_camera_angle(self):
#         """_summary_
#         Updates the internal numpy array to do calcualtions with it.
#         """

#     def pan_left(self):
#         """_summary_
#         What is called to move camera 'left' in the current perspective
#         """
#         array = np.array([-1, 0, 0])
#         self.world.camera.setPos(array.dot(self.camera_perspective))

#     def pan_right(self):
#         """_summary_
#         What is caleed to move 'right' in the current perspective
#         """

#         array = np.array([1, 0, 0])
#         self.world.camera.setPos(array.dot(self.camera_perspective))

#     def zoom_in(self):
#         """_summary_
#         Moves the camera forward in its current perspective
#         """

#         array = np.array([0, -1, 0])
#         self.world.camera.setPos(array.dot(self.camera_perspective))

#     def zoom_out(self):
#         """_summary_
#         Moves the camera in reverse from its current perspective
#         """

#         array = np.array([0, 1, 0])
#         self.world.camera.setPos(array.dot(self.camera_perspective))
