from direct.showbase import DirectObject, ShowBase
from direct.task import Task
import numpy as np
from scipy.spatial.transform import Rotation
from panda3d.core import PNMImage, PNMImageHeader, DisplayRegion
import datetime
from PIL import Image
import os
 

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


        self.world = world
        self.camera_index = 0
        self.old_index = -1

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

    ## Event methods to call when changing the camera

    # """ Keeps track of camera index """

    def camera_list_forward(self) -> None:
        """ Switches the camera to the next camera in the list"""
        
       # print(f"Camera List Forward! Current Camera Index is:{self.camera_index}. Current Old Index is{self.old_index}" )
        self.old_index = self.camera_index
        self.camera_index += 1
        self.change_camera(self.camera_index, self.old_index)
        #print(f"Changed! Current Camera Index is:{self.camera_index}. Current Old Index is{self.old_index}" )

    def camera_list_back(self) -> None:
        """ Switches the camera in use to the previous camera in the list"""
       # print(f"Camera List Backwards! Current Camera Index is:{self.camera_index}. Current Old Index is{self.old_index}" )
        self.old_index = self.camera_index
        self.camera_index -= 1
        self.change_camera(self.camera_index, self.old_index)
      #  print(f"Changed! Current Camera Index is:{self.camera_index}. Current Old Index is{self.old_index}" )
        


    def change_camera(self, index: int, old_index=None) -> None:
        """_summary_
        Switches the currently rendering camera in `world.camera_list` to the index given
        Args:
            index (int): index of camera in list
        """
        
        # Fix going over and under
        self.camera_index = abs(index % len(self.world.camera_list))
        self.old_index = abs(old_index % len(self.world.camera_list))
            
        # Index 0 will always be the base camera class,
        # which does not have a display region and will always render.
        # Do nothing for default base camera
            
        if self.old_index == 0:
            base.camera.detachNode()
        elif self.old_index:
            self.world.camera_list[self.old_index].display_region.setActive(False)

        if self.camera_index == 0:    
            base.camera.reparentTo(self.world.render)
        else:
            self.world.camera_list[self.camera_index].display_region.setActive(True)

    def save_current_camera_image(self, camera_list_index: int) -> None:
        """_summary_
        Takes in the current camera view index, and from it's DisplayRegion, capture a PMN Image.
        Args:
            camera_list_index (int): _description_
        """
        
        print(f"Print screen was called on {camera_list_index}")
        print(f"Index was {self.camera_index}")
        print(f"List is {self.world.camera_list}")
        
        if camera_list_index == 0:
            image_to_save = PNMImage()            
            base.win.getScreenshot(image_to_save)
            result = image_to_save.write("\\logs\\buffer\\buffer.ppm")
        if camera_list_index != 0: # Might grab default camera, which has non of our api!    
            current_display = self.world.camera_list[camera_list_index].display_region

            image_to_save = PNMImage()            
            current_display.getScreenshot(image_to_save)
            result = image_to_save.write("\\logs\\buffer\\buffer.ppm")
            print(result)
        
        export_image_buffer(f"{datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")}.png")
            
            
            
def export_image_buffer(filename: str, file_format="PNG") -> None:
    """_summary_
    Exports the most recent buffer into a file in the /log directory.
    Args:
        filename (str): The filename to name the file (with the file ending)
        file_format (str): A Pillow 
    """
    with Image.open(os.path.join(".", "logs", "buffer", "buffer.ppm")) as im:
        im.save(os.path.join(".", "logs", filename), file_format)