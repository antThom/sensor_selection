import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class bearing_sensor:
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = 2*np.pi
        self.range = 10 
        
    def output(self,x_p,x_t):
        delta_y = x_t[1]-x_p[1]
        delta_x = x_t[0]-x_p[0]
        bearing = np.arctan2(delta_y,delta_x)
        return bearing
    
    def plot_sector(self,ax,center):
        sector = patches.Wedge((center[0], center[1]), self.range, 0, self.FOV*(180/np.pi), facecolor="green", alpha=0.5)
        ax.add_patch(sector)
    
class camera:
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = 2*np.pi
        self.range = 15 