import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class bearing_sensor:
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = 2*np.pi
        self.range = 2 
        self.noise_var = np.pi/100
        
    def output(self, x_p, x_t, env):
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        bearing = np.arctan2(delta_y,delta_x)
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, np.linalg.norm(np.hstack([delta_y,delta_x])))
        
        self.y = bearing + self.noise
    
    def added_noise(self, agent_in, target_in, R):
        # If in preclusion
        self.noise = 0
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += np.random.normal(0,self.noise_var) + ( int(a_in[0]) * (np.random.normal(0,2*np.pi*a_in[1])) + int(t_in[0]) * (np.random.normal(0,2*np.pi*t_in[1])) )
        




    def plot_sector(self,ax,center):
        sector = patches.Wedge((center[0], center[1]), self.range, 0, self.FOV*(180/np.pi), facecolor="green", alpha=0.5)
        ax.add_patch(sector)
    
class camera:
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = 2*np.pi
        self.range = 15 