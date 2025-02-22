import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SENSOR:
    def __init__(self):
        self.FOV = 0
        self.range = 0
        self.noise_var = 0

    def plot_sector(self,ax,center,theta):
        sector = patches.Wedge((center[0], center[1]), self.range, np.rad2deg(theta-0.5*self.FOV), np.rad2deg(theta+0.5*self.FOV), facecolor="green", alpha=0.5)
        ax.add_patch(sector)

class bearing_sensor(SENSOR):
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = 2*np.pi
        self.range = 2 
        self.covariance = np.pi/100
        
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
            self.noise += np.random.normal(0,self.covariance) + ( int(a_in[0]) * (np.random.normal(0,2*np.pi*a_in[1])) + int(t_in[0]) * (np.random.normal(0,2*np.pi*t_in[1])) )
        
    
class depth_bearing_sensor(SENSOR):
    def __init__(self,num):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = (np.pi/180) * 45
        self.range = 20 
        self.mean = np.array([0, 0])
        self.covariance = np.array([[1, 0], [0, np.pi/100]])
        # self.noise_var = np.pi/100

    def output(self, x_p, x_t, env):
        # Bearing
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        bearing = np.arctan2(delta_y,delta_x)

        # Range
        range = np.linalg.norm(np.hstack([delta_x,delta_y]))
        self.h = np.array([[range, bearing]])
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)

        # Linearize
        self.H = 1/range * np.array([ [delta_x, delta_y, 0, 0] , [-np.sin(bearing), np.cos(bearing), 0, 0] ])

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, range)
        self.y = np.vstack([range + self.range_noise, (bearing + self.bearing_noise) % (2*np.pi)])
        return self.y

    def added_noise(self, agent_in, target_in, R):
        # Define the mean vector and covariance matrix
        mean = np.array([0, 0])
        covariance = np.array([[5, 0], [0, 2*np.pi]])
        # If in preclusion
        self.noise = 0
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += np.random.multivariate_normal(self.mean, self.covariance, 1) + ( int(a_in[0]) * np.random.multivariate_normal(mean, covariance, 1) ) + ( int(t_in[0]) * np.random.multivariate_normal(mean, covariance, 1) )

        self.range_noise   = self.noise[0,0]
        self.bearing_noise = self.noise[0,1]