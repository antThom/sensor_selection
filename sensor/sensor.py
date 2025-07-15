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

class CAMERA:
    def __init__(self):
        self.fx
        self.fy
        self.cx
        self.cyS

class bearing_sensor(SENSOR):
    def __init__(self,num,FOV,R,cov):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = FOV
        self.range = R
        self.mean = 0
        self.covariance = np.pi/100
        
    def truth_output(self, x_p, x_t):
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        self.h = np.arctan2(delta_y,delta_x)

    def output(self, x_p, x_t, env):
        # Truth Output
        self.truth_output(x_p, x_t)
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, np.linalg.norm(np.hstack([delta_y,delta_x])))
        
        self.y = self.h + self.noise
    
    def added_noise(self, agent_in, target_in, R):
        # If in preclusion
        self.noise = 0
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += np.random.normal(0,self.covariance) + ( int(a_in[0]) * (np.random.normal(0,2*np.pi*a_in[1])) + int(t_in[0]) * (np.random.normal(0,2*np.pi*t_in[1])) )
            
class range_bearing_sensor(SENSOR):
    def __init__(self,num,FOV,R,cov):
        self.num_sensor = num
        self.num_output = 2
        self.FOV = FOV
        self.range = R
        self.mean = np.array([0, 0])
        self.covariance = cov
        # self.noise_var = np.pi/100

    def truth_output(self, x_p, x_t):
        # Bearing
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        bearing = np.arctan2(delta_y,delta_x)

        # Range
        range = np.linalg.norm(np.hstack([delta_x,delta_y]))
        self.h = np.array([[range, bearing]]).reshape((2,1))

    def output(self, x_p, x_t, env):
        # Truth Output
        self.truth_output(x_p, x_t)
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)

        # Linearize
        # self.H = 1/range * np.array([ [delta_x, delta_y, 0, 0] , [-np.sin(bearing), np.cos(bearing), 0, 0] ])

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, self.h[0,0])
        self.y = np.vstack([self.h[0,0] + self.range_noise, (self.h[1,0] + self.bearing_noise) % (2*np.pi)]).reshape((2,1))
        return self.y

    def added_noise(self, agent_in, target_in, R):
        # Define the mean vector and covariance matrix (THESE SHOULD BE IN ENVIRONMENT)
        mean = np.array([0, 0])
        covariance = np.array([[5, 0], [0, 2*np.pi]])
        # If in preclusion
        self.noise = np.random.multivariate_normal(self.mean, self.covariance, 1)
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += ( int(a_in[0]) * np.random.multivariate_normal(mean, covariance, 1) ) + ( int(t_in[0]) * np.random.multivariate_normal(mean, covariance, 1) )

        self.range_noise   = self.noise[0,0]
        self.bearing_noise = self.noise[0,1]

class direction_sensor(SENSOR):
    def __init__(self,num,FOV,R,cov):
        self.num_sensor = num
        self.num_output = 1
        self.FOV = FOV
        self.range = R
        self.mean = 0
        self.covariance = np.pi/100
        
    def truth_output(self, x_p, x_t):
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        self.h = np.arctan2(delta_y,delta_x)

    def output(self, x_p, x_t, env):
        # Truth Output
        self.truth_output(x_p, x_t)
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, np.linalg.norm(np.hstack([delta_y,delta_x])))
        
        self.y = self.h + self.noise
    
    def added_noise(self, agent_in, target_in, R):
        # If in preclusion
        self.noise = 0
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += np.random.normal(0,self.covariance) + ( int(a_in[0]) * (np.random.normal(0,2*np.pi*a_in[1])) + int(t_in[0]) * (np.random.normal(0,2*np.pi*t_in[1])) ) 

class range_bearing_range_rate_sensor(SENSOR):
    def __init__(self,num,FOV,R,cov):
        self.num_sensor = num
        self.num_output = 3
        self.FOV = FOV
        self.range = R
        self.mean = np.zeros((self.num_output,1))
        self.covariance = cov
        # self.noise_var = np.pi/100

    def truth_output(self, x_p, x_t):
        # Bearing
        delta_y = x_t[1,0]-x_p[1,0]
        delta_x = x_t[0,0]-x_p[0,0]
        bearing = np.arctan2(delta_y,delta_x)

        # Range
        range = np.linalg.norm(np.hstack([delta_x,delta_y]))

        # Range Rate
        delta_vx = x_t[2,0]-x_p[2,0]
        delta_vy = x_t[3,0]-x_p[3,0]
        rel_pos = np.vstack([delta_x,delta_y])
        rel_vel = np.vstack([delta_vx,delta_vy])
        if range>0:
            range_rate = np.dot(rel_pos,rel_vel)/range
        else:
            range_rate = 0
        self.h = np.array([[range, range_rate, bearing]]).reshape((self.num_output,1))

    def output(self, x_p, x_t, env):
        # Truth Output
        self.truth_output(x_p, x_t)
        
        # Check if agent and target are in preclusion
        agent_in_preclusion  = env.in_preclusion(x_p)
        targer_in_preclusion = env.in_preclusion(x_t)     

        # Add Noise
        self.added_noise(agent_in_preclusion, targer_in_preclusion, self.h[0,0])
        self.y = np.vstack([self.h[0,0] + self.range_noise, self.h[1,0] + self.range_rate_noise, (self.h[2,0] + self.bearing_noise) % (2*np.pi)]).reshape((2,1))
        return self.y

    def added_noise(self, agent_in, target_in, R):
        # Define the mean vector and covariance matrix
        mean = np.zeros((self.num_output,1))
        covariance = np.array([[5, 0, 0], [0, 10, 0], [0, 0, 2*np.pi]])
        # If in preclusion
        self.noise = np.random.multivariate_normal(self.mean, self.covariance, 1)
        for a_in, t_in in zip(agent_in,target_in):
            self.noise += ( int(a_in[0]) * np.random.multivariate_normal(mean, covariance, 1) ) + ( int(t_in[0]) * np.random.multivariate_normal(mean, covariance, 1) )

        self.range_noise      = self.noise[0,0]
        self.bearing_noise    = self.noise[0,2]
        self.range_rate_noise = self.noise[0,1]



def random_diagonal_matrix(data_type):
    size = len(data_type)
    entry = []
    for i in data_type:
        if i == 'range':
            entry.append(1.5*np.random.rand())
        elif i =='bearing':
            entry.append(np.pi/6*np.random.rand())
        elif i =='range_rate':
            entry.append(0.5*np.random.rand())
    return np.diag(entry)