import numpy as np
import sensor as S
import estimator as kf

class agent:
    def __init__(self,state,dim,dt,sensors):
        self.x          = np.vstack(state)
        self.u          = np.zeros((2,1))
        self.dim        = dim
        self.dt         = dt
        self.sensors    = []
        # Setup the Sensor
        for sensor, params in sensors.items():
            if sensor == "bearing_sensor":
                self.sensors.append( S.bearing_sensor(num=params[0], FOV=params[1], R=params[2], cov=params[3]) )
            elif sensor == "range_bearing_sensor":
                self.sensors.append( S.range_bearing_sensor(num=params[0], FOV=params[1], R=params[2], cov=params[3]) )
            elif sensor == "range_bearing_range_rate_sensor":
                self.sensors.append( S.range_bearing_range_rate_sensor(num=params[0], FOV=params[1], R=params[2], cov=params[3]) )
        # Initialize the selected sensor 
        self.select = 0

        # Initialize the filter 
        self.initialize_filter(dim, dt)
        

    def initialize_filter(self, dim, dt):
        F = np.vstack([np.hstack([np.eye(dim),dt*np.eye(dim)]),np.hstack([np.zeros((dim,dim)),np.eye(dim)])])
        B  = np.array([[0,0],[0,0],[1,0],[0,1]])
        H  = np.array([[1,0,0,0],[0,1,0,0]])
        P0 = np.eye(2*dim)
        Q  = 1 * np.eye(2*dim)
        R  = 1 * np.eye(2)
        x0 = np.zeros((2*dim,1))
        self.KF = kf.Kalman_Filters(F, B, H, Q, R, x0, P0)
        self.output = None
        self.xt_pred = None
        self.xt_hat = None

    def eom(self, target, env):
        if self.dim==2:
            # Nonholonomic Dynamics
            self.input(target, env)
            V = self.x[3]
            th = self.x[2]
            self.x += self.dt* np.vstack( [ V*np.cos(th), V*np.sin(th), self.u[1], self.u[0] ] )
        
    def read_sensor(self,target,env):
        # Get Sensor Output
        curr_sensor = self.sensors[self.select]
        self.output = curr_sensor.output(self.x[:2], target[0].x[:2], env)

        # Filter Output
        # input_cov = 0.1*np.array([ [self.dt**3/3,self.dt**2/2],[self.dt**2/2,self.dt**3/3] ])
        # u = 1*np.random.multivariate_normal(np.zeros(self.dim), input_cov, 1)
        # self.xt_pred = self.KF.predict(u.T)
        # self.est_position = self.x[:2]+self.output[0,0] * np.array([[np.cos(self.output[1,0])],[np.sin(self.output[1,0])]])
        # self.xt_hat  = self.KF.update(self.est_position, curr_sensor.H)

    def input(self, target, env):
        self.u = np.zeros((2,1))

        # # Get Sensor Output
        # curr_sensor = self.sensors[self.select]
        # curr_sensor.output(self.x[:2], target[0].x[:2], env)

        # # Set Speed
        # range = np.linalg.norm((self.x[:2] - target[0].x[:2]))
        # range_ref = 3
        # self.u[1] = -0.1 * (range_ref - range)

        # # Set Heading
        # LOS = (self.x[:2] - target[0].x[:2])
        # LOS_ang = np.arctan2(LOS[1,0],LOS[0,0])
        # self.u[0] = 5 * np.sin(curr_sensor.y - self.x[-1,0])
        



