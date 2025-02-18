import numpy as np
import sensor as S

class agent:
    def __init__(self,state,dim,dt,sensors):
        self.x          = np.vstack(state)
        self.u          = np.zeros((2,1))
        self.dim        = dim
        self.dt         = dt
        self.sensors    = []
        for sensor, quanity in sensors.items():
            if sensor == "bearing_sensor":
                self.sensors.append( S.bearing_sensor(quanity) )

        self.select = 0
        self.t_A         = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.t_B         = np.array([[0,0],[0,0],[1,0],[0,1]])
        self.t_C         = np.eye(1,2*dim)
        self.cov         = np.eye(2*dim)
        self.Q           = 10 * np.eye(2*dim)
        self.R           = np.eye(2)
        self.xhat   = np.zeros((2*dim,1))

    def eom(self, target, env):
        if self.dim==2:
            # Nonholonomic Dynamics
            self.input(target, env)
            V = self.x[2]
            th = self.x[3]
            self.x += self.dt* np.vstack( [ V*np.cos(th), V*np.sin(th), self.u[1], self.u[0] ] )

    def ekf(self, curr_sensor):
        # Predict
        predicted_u = np.random.normal(0,20,(2,1))
        self.xhat = self.t_A @ self.xhat + self.t_B @ predicted_u
        self.cov  = self.t_A @ self.cov @ self.t_A.T + self.Q

        # Update
        zk = curr_sensor.y
        self.yhat_k = zk - self.t_C @ self.xhat_pred
        self.Sk     = self.t_C @ self.cov @ self.t_C.T + self.R
        self.Kk     = self.cov @ self.t_C.T @ np.invert(self.Sk)
        self.xhat_k = self.xhat_pred + self.Kk @ self.yhat_k
        self.cov    = (np.eye(self.cov.shape) - self.Kk @ self.t_C) @ self.cov
        self.yhat   = zk - self.t_C @ self.xhat_k
        self.xhat   = self.xhat + self.Kk @ zk


    def input(self, target, env):
        # Get Sensor Output
        curr_sensor = self.sensors[self.select]
        curr_sensor.output(self.x[:2], target[0].x[:2], env)

        # Kalman Filter
        # self.ekf(curr_sensor)

        # Set Speed
        range = np.linalg.norm((self.x[:2] - target[0].x[:2]))
        range_ref = 3
        self.u[1] = -0.1 * (range_ref - range)

        # Set Heading
        LOS = (self.x[:2] - target[0].x[:2])
        LOS_ang = np.arctan2(LOS[1,0],LOS[0,0])
        self.u[0] = 5 * np.sin(curr_sensor.y - self.x[-1,0])
        



