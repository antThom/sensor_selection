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

    def eom(self,target):
        if self.dim==2:
            # Nonholonomic Dynamics
            self.input(target)
            V = self.x[2]
            th = self.x[3]
            self.x += self.dt* np.vstack( [ V*np.cos(th), V*np.sin(th), self.u[1], self.u[0] ] )

    def input(self,target):
        # u = np.zeros((2,1))
        # Set Speed
        range = np.linalg.norm((self.x[:2] - target[0].x[:2]))
        range_ref = 3
        self.u[1] = 1 * (range_ref - range)

        # Set Heading
        LOS = (self.x[:2] - target[0].x[:2])
        LOS_ang = np.arctan2(LOS[1,0],LOS[0,0])
        self.u[0] = 1 * np.sin(LOS_ang - self.x[-1])



