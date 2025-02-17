import numpy as np
import random

class target:
    def __init__(self,state,dim,dt):
        self.x          = np.vstack(state)
        self.dim        = dim
        self.dt         = dt
        if self.dim==2:
            self.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
            self.B = np.array([[0,0],[0,0],[1,0],[0,1]])

    def eom(self):
        u = np.random.uniform(low=-10,high=10,size=(2,1))
        self.x += self.dt * ( self.A @ self.x + self.B @ u )
            
