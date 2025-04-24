import numpy as np
import random

class preclusion:
    def __init__(self,size,x,noise,label,type):
        self.x = x
        self.radii = size
        self.noise = noise
        self.label = label
        self.type  = type

    def noise(self):
        print('noise')
        

class environment:
    def __init__(self,dim,num_preclusions,preclusion_size):
        self.dim = dim
        self.num_preclusions = num_preclusions
        self.preclusion_size = preclusion_size
        self.radii = None
        self.noise = None
        # preclusion_type = ['fog','trees','stream','snow field','blizard','desert','dust storm','dense urban','wind','rain']
        
        if num_preclusions>0:
            self.generate_preclusions()

    def generate_preclusions(self):
        # Create
        self.preclusion = []
        while True:
            labels = np.arange(self.num_preclusions)
            x = np.random.uniform(-15, 15, (self.num_preclusions, self.dim))
            radii = np.random.uniform(2, 5, self.num_preclusions)
            noise_level = np.random.uniform(0.001, 1, self.num_preclusions)
            preclusion_list = ['bearing_distort', 'range_distort','range_bearing_distort']
            preclusion_type = random.choice(preclusion_list)


            overlap = False
            for i in range(self.num_preclusions):
                for j in range(i + 1, self.num_preclusions):
                    dist = np.linalg.norm(x[i] - x[j])
                    if dist < (radii[i] + radii[j]):  # Check for overlap
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                self.preclusion.append(preclusion(size=radii,x=x,label=labels,noise=noise_level,type=preclusion_type))
                return #self.x, self.radii
            
    def in_preclusion(self,pos):
        # Determine if the agent is in the preclusion
        R = []
        if self.num_preclusions>0:
            in_pre = self.x - pos.T
            for inside, radii, noise in zip(in_pre, self.radii, self.noise):
                R.append( [np.linalg.norm(inside)<=radii,noise] )
            return R
        else:
            return R
        
