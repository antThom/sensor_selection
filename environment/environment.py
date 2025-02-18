import numpy as np

class environment:
    def __init__(self,dim,num_preclusions,preclusion_size):
        self.dim = dim
        self.num_preclusions = num_preclusions
        self.preclusion_size = preclusion_size
        self.generate_preclusions()

    def generate_preclusions(self):
        # Create
        while True:
            self.x = np.random.uniform(-15, 15, (self.num_preclusions, self.dim))
            self.radii = np.random.uniform(2, 5, self.num_preclusions)
            self.noise = np.random.uniform(0.001, 1, self.num_preclusions)
            
            overlap = False
            for i in range(self.num_preclusions):
                for j in range(i + 1, self.num_preclusions):
                    dist = np.linalg.norm(self.x[i] - self.x[j])
                    if dist < (self.radii[i] + self.radii[j]):  # Check for overlap
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                return #self.x, self.radii
            
    def in_preclusion(self,pos):
        # Determine if the agent is in the preclusion
        in_pre = self.x - pos.T
        R = []
        for inside, radii, noise in zip(in_pre, self.radii, self.noise):
            R.append( [np.linalg.norm(inside)<=radii,noise] )

        return R