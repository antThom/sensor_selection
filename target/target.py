import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import CubicSpline

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
            

    
    def trajectory_generation(self, dim=2, n_points=200, n_control_points=10):
        """
        Generates smooth random numbers using cubic spline interpolation.

        Parameters:
        - n_points (int): Total number of points for the smooth curve.
        - n_control_points (int): Number of random control points to interpolate.
        - random_seed (int): Seed for reproducibility.

        Returns:
        - x (np.ndarray): X values.
        - y (np.ndarray): Smooth random function values.
        """
        if dim == 2:
            # Randomly sampled control points
            half_point = int(np.floor(n_control_points/2))
            x_control = np.hstack((np.linspace(self.x[0,0], 30, half_point), np.linspace(30, self.x[0,0], half_point)))
            y_control = np.hstack((self.x[1,0],np.random.randn(n_control_points-2),self.x[1,0]))
            # y_control = np.random.randn(n_control_points)

            # Cubic spline interpolation
            spline = CubicSpline(x_control, y_control)

            # Generate the smooth curve
            x = np.linspace(0, 10, n_points)
            y = spline(x)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Smooth Random Numbers', color='b')
        plt.scatter(x_control, y_control, color='r', label='Control Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Smooth Random Numbers with Continuous Derivative (Cubic Spline)')
        plt.grid(True)
        plt.legend()
        plt.show()


        
        # GP Kernel: smoothness controlled by length_scale
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale)
        
        # Initial Condition
        # if dim==2:
        #     print ("hi")



def generate_smooth_random_numbers(n_points=100, length_scale=1.0, random_seed=None):
    """
    Generates random numbers with a continuously differentiable derivative
    using a Gaussian Process with an RBF kernel.

    Parameters:
    - n_points (int): Number of points.
    - length_scale (float): Smoothness of the function.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - x (np.ndarray): X values.
    - y (np.ndarray): Smooth random function values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # X values (time or index)
    x = np.linspace(0, 10, n_points).reshape(-1, 1)

    # GP Kernel: smoothness controlled by length_scale
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale)
    
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # Generate random target values for fitting
    random_targets = np.random.randn(n_points).reshape(-1, 1)

    # Fit GP
    gp.fit(x, random_targets)
    
    # Predict smooth random function
    y, _ = gp.predict(x, return_std=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Smooth Random Numbers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smooth Random Numbers with Continuous Derivative')
    plt.grid(True)
    plt.legend()
    plt.show()

    return x.flatten(), y.flatten()

def generate_smooth_random_numbers_spline(n_points=200, n_control_points=10, random_seed=None):
    """
    Generates smooth random numbers using cubic spline interpolation.

    Parameters:
    - n_points (int): Total number of points for the smooth curve.
    - n_control_points (int): Number of random control points to interpolate.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - x (np.ndarray): X values.
    - y (np.ndarray): Smooth random function values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly sampled control points
    x_control = np.linspace(0, 10, n_control_points)
    y_control = np.random.randn(n_control_points)

    # Cubic spline interpolation
    spline = CubicSpline(x_control, y_control)

    # Generate the smooth curve
    x = np.linspace(0, 10, n_points)
    y = spline(x)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Smooth Random Numbers', color='b')
    plt.scatter(x_control, y_control, color='r', label='Control Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smooth Random Numbers with Continuous Derivative (Cubic Spline)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return x, y

if __name__ == "__main__":
    # Example usage
    # generate_smooth_random_numbers(n_points=200, length_scale=1.5, random_seed=42)

    # Example usage
    generate_smooth_random_numbers_spline(n_points=300, n_control_points=12, random_seed=42)