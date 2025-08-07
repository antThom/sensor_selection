import numpy as np
import matplotlib.pyplot as plt

def compute_velocity(xt,xtm1,dt=0.1):
    # Compute the derivative between two points
    return (xt-xtm1)/dt

def compute_lipschitz_constant(xt,xtm1):
    # Compute lipschitz constant
    return np.linalg.norm(xt[:2]-xtm1[:2])/np.linalg.norm(xt[2:]-xtm1[2:])

def generate_lipschitz(dim=2, dt=0.1, x0=None, L=1):
    num_waypoints = 10
    waypoints = np.zeros((dim*2,num_waypoints))
    # waypoints[:,0] = np.reshape(x0,(4,))
    # waypoints[:,-1] = np.reshape(x0,(4,))
    if dim == 2:
        for col, pt in enumerate(waypoints.T):
            # Generate a new waypoint position
            if col == 0 or col == num_waypoints-1:
                # Start and End in same position
                pt = np.reshape(x0,(4,))
            else:
                Lipschitz_constant = np.inf
                # pt[:2] = np.random.randn(2)
                # Compute Velocity
                # pt[2:] = compute_velocity(xt=pt,xtm1=waypoints[:,col-1])
                # Check Lispchitz Constant
                # Lipschitz_constant = compute_lipschitz_constant(xt=pt,xtm1=waypoints[:,col-1])
                while Lipschitz_constant>L:
                    pt[:2] = np.random.randn(2)
                    # Compute Velocity
                    pt[2:] = compute_velocity(xt=pt[:2],xtm1=waypoints[:2,col-1])
                    # Check Lispchitz Constant
                    Lipschitz_constant = compute_lipschitz_constant(xt=pt,xtm1=waypoints[:,col-1])
            waypoints[:,col] = pt
    elif dim == 3:
        print("3D")

    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(waypoints.T[:,0], waypoints.T[:,1],linewidth=2)
    ax.quiver(waypoints.T[:,0], waypoints.T[:,1], waypoints.T[:,2], waypoints.T[:,3],
                 color='red', angles='uv',linewidths=5, width=0.05)
    for i in np.arange(num_waypoints):
        plt.annotate(f'({i})', (waypoints.T[i,0], waypoints.T[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_aspect('equal') 
    plt.tight_layout()
    plt.show(block=True)





if __name__ == "__main__":
    np.random.seed(1)
    dim = 2
    dt  = 0.05 
    target_x = np.reshape(np.random.uniform(-10,10,4),(4,1))
    lipschitz_constant = 0.08
    generate_lipschitz(dim=dim, dt=dt, x0=target_x, L=lipschitz_constant)