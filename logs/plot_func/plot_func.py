import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

def plot_scene(agents, targets, env, ax, fig):
    fontSize = 16
    # fig, ax = plt.subplots()
    ax.set_aspect('equal') 
    # plt.grid()   
    
    # Plot agents as blue arrows
    for agent in agents:
        heading = agent.x[3,0]
        V = agent.x[2,0]
        vx = V*np.cos(heading)
        vy = V*np.sin(heading)
        speed = np.linalg.norm((vx,vy))
        ax.quiver(agent.x[0,0], agent.x[1,0], vx/speed, vy/speed,
                 color='blue', angles='uv')
        ax.scatter(agent.x[0,0], agent.x[1,0],c='blue', s=18)
        
        # Plot the Sensor
        agent.sensors[agent.select].plot_sector(ax,(agent.x[0,0], agent.x[1,0]),agent.x[3,0])
    
    # Plot targets as red circles
    for target in targets:
        speed = np.linalg.norm(target.x[2:-1,0])
        ax.quiver(target.x[0,0], target.x[1,0], target.x[2,0]/speed, target.x[3,0]/speed,
                 color='red', angles='uv')
        ax.scatter(target.x[0,0], target.x[1,0],c='red', s=18)
    
    # Plot circular obstacles in black
    for ii in np.arange(env.num_preclusions):
        center = env.x[ii,:]
        radius = env.radii[ii]
        circle = plt.Circle(center, radius, color='black', fill=True, linewidth=2, alpha=remap_noise(env.noise[ii]))
        ax.add_patch(circle)

    # Plot Estimated Target pos and vel (For 1 target)
    speed_hat = np.linalg.norm(agents[0].xt_hat[2:,0])
    est_color = np.array([255, 204, 204])/255

    ax.quiver(agents[0].xt_hat[0,0], agents[0].xt_hat[1,0], agents[0].xt_hat[2,0]/speed_hat, agents[0].xt_hat[3,0]/speed_hat,
                 color=est_color, angles='uv')
    ax.scatter(agents[0].xt_hat[0,0], agents[0].xt_hat[1,0],c=est_color, s=24, marker='+')
    
    set_max_bounds(agents,targets,env, ax)
    set_axis(ax,fig,fontSize)
    # plt.show(block=False)
    plt.pause(0.05)

def set_max_bounds(agents,targets,env,ax):
    x_min_limit, x_max_limit = plt.gca().get_xlim()
    y_min_limit, y_max_limit = plt.gca().get_ylim()

    x = []
    y = []
    attr = [x for xs in [agents,targets] for x in xs]
    attr.append(env)
    # Collect all of the x and y positions
    for agent in attr:
        if hasattr(agent,"x"):
            x.append(agent.x[0,0])
            if agent.x.shape[1] == 1:
                y.append(agent.x[1,0])
            elif agent.x.shape[1] == 2:
                y.append(agent.x[0,1])
    
    # Find the max and min of x and y
    scale = 1.25
    max_x = np.ceil(np.max(x) * scale)
    min_x = np.ceil(np.min(x) * scale)
    max_y = np.ceil(np.max(y) * scale)
    min_y = np.ceil(np.min(y) * scale)



    # Assign Limits
    current_limit = (x_min_limit-x_max_limit)**2 + (y_min_limit-y_max_limit)**2
    new_limit     = (min_x-max_x)**2 + (min_y-max_y)**2
    if new_limit>current_limit:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    # return max_x, min_x, max_y, min_y


def set_axis(ax,fig,fontSize):
    ax.set_xlabel("X (m)", fontsize=fontSize)
    ax.set_ylabel("Y (m)", fontsize=fontSize)
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.locator_params(axis='x',nbins=6)
    plt.locator_params(axis='y',nbins=6)
    plt.tight_layout()


def remap_noise(noise):
    """
    Noise is a value [0,1], where 0 is no added noise and 1 is an 
    overwhelming amount of noise. For plotting purposes remap these
    to [0,0.7]
    """
    return noise * 0.7
