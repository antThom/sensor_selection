import os
import numpy as np
import sys

sys.path.append(os.path.abspath("agent"))
sys.path.append(os.path.abspath("sensor"))
sys.path.append(os.path.abspath("target"))
sys.path.append(os.path.abspath("environment"))
sys.path.append(os.path.abspath("plot_func"))
import agent as A
import sensor as S
import target as T
import environment as E
import plot_func
import matplotlib.pyplot as plt






def main(agents, targets, env, ax, fig, time, dt, plotting):
    for t in np.arange(time[0],time[1],dt):
        # STEP 1: MOVE THE AGENT
        for agent in agents:
            agent.eom(targets, env)

        # STEP 4: PLOT
        if plotting:
            plt.cla()
            plt.title(label=f"Time: {t:2f} sec")
            plot_func.plot_scene(agents, targets, env, ax, fig)
            






if __name__ == "__main__":
    np.random.seed(1)
    dim = 2
    dt  = 0.05 
    time = [0,30]
    plotting = True
    # TARGET PARAMS
    num_target  = 1
    targets = []
    for tar in np.arange(num_target):
        target_x = np.reshape(np.random.uniform(-10,10,4),(4,1))
        targets.append( T.target(target_x,dim,dt) )

    # AGENT PARAMS
    num_agent   = 1
    sensors     = {"bearing_sensor": 1}
    agents = []
    for tar in np.arange(num_agent):
        agent_x = np.reshape(np.hstack([np.random.uniform(-10,10,2),np.random.uniform(0,2*np.pi,1),np.random.uniform(0.1,4,1)]),(4,1))
        agents.append( A.agent(agent_x,dim,dt,sensors) )

    # ENVIRONMENT
    num_preclusions = 8
    preclusion_size = 0.1
    env = E.environment(dim,num_preclusions,preclusion_size)
    if plotting:
        plt.ion()
        fig, ax = plt.subplots()
        plot_func.plot_scene(agents, targets, env, ax, fig)
    main(agents, targets, env, ax, fig, time, dt, plotting)