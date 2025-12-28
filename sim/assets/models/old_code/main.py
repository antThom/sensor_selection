import os
import numpy as np
import sys

sys.path.append(os.path.abspath("agent"))
sys.path.append(os.path.abspath("sensor"))
sys.path.append(os.path.abspath("target"))
sys.path.append(os.path.abspath("environment"))
sys.path.append(os.path.abspath("plot_func"))
sys.path.append(os.path.abspath("estimator"))
import agent as A
import sensor as S
import target as T
import estimator as kf
import environment as E

import plot_func
import matplotlib.pyplot as plt






def main(agents, targets, env, time, dt, plotting):
    if plotting:
        plt.ion()
        fig, ax = plt.subplots()
    for t in np.arange(time[0],time[1],dt):
        # STEP 1: MOVE THE AGENT
        for agent in agents:
            agent.read_sensor(targets, env)
            # agent.eom(targets, env)
            

        # STEP 4: PLOT
        if plotting:
            plt.cla()
            plt.title(label=f"Time: {np.round(t,2):2f} sec")
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
        # targets[tar].trajectory_generation(dim=dim, n_points=400, n_control_points=40)

    # AGENT PARAMS
    num_agent   = 1
    sensors     = {"range_bearing_sensor": (1,np.pi/2,4,S.random_diagonal_matrix(['range','bearing'])), "bearing_sensor": (1,2*np.pi,3,S.random_diagonal_matrix(['bearing'])), "range_bearing_range_rate_sensor": (1,np.pi/4,5,S.random_diagonal_matrix(['range','range_rate','bearing']))}
    agents = []
    for tar in np.arange(num_agent):
        # Agent State: [x,y,theta,V]
        agent_x = np.reshape(np.hstack([np.random.uniform(-10,10,2),np.random.uniform(0,2*np.pi,1),np.random.uniform(0.1,4,1)]),(4,1))
        agents.append( A.agent(agent_x,dim,dt,sensors) )

    # ENVIRONMENT
    num_preclusions = 2
    preclusion_size = 0.1
    env = E.environment(dim,num_preclusions,preclusion_size)
    # if plotting:
    #     plt.ion()
    #     fig, ax = plt.subplots()
    #     plot_func.plot_scene(agents, targets, env, ax, fig)
    main(agents, targets, env, time, dt, plotting)