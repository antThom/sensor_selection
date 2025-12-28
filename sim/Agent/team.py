import numpy as np
import os 
import sim.print_helpers as ph
import json
from pathlib import Path
from sim.Agent import agent as AGENT

class Team:
    def __init__(self, config: dict, team_name="team"):
        print(f"{ph.GREEN}Define {team_name} team{ph.RESET}")
        
        self.config = config
        self.team = team_name
        
        # Get number of agents on the team
        self.getNumAgents()

        # Assign agents to the team
        self.assignAgents()

        # Assign team color
        self.team_color = self.config.get("color", [0.3,0.3,0.3,1])

    def _reset_states(self,terrain_bound=(None,None),physicsClient=None):
        for agent in self.agents:
            agent._reset_states(terrain_bound=terrain_bound,physicsClient=physicsClient,team=self.team_color)

    def assignAgents(self):
        self.agents = []
        for key, val in self.config.items():
            if "agent" in key:
                self.agents.append( AGENT.Agent(val) )

    def getNumAgents(self):
        self.Num_agents = 0
        # get the number of agents from the dict keys
        for key in self.config.keys():
            if "agent" in key:
                self.Num_agents += 1

    def getNumSensors(self):
        self.Num_sensors = []
        # get the number of agents from the dict keys
        for agent in self.agents:
            self.Num_sensors.append(len(agent.sensors))
        return self.Num_sensors

    def get_states(self, physics_client):
        states = {}
        for idx, agent in enumerate(self.agents):
            pos, ori, vel, ang_rate = agent.get_states(physics_client)
            states[idx] = {"pos": pos, "ori": ori, "vel": vel, "ang_rate": ang_rate}
        return states
    
    def assignSenor(self,action):
        for agent, act in zip(self.agents,action):
            agent.assignSenor(act)