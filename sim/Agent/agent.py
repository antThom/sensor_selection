import numpy as np
import os 
import json
import sim.print_helpers as ph
import pybullet as p
import pybullet_data
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from sim.Sensor.sensor import load_sensor_from_file

class Agent:
    def __init__(self,filepath):
        print(f"{ph.GREEN}Define Agent{ph.RESET}")
        self.position = np.zeros((3,1))
        self.velocity = np.zeros((3,1))
        self.orientation = np.zeros((3,1))
        self.angular_rates = np.zeros((3,1))
        self.mass = 1
        self.inertia = np.eye(3)
        self.has_sensor = False
        self.tf = {}
        if self.has_sensor:
            self.tf['body2Sensor'] = [Rot.from_matrix(np.eye(3)),np.zeros((3,1))]
        self.filepath = filepath
        self.max_vel = 1


        self._update_states()

    def _update_states(self):
        self.x = np.vstack([self.position,self.orientation,self.velocity,self.angular_rates])

    def _reset_states(self,x=None, terrain_bound=(None,None),physicsClient=None,team=None):
        if x:
            print("state is defined but Need to fill out in agent")
        elif not x and not terrain_bound.any():
            print("neither the state nor the terrain bounds are defined, need to fill out")
        else:
            self.position[:2] = np.array(terrain_bound).reshape((2,1)) * np.random.uniform(low=0.0, high=0.60, size=(2,1))
            self.position[-1] = 10
            self.velocity = self.max_vel * np.random.uniform(low=0.0, high=1.0, size=(3,1))
            self.orientation = np.zeros((3,1))
            self.angular_rates = np.zeros((3,1))
            self._update_states()
            self._load_file(physicsClient,team)

    def _load_file(self,physicsClient=None,team=None):
        # Load the agent's json 
        with open(self.filepath, "r") as f:
            config_data = json.load(f)
        self.config = config_data
        
        # Load the urdf file
        agent_file  = self.config.get("agent", "r2d2.urdf")  # was .udrf
        agent_file = Path(os.path.abspath(agent_file))
        # resolve relative to the agent JSON file location
        # base_dir = os.path.dirname(self.filepath)
        # agent_file = agent_file if os.path.isabs(agent_file) else os.path.join(base_dir, agent_file)

        self.id = p.loadURDF(
            fileName=str(agent_file),
            basePosition=self.position.flatten().tolist(),
            baseOrientation=p.getQuaternionFromEuler(self.orientation.flatten().tolist()),
            physicsClientId=physicsClient,
            globalScaling=1
        )

        if team:
            p.changeVisualShape(self.id, linkIndex=-1, rgbaColor=team)
        else:
            p.changeVisualShape(self.id, linkIndex=-1, rgbaColor=[0.3,0.3,0.3,1])


        # Load the sensors
        sensors_cfg = self.config.get("sensors", {})
        self.sensors = []
        sensor_mounts = {}

        for name, entry in sensors_cfg.items():
            if not isinstance(entry, dict) or "file" not in entry:
                raise ValueError(f"Sensor '{name}' must be an object with a 'file' key")

            sensor_path = os.path.abspath(entry["file"])
            # sensor_path = sensor_rel if os.path.isabs(sensor_rel) else os.path.join(base_dir, sensor_rel)
            # sensor_path = os.path.abspath(sensor_path)

            sensor = load_sensor_from_file(sensor_path)  # returns a subclass instance, e.g. Camera(...)
            self.sensors.append(sensor)

            # optional mount info (e.g., for later attaching transforms)
            sensor_mounts[name] = {
                "pos": entry.get("sensor_pos", [0, 0, 0]),
                "rpy": entry.get("sensor_rpy", [0, 0, 0])
            }