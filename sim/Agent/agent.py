import numpy as np
import os 
import json
import sim.print_helpers as ph
import pybullet as p
import pybullet_data
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.Sensor.sensor import load_sensor_from_file
from sim.Constants import *

class Agent:
    def __init__(self,filepath, thermal: ThermalManager):
        print(f"{ph.GREEN}Define Agent{ph.RESET}")
        self.position = np.zeros((3,1))
        self.velocity = np.zeros((3,1))
        self.orientation = np.zeros((3,1))
        self.angular_rates = np.zeros((3,1))
        self.mass = 1
        self.agent_id = []
        self.thermal = thermal
        self.inertia = np.eye(3)
        self.tf = {}
        self.filepath = filepath
        self.max_vel = 1
        with open(self.filepath, "r") as f:
            config_data = json.load(f)
        self.config = config_data
        sensors_cfg = self.config.get("sensors", {})
        if sensors_cfg:
            self.has_sensor = True
            self.config_sensors(sensors_cfg)
            self.init_sensor_set()
            self.assignSenor(2)
            # self.active_sensor_set = 1
        else:
            self.has_sensor = False
        self._update_states()
        sound_cfg = self.config.get("sound", None)
        if sound_cfg is not None:
            self._init_sound(sound=sound_cfg)        

    def _init_sound(self,sound):
        from sim.Sound.point_source import SoundPointSource
        self.sound = SoundPointSource(sound_file=sound,dt=sim_dt,loop=True,position=self.position,velocity=self.velocity)
        

    def start_sound(self):
        self.sound.start()

    def _update_states(self):
        self.x = np.vstack([self.position,self.orientation,self.velocity,self.angular_rates])

    def _reset_states(self,x=None, terrain_bound=(None,None),physicsClient=None,team=None):
        if x:
            print("state is defined but Need to fill out in agent")
        elif not x and not terrain_bound.any():
            print("neither the state nor the terrain bounds are defined, need to fill out")
        else:
            self.position[:2] = np.array(terrain_bound/6).reshape((2,1)) * np.random.uniform(low=0.0, high=0.20, size=(2,1))
            self.position[-1] = 25
            self.velocity = self.max_vel * np.random.uniform(low=0.0, high=1.0, size=(3,1))
            self.orientation = np.zeros((3,1))
            self.angular_rates = np.zeros((3,1))
            self._update_states()
            self._load_file(physicsClient,team)
            if hasattr(self,'sound'):
                self.sound.pos = self.position.flatten()
                self.sound.vel = self.velocity.flatten()
                self.sound.set_active(True)

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

        self.agent_id = p.loadURDF(
            fileName=str(agent_file),
            basePosition=self.position.flatten().tolist(),
            baseOrientation=p.getQuaternionFromEuler(self.orientation.flatten().tolist()),
            physicsClientId=physicsClient,
            globalScaling=1
        )

        if team:
            p.changeVisualShape(self.agent_id, linkIndex=-1, rgbaColor=team)
        else:
            p.changeVisualShape(self.agent_id, linkIndex=-1, rgbaColor=[0.3,0.3,0.3,1])

        self.thermal.register_body(self.agent_id, str(agent_file), per_link=False)


        # Load the sensors
        # sensors_cfg = self.config.get("sensors", {})
        # if sensors_cfg:
        #     self.config_sensors(sensors_cfg)
        # else:
        #     self.sensors = None
        
    def config_sensors(self,sensors_cfg):
        self.sensors = []
        sensor_mounts = {}

        for name, entry in sensors_cfg.items():
            if not isinstance(entry, dict) or "file" not in entry:
                raise ValueError(f"Sensor '{name}' must be an object with a 'file' key")

            sensor_path = os.path.abspath(entry["file"])
            # sensor_path = sensor_rel if os.path.isabs(sensor_rel) else os.path.join(base_dir, sensor_rel)
            # sensor_path = os.path.abspath(sensor_path)

            sensor = load_sensor_from_file(sensor_path, name, self.thermal)  # returns a subclass instance, e.g. Camera(...)
            sensor.attach_to_agent(self)
            sensor.tf[name] = {"pos": entry.get("sensor_pos", [0, 0, 0]), "rpy": entry.get("sensor_rpy", [0, 0, 0])}
            self.sensors.append(sensor)
            self.tf["body2Sensor"] = [ Rot.from_euler('xyz',self.sensors[-1].tf[name]['rpy'],degrees=True), self.sensors[-1].tf[name]['pos'] ]
            # optional mount info (e.g., for later attaching transforms)
            
        self.sensors[0].start_capture(rate_hz=self.sensors[0]._rate_hz) 

    def init_sensor_set(self):
        self.sensor_setcombos = {}
        p = len(self.sensors)
        for i in range(1, 2**p):  # skip 0 (empty set)
            active = [self.sensors[j] for j in range(p) if (i >> j) & 1]
            self.sensor_setcombos[i] = active
        
    def assignSenor(self,action):
        self.active_sensor_set = action

    def get_states(self,physics_client):
        pos, ori = p.getBasePositionAndOrientation(self.agent_id, physicsClientId=physics_client)
        vel, ang_vel = p.getBaseVelocity(self.agent_id)
        self.position = np.array(pos).reshape((3,1))
        self.orientation = np.array(ori).reshape((4,1))
        self.velocity = np.array(vel).reshape((3,1))
        self.angular_rates = np.array(ang_vel).reshape((3,1))
        self._update_states()
        return self.position, self.orientation, self.velocity, self.angular_rates
    
    def get_id(self):
        return self.agent_id