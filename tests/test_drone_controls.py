from panda3d.core import NodePath, PandaNode

from sim.Agent.agent import Agent
from sim.Agent.drone_controls import DroneControls


class FakeTaskManager:
    def add(self, callback, name):
        self.callback = callback
        self.name = name


class FakeWorld:
    def __init__(self, agent):
        self.agent_list = [agent]
        self.render = NodePath(PandaNode("render"))
        self.taskMgr = FakeTaskManager()
        self.bindings = {}

    def accept(self, event, callback, arguments):
        self.bindings[event] = (callback, arguments)


def make_controls():
    agent = Agent()
    agent.object_node_path = NodePath(PandaNode("drone"))
    world = FakeWorld(agent)
    agent.object_node_path.reparentTo(world.render)
    return agent, world, DroneControls(world, move_speed=10.0, rotation_speed=90.0)


def test_forward_and_vertical_controls_move_agent_and_sync_position():
    agent, _, controls = make_controls()
    controls.set_key("forward", True)
    controls.set_key("up", True)

    controls.apply_motion(1.0)

    assert agent.object_node_path.getY() > 0.0
    assert agent.object_node_path.getZ() > 0.0
    assert agent.position[1, 0] == agent.object_node_path.getY()
    assert agent.position[2, 0] == agent.object_node_path.getZ()


def test_arrow_controls_rotate_agent():
    agent, _, controls = make_controls()
    controls.set_key("turn_right", True)
    controls.set_key("pitch_up", True)

    controls.apply_motion(0.5)

    assert agent.object_node_path.getH() == 45.0
    assert agent.object_node_path.getP() == 45.0


def test_release_stops_motion():
    agent, _, controls = make_controls()
    controls.set_key("forward", True)
    controls.apply_motion(0.5)
    first_position = agent.object_node_path.getPos()

    controls.set_key("forward", False)
    controls.apply_motion(0.5)

    assert agent.object_node_path.getPos() == first_position
    assert not agent.velocity.any()
