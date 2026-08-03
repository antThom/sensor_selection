"""A dummy sensor for testing and creating a sensor"""

from sim.sensors.sensor import Sensor, SensorType


class DummySensor(Sensor):
    """_summary_
    A dummy sensor that contains all the things to make a sensor... but no special functions

    Args:
        Sensor (_type_): _description_
    """

    def __init__(self):
        super().__init__()

        self.model = None
        self.sensor_id = None
        self.type = SensorType.DUMMY
        self.name = "dummy"
