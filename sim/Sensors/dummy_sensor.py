""" A dummy sensor for testing and creating a sensor """

from sim.sensors.sensor import Sensor, SensorType
from sim.utils.functions import set_attr_from_configuration

class DummySensor():
    """_summary_
    A dummy sensor that contains all the things to make a sensor... but no special functions
    
    Args:
        Sensor (_type_): _description_
    """
    
    def __init__(self):
       # super.__init__()
        
        self.model = None
        self.sensor_id = None
        self.type = SensorType.DUMMY
        self.name = None
    
    def set_attributes(self, config:dict) -> None:
        """_summary_
        Overload of initalization created with loading it's own attributes
        Args:
            config (dict): _description_
        """
        
        set_attr_from_configuration(self, config)
        