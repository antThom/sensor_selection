from datetime import datetime, time
import yaml
import os
import json


def string_to_time(time_str: str, fmt: str = "%H:%M:%S") -> time:
    """
    Convert a string to a time object.

    :param time_str: The time string to convert (e.g., "14:30:15").
    :param fmt: The format of the time string (default is 24-hour HH:MM:SS).
    :return: A datetime.time object.
    :raises ValueError: If the string does not match the format.
    """
    if not isinstance(time_str, str):
        raise TypeError("time_str must be a string.")

    try:
        # Parse the string into a datetime object
        parsed_time = datetime.strptime(time_str.strip(), fmt).time()
        return parsed_time
    except ValueError as e:
        raise ValueError(f"Invalid time format. Expected '{fmt}'. Error: {e}")


def extract_yaml_configurations(file_path: str):
    """_summary_
    Loads a yaml file given a path

    Args:
        file_path (str): String of file path

    Returns:
        dict: generated dictionary created from yaml
    """
    try:
        with open(file_path, "r") as file:
            configs = yaml.safe_load(file)
    except FileNotFoundError as error:
        print(f"Check configuration again! Path {file_path} was not found. ")
        raise error
    
    return configs


def filter_arguements(valid_keys: dict, dict_to_filter, recuriveness=1) -> dict:
    """_summary_
    Filters through a dictionary and searches for valid keys.

    Args:
        valid_keys (dict): _description_
        dict_to_filter (_type_): _description_
        recurseiveness (int): How many levels the filter searches through

    Returns:
        dict: _description_
    """

def set_attr_from_configuration(agent: object, config: dict,*args, **kwargs) -> None:
    """_summary_
    Given an agent object and configuration, will edit the internal configurations of the
    agent according to a configuration file. If the attribute is not in the object or has a value of None, it will not be assigned.

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """

    all_config_dict = dict()
    
    def search_dictionary(dictionary:dict): 
        for key, value, in dictionary.items():
            if isinstance(value, dict):
                search_dictionary(value) # Unpack nested dictionaries
            else:
                all_config_dict[key] = value

    search_dictionary(config)
    search_dictionary(kwargs) # kwargs is a dictionary
    
    # Pick up any stray dictionaries that are passed
    for arg in args:
        if isinstance(arg, dict):
            search_dictionary(arg)
            
    
    for attr, value in all_config_dict.items():
        # Prevent empty configurations from writing
        if value is None:
            continue
        
        if getattr(agent, attr, None) is None:
        #    print(f"Attribute "{attr}"" is not a valid attribute of object {type(agent)} !")
        # Let other attributes pass through (like sensor types on agent objects)
            continue 
        
        setattr(agent, attr, value)