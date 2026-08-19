from datetime import datetime, time

import yaml
import numpy as np


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
    except ValueError as error:
        raise ValueError(
            f"Invalid time format. Expected '{fmt}'. Error: {error}"
        ) from error


def extract_yaml_configurations(file_path: str):
    """Load and return one UTF-8 YAML mapping."""
    try:
        with open(file_path, encoding="utf-8") as file:
            configs = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"configuration file not found: {file_path}") from None
    except yaml.parser.ParserError:
        raise yaml.parser.ParserError(
            f"A syntax error occured in file {file_path}. Fix the file an try again"
        )

    if not isinstance(configs, dict):
        raise TypeError(f"configuration root must be a mapping: {file_path}")
    return configs


def filter_arguments(valid_keys, dictionary, recursion=1) -> dict:
    """Return recognized keys found up to ``recursion`` mappings deep."""
    allowed = set(valid_keys)
    result = {}

    def visit(current, remaining):
        for key, value in current.items():
            if key in allowed:
                result[key] = value
            if remaining > 0 and isinstance(value, dict):
                visit(value, remaining - 1)

    visit(dictionary, max(int(recursion), 0))
    return result


def filter_arguements(valid_keys, dict_to_filter, recuriveness=1) -> dict:
    """Deprecated compatibility wrapper for the historical misspelling."""
    return filter_arguments(valid_keys, dict_to_filter, recuriveness)


def set_attr_from_configuration(agent: object, config: dict, *args, **kwargs) -> None:
    """Apply recognized, non-null leaf configuration values to an object.

    Nested YAML sections are flattened because simulation classes expose
    detector, model, and mount settings as ordinary attributes. Unknown keys
    are intentionally ignored so one configuration can include metadata used
    by multiple loaders.
    """

    all_config_dict = {}

    def search_dictionary(dictionary: dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                search_dictionary(value)  # Unpack nested dictionaries
            else:
                all_config_dict[key] = value

    if isinstance(config, (tuple, list)):
        for i in config:
            search_dictionary(i)
    else:
        search_dictionary(config)
    search_dictionary(kwargs)  # kwargs is a dictionary

    # Pick up any stray dictionaries that are passed
    for arg in args:
        if isinstance(arg, dict):
            search_dictionary(arg)

    for attr, value in all_config_dict.items():

        # Prevent empty configurations from writing
        if value is None:
            continue

        try:
            if not hasattr(agent, attr):
                continue
        except TypeError as error:
            # Congrats if you trip this
            print(
                f"The attribute {attr} is not a string, it is type {type(attr)}. The attribute {attr} had the value of {value}. The attributes being parsed were: {all_config_dict}",
                error,
            )

        setattr(agent, str(attr), value)


def accept_ndarrays(func):
    """Decorator to convert possible numpy arrays into lists."""

    def wrapper(*args, **kwargs):
        new_args = list()
        new_kwargs = dict()

        # Convert np.array args into lists, if any
        for arg in args:
            if isinstance(arg, np.ndarray):
                new_args.append(arg.tolist())
            else:
                new_args.append(arg)

        # Convert values attached to their keys
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                new_kwargs[key] = value.tolist()
            else:
                new_kwargs[key] = value

        func(*new_args, **new_kwargs) if new_args and new_args else None
        func(*new_args) if new_kwargs == {} else None
        func(**new_kwargs) if new_args == [] else None

    return wrapper


def unbox_1d_ndarray_list(list_to_format: list) -> list:
    """
    If a nested list inside a list, take out the value. For 1D arrays.
    Used to unbox the values inside a list after being converted out of an ndarray
    """

    for index, value in enumerate(list_to_format):
        if isinstance(value, list):
            list_to_format[index] = value[0]

    return list_to_format
