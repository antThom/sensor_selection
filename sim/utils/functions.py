from datetime import datetime, time

import yaml


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
        if type(attr) is not str:
            # Congrats if you trip this
            print(f"The attribute {attr} is not a string, it is type {type(attr)}. The attribute {attr} had the value of {value}. The attributes being parsed were: {all_config_dict}")
            raise TypeError
        
        # Prevent empty configurations from writing
        if value is None:
            continue

        if not hasattr(agent, attr):
            continue

        setattr(agent, attr, value)
