from datetime import datetime, time

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
