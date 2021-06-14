"""
Made by Michal Borsky, 2020, copyright (C) RU
"""
import copy
from copy import deepcopy
import datetime
from datetime import datetime, timedelta
import sleepat
from sleepat import opts

class TimeStamp(object):
    """
    A timestamp class that handles conversions between datetime.datetime and string types.
    Used in JSON dicts that handles events, periods, segments, etc. Default timestamp format
    is '%Y/%m/%dT%H:%M:%S.%f' and value is '0001/01/01T00:00:00.00000'. The actual value is
    saved as datetime.datetime object. 
    """

    def __init__(self, tstamp:str='0001/01/01T00:00:00.000000', config:str=None, **kwargs):
        """
        Arguments:
            tstamp ... a timestamp string (def:str='0001/01/01T00:00:00.000000')
            <<format>> ... timestamp format (def:str='%Y/%m/%dT%H:%M:%S.%f')
            config ... a config JSON file to set optional args <>/<<>>
            **kwargs ... setting optional args <>/<<>> through kwargs
        """
        self.conf = opts.TimeStamp(config,**kwargs)
        self.tstamp = datetime.strptime(tstamp, self.conf.format)       

    def as_string(self) -> str:
        """
        Return datetime value as a string.
        """

        return self.tstamp.strftime(self.conf.format)

    def as_date(self) -> datetime:
        """
        Return datetime value as a datetime.datetime.
        """

        return self.tstamp

    def increment(self, seconds:float=0.0) -> None:
        """
        Increment TimeStamp by defined number of 'seconds'. Can be a negative 
        number, the function then decrements.

        Arguments:
            seconds ... time shift in seconds (def:float = 0.0)
        """

        self.tstamp = self.tstamp + timedelta(seconds=seconds)   

    def to_format(self, format:str='%Y/%m/%dT%H:%M:%S.%f') -> None:
        """
        Change the timestamp to the specified format by updating the self.conf.format.
        Format is a Python keyword but we are OK redefining it, its only inside this
        method anyway.

        Arguments:
            format ... timestamp format to cast into (def:str = '%Y/%m/%dT%H:%M:%S.%f')
        """
        if not isinstance(format, str):
            print(f'Error Nox.TimeStamp: "format" expects string, got {type(format)}.')
            exit(1)

        self.conf.format = format

    def copy(self):
        """
        Copy the whole object. The return is a new instance of the TimeStamp class.
        """

        return deepcopy(self)

    def delta(self, timestamp) -> float:
        """
        Calculate the difference (delta) with respect to the provided TimeStamp.
        Used to hide the timedelta calculation. The output is in seconds.
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        delta = (self.tstamp - timestamp.tstamp).total_seconds() 
        return round(delta, 8)

    def __eq__(self, timestamp) -> bool:
        """
        Define equality betweent two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp == timestamp.tstamp

    def __ne__(self, timestamp) -> bool:
        """
        Define not equal between two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp != timestamp.tstamp

    def __gt__(self, timestamp) -> bool:
        """
        Define greater then between two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp > timestamp.tstamp

    def __ge__(self, timestamp) -> bool:
        """
        Define greater or equal between two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp >= timestamp.tstamp

    def __lt__(self, timestamp) -> bool:
        """
        Define lower then between two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.        
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp < timestamp.tstamp

    def __le__(self, timestamp) -> bool:
        """
        Define lesser or equal between two TimeStamp objects. The comparison is made
        between their datetimes, format plays no role.        
        """
        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        return self.tstamp <= timestamp.tstamp

    def __repr__(self) -> str:
        """
        Print the timestamp as a string when calling the object. Identical
        to TimeStamp.date_to_string(). For the sake of convenience.
        """

        return self.as_string()


    """
    def __sub__(self, timestamp):
        Subtraction of two TimeStamps is again a TimeStamp, the resulting datetime is equal
        to their difference, and the format is equal to the format of the parent TimeStamp.

        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        out = self.copy()
        out.increment(seconds = timestamp.tstamp.total_seconds())
        return out

  
    def __add__(self, timestamp):
        Addition of two TimeStamps is again a TimeStamp, the resulting datetime is equal
        to their combined tstamps, and the format is equal to the format of the parent 
        TimeStamp.

        if not isinstance(timestamp, TimeStamp):
            print(f'Error Nox.TimeStamp: timestamp expects object.TimeStamp object, got {type(timestamp)}.')
            exit(1)

        out = self.copy()
        out.increment(seconds = timestamp.tstamp.total_seconds())
        return out
    """