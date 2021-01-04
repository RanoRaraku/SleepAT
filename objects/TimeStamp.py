"""
Made by Michal Borsky, 2020, copyright (C) RU
"""
import datetime
from datetime import datetime, timedelta
import sleepat
from sleepat import opts

class TimeStamp(object):
    """
    A timestamp class that handles conversions between datetime.datetime and string types.
    Used in JSON dicts that handle events, periods, segments, etc. Default timestamp format
    is '%Y/%m/%dT%H:%M:%S.%f' and value is '0000/01/01T00:00:00.00000'.
    """
    def __init__(self, config:str=None, **kwargs):
        """
        Arguments:
            <<stamp>> ... a timestamp string (def:str='0000/01/01T00:00:00.00000')
            <<offset>> ... an increment in seconds to offset the <<stamp>> (def:float=0.0)
            <<format>> ... timestamp format (def:str='%Y/%m/%dT%H:%M:%S.%f')
            config ... a config JSON file to set optional args <>/<<>>
            **kwargs ... setting optional args <>/<<>> through kwargs
        """
        self.conf = opts.TimeStamp(config,**kwargs)
        self.stamp = datetime.strptime(self.conf.stamp,self.conf.format)
        self.increment(self.conf.offset)

    def date_to_string(self):
        """
        Return datetime value as a string.
        """
        return self.stamp.strftime(self.conf.format)

    def string_to_date(self):
        """
        Return datetime value as a datetime.datetime.
        """
        return datetime.strptime(self.stamp,self.conf.format)

    def increment(self, seconds:float=0.0):
        """
        Increment TimeStamp by defined number of 'seconds'.
        """
        self.stamp = self.stamp + timedelta(seconds=seconds)

    def print(self):
        """
        Print datetime value as a string. Identical to TimeStamp.date_to_string().
        """
        return self.date_to_string()

