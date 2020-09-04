"""
Made by Michal Borsky, 2019, copyright (C) RU
Returns a datatime obj as string in default format (%Y/%m/%dT%H:%M:%S.%f).

Arguments:
    data ... a datetime object
"""
def date_to_string(date):
    return date.strftime('%Y/%m/%dT%H:%M:%S.%f')
