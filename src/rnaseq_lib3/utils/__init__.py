import requests


def rget(url, params=None):
    """
    Wrapper for requests.get that checks status code

    :param str url: Request URL
    :param dict params: Parameters for request
    :return: Request from URL or None if status code != 200
    :rtype: requests.models.Response
    """
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    else:
        return r
