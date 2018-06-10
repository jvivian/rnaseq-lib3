import requests


def rget(url, params=None) -> requests.models.Response:
    """requests.get wrapper that checks status code for 200 or returns None"""
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    else:
        return r
