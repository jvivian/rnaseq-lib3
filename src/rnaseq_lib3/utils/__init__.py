import requests


def rget(url: str, params: dict = None) -> requests.models.Response:
    """requests.get wrapper that checks status code for 200 or returns None"""
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    else:
        return r

def rreplace(s, old, new, occurrence):
    """https://stackoverflow.com/users/230454/mg"""
    li = s.rsplit(old, occurrence)
    return new.join(li)
