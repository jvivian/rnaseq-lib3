import os
from subprocess import check_call

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


def curl(url: str, work_dir: str = None):
    """Wrapper for cURL"""
    work_dir = os.getcwd() if work_dir is None else work_dir
    file_path = os.path.join(work_dir, os.path.basename(url))

    # Download if file doesn't exist
    if os.path.exists(file_path):
        print(f'File already downloaded: {file_path}')
    else:
        check_call(['curl', '-fs', '--retry', '5', url, '-o', file_path])
    return file_path
