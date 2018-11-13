import os
import tarfile
from subprocess import check_call
from typing import List, Optional

import requests


# Web / Downloading
def rget(url: str, params: dict = None) -> Optional[requests.models.Response]:
    """requests.get wrapper that checks status code for 200 or returns None"""
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    else:
        return r


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


# Files
def tarball_files(tar_name: str, file_paths: List[str], output_dir: str = '.', prefix: str = None):
    """
    Creates a tarball from a group of files

    Args:
        tar_name: Name of tarball
        file_paths: Absolute file paths to include in the tarball
        output_dir: Output destination
        prefix: Optional prefix for files inside the tarball
    """
    prefix = '' if prefix is None else prefix
    with tarfile.open(os.path.join(output_dir, tar_name), 'w:gz') as f_out:
        for file_path in file_paths:
            if not file_path.startswith('/'):
                raise ValueError('Path provided is relative not absolute.')
            arcname = prefix + os.path.basename(file_path)
            f_out.add(file_path, arcname=arcname)


# Naming operations
def rreplace(s, old, new, occurrence):
    """https://stackoverflow.com/users/230454/mg"""
    li = s.rsplit(old, occurrence)
    return new.join(li)
