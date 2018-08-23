import os
import subprocess
from typing import List


def get_base_call(mount_path: str) -> List[str]:
    """Return base docker call with mounted path"""
    return ['docker', 'run',
            '--log-driver=none',
            '-v', '{}:/data'.format(os.path.abspath(mount_path))]


def fix_permissions(tool: str, work_dir: str):
    """Fix permissions of a mounted docker directory by reusing the tool"""
    base_docker_call = get_base_call(work_dir)
    base_docker_call.append('--entrypoint=chown')
    stat = os.stat(work_dir)
    command = base_docker_call + [tool] + ['-R', '{}:{}'.format(stat.st_uid, stat.st_gid), '/data']
    subprocess.check_call(command)
