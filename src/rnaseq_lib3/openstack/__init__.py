from subprocess import check_call
from subprocess import PIPE
from subprocess import Popen
import os


USER = 'ubuntu'
IPS = [
    '10.50.100.34',
    '10.50.100.32',
    '10.50.100.79',
    '10.50.100.80',
]

LOCAL_IPS = [
    '10.104.0.9',
    '10.104.0.8',
    '10.104.0.5',
    '10.104.0.15',
]


def ccall(cmd: str) -> None:
    """
    Cluster CALL. Runs command across all clusters

    Args:
        cmd: Command to execute across all IPs
    """
    commands = [f'ssh {USER}@{ip} {cmd}' for ip in IPS]
    parallel_popen(commands)


def scall(cmd: str, screen_name: str = None):
    """
    Screen CALL. Same as ccall but runs in a background screen.
    The screen exits upon the process terminating

    Args:
        cmd: Command to run in a screen across all IPs
        screen_name: Optional screen name
    """
    command = 'screen -m -d '
    if screen_name:
        command += f'-S {screen_name} '
    command += cmd
    ccall(command)


def ccopy(src, dest):
    """
    Cluster COPY. Copies file from src to all destinations

    Args:
        src: Path to source file
        dest: Full path to remote destination
    """
    if os.path.isdir(src):
        base = 'scp -r'
    elif os.path.isfile(src):
        base = 'scp'
    else:
        raise RuntimeError('SRC is neither a file nor a directory')

    commands = [f'{base} {src} {USER}@{ip}:{dest}' for ip in IPS]
    parallel_popen(commands)


def ccollect(src, dest, is_dir=False):
    """
    Cluster COLLECT. Collects files from all srcs to local destination

    Args:
        src: Full path to remote source
        dest: Local destination
    """
    base = 'scp -r' if is_dir else 'scp'
    commands = [f'{base} {USER}@{ip}:{src} {dest}' for ip in IPS]
    parallel_popen(commands)


def split_copy(src, dest):
    """
    Splits src into N number of files and send one to each remote destinations

    Args:
        src: Path to source file
        dest: Full path to destination
    """
    ext = os.path.splitext(src)[1]
    basename = os.path.basename(src)
    fpaths = [f'{i}{ext}' for i in range(len(IPS))]
    with open(src, 'r') as inp:
        files = [open(fpath, 'w') for fpath in fpaths]
        for i, line in enumerate(inp):
            files[i % len(IPS)].write(line)
        for f in files:
            f.close()

    # Copy one piece of the file to each IP
    commands = [f'scp {fpath} {USER}@{ip}:{dest}' for fpath, ip in zip(fpaths, IPS)]
    parallel_popen(commands)
    # Unify names
    commands = [f'ssh {USER}@{ip} mv {dest}/{fpath} {dest}/{basename}' for fpath, ip in zip(fpaths, IPS)]
    parallel_popen(commands)
    [os.remove(x) for x in fpaths]


def parallel_popen(commands):
    processes = [Popen(cmd, shell=True) for cmd in commands]
    [p.wait() for p in processes]
