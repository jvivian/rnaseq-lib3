import os
import subprocess
import multiprocessing


def fix_permissions(tool: str, work_dir: str):
    """Fix permissions of a mounted docker directory by reusing the tool"""
    base_docker_call = get_base_call(work_dir)
    base_docker_call.append('--entrypoint=chown')
    stat = os.stat(work_dir)
    command = base_docker_call + [tool] + ['-R', '{}:{}'.format(stat.st_uid, stat.st_gid), '/data']
    subprocess.check_call(command)


def get_base_call(mount_path: str):
    """Return base docker call with mounted path"""
    return ['docker', 'run',
            '--log-driver=none',
            '-v', '{}:/data'.format(os.path.abspath(mount_path))]


def run_macs(pair):
    p, c = pair
    print(f'Running: {p}\t{c}')
    sample = p.split('.deduplicated.bam')[0]
    out = os.path.join(out_dir, sample)
    if os.path.exists(out):
        print('Output already exists')
        return
    os.makedirs(out)

    parameters = base_params + ['-t', f'/data/BAMs/{p}',
                                '-c', f'/data/BAMs/{c}',
                                '-n', sample,
                                '--outdir', f'/data/MACS-output/{sample}']
    subprocess.check_call(parameters)
    fix_permissions(tool, out)


bam_dir = '/mnt/CEGX/BAMs'
out_dir = '/mnt/CEGX/MACS-output/'
tool = 'dceoy/macs2'

bams = os.listdir(bam_dir)
pd = sorted([x for x in bams if 'PC.deduplicated.bam' in x])
ic = sorted([x for x in bams if 'IC.deduplicated.bam' in x])

base_params = get_base_call('/mnt/CEGX') + [tool, 'callpeak']

# Parallelize run
p = multiprocessing.Pool(10)
p.map(run_macs, zip(pd, ic))
