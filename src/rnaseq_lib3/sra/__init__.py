import multiprocessing
import os
from subprocess import Popen, check_call, PIPE

from rnaseq_lib3.docker import get_base_call, fix_permissions


def fastq_dump(sra_id: str, work_dir: str = None, threads: int = None):
    work_dir = os.getcwd() if work_dir is None else os.path.abspath(work_dir)
    work_dir = os.path.join(work_dir, sra_id)
    os.makedirs(work_dir, exist_ok=True)

    # Prefetch SRA file
    sra_path = _prefetch(sra_id, work_dir)

    # Params
    base_call = get_base_call(work_dir)
    tool = 'nunoagostinho/parallel-fastq-dump'
    threads = multiprocessing.cpu_count() if threads is None else threads
    parameters = ['parallel-fastq-dump',
                  '--threads', str(threads),
                  '--tmpdir', '/data',
                  '--outdir', '/data',
                  '--gzip', '--skip-technical', '--readids', '--read-filter', 'pass',
                  '--dumpbase', '--split-3', '--clip',
                  '-s', f'/data/{os.path.basename(sra_path)}']
    p = Popen(base_call + [tool] + parameters, stderr=PIPE, stdout=PIPE, universal_newlines=True)
    out, err = p.communicate()
    if p.returncode != 0:
        print(out, err)
    else:
        os.remove(sra_path)
        fix_permissions(tool, work_dir=work_dir)
        fastqs = os.listdir(work_dir)
        r1 = [x for x in fastqs if x.endswith('_1.fastq.gz')][0]
        r2 = [x for x in fastqs if x.endswith('_2.fastq.gz')][0]
        return r1, r2


def _prefetch(sra_id: str, work_dir: str = None):
    """Prefect SRA file"""
    # Construct SRA URL
    base = 'ftp://ftp-trace.ncbi.nih.gov'
    p1, p2 = sra_id[:3], sra_id[:6]
    address = f'sra/sra-instant/reads/ByRun/sra/{p1}/{p2}/{sra_id}/{sra_id}.sra'
    url = os.path.join(base, address)
    # Call cURL and return file_path
    return _curl(url, work_dir)


def _curl(url: str, work_dir: str = None):
    work_dir = os.getcwd() if work_dir is None else work_dir
    file_path = os.path.join(work_dir, os.path.basename(url))
    check_call(['curl', '-fs', '--retry', '5', url, '-o', file_path])
    return file_path
