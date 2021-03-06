import multiprocessing
import os
from subprocess import Popen, PIPE

from rnaseq_lib3.docker import get_base_call, fix_permissions
from rnaseq_lib3.utils import curl


def fastq_dump(sra_id: str, work_dir: str = None, threads: int = None, split: str ='--split-3'):
    """
    Wrapper for fastq-dump

    Args:
        sra_id: SRA run ID
        work_dir: Output directory
        threads: Number of cores to use
        split: split argument

    Returns:
        If paired, will attempt to return filepaths to paired files
    """
    work_dir = os.getcwd() if work_dir is None else os.path.abspath(work_dir)
    work_dir = os.path.join(work_dir, sra_id)
    os.makedirs(work_dir, exist_ok=True)

    # Prefetch SRA file
    if len(os.listdir(work_dir)) > 0:
        print(f'Files in directory {work_dir}')
        return 0
    print(f'Prefetching SRA file: {sra_id}')
    sra_path = _prefetch(sra_id, work_dir)

    # Params
    print('\tRunning parallel-fastq-dump')
    base_call = get_base_call(work_dir)
    tool = 'nunoagostinho/parallel-fastq-dump'
    threads = multiprocessing.cpu_count() if threads is None else threads
    parameters = ['parallel-fastq-dump',
                  '--threads', str(threads),
                  '--tmpdir', '/data',
                  '--outdir', '/data',
                  '--gzip', '--skip-technical', '--readids', '--read-filter', 'pass',
                  '--dumpbase', '--clip', split,
                  '-s', f'/data/{os.path.basename(sra_path)}']
    p = Popen(base_call + [tool] + parameters, stderr=PIPE, stdout=PIPE, universal_newlines=True)
    out, err = p.communicate()
    if p.returncode != 0:
        print(out, err)
    else:
        os.remove(sra_path)
        fix_permissions(tool, work_dir=work_dir)
        fastqs = os.listdir(work_dir)
        try:
            r1 = [x for x in fastqs if x.endswith('_1.fastq.gz')][0]
            r2 = [x for x in fastqs if x.endswith('_2.fastq.gz')][0]
            return r1, r2
        except IndexError:
            print(f'\tSample {sra_id} does not appear paired. {fastqs}')
            return 1


def _prefetch(sra_id: str, work_dir: str = None):
    """Prefetch SRA file via URL as its much faster than using fastq-dump"""
    # Construct SRA URL
    base = 'ftp://ftp-trace.ncbi.nih.gov'
    p1, p2 = sra_id[:3], sra_id[:6]
    address = f'sra/sra-instant/reads/ByRun/sra/{p1}/{p2}/{sra_id}/{sra_id}.sra'
    url = os.path.join(base, address)
    # Call cURL and return file_path
    return curl(url, work_dir)
