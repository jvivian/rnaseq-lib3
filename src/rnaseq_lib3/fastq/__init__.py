import gzip
import multiprocessing
import os
from subprocess import Popen, PIPE
from typing import Tuple

from tqdm import tqdm

from rnaseq_lib3.docker import get_base_call, fix_permissions


# Adopted from: https://github.com/linsalrob/EdwardsLab/blob/master/bin/pair_fastq_fast.py
def pair_fastq(r1_path: str, r2_path: str, output_singles: bool = False) -> None:
    # read the first file into a data structure
    seqs = {}
    for seqid, header, seq, qual in tqdm(_stream_fastq(r1_path)):
        seqid = seqid.replace('.1', '')
        seqs[seqid] = [header, seq, qual]

    lp = open("{}.paired.fastq".format(r1_path.replace('.fastq', '').replace('.gz', '')), 'wt')
    rp = open("{}.paired.fastq".format(r2_path.replace('.fastq', '').replace('.gz', '')), 'wt')
    if output_singles:
        lu = open("{}.singles.fastq".format(r1_path.replace('.fastq', '').replace('.gz', '')), 'w')
        ru = open("{}.singles.fastq".format(r2_path.replace('.fastq', '').replace('.gz', '')), 'w')

    # read the first file into a data structure
    seen = set()
    for seqid, header, seq, qual in tqdm(_stream_fastq(r2_path), total=len(seqs)):
        seqid = seqid.replace('.2', '')
        if seqid in seqs:
            # Remove from seqs dict for memory
            header_l, seq_l, qual_l = seqs.pop(seqid)
            lp.write("@" + header_l + "\n" + seq_l + "\n+\n" + qual_l + "\n")
            rp.write("@" + header + "\n" + seq + "\n+\n" + qual + "\n")
        elif output_singles:
            seen.add(seqid)
            ru.write("@" + header + "\n" + seq + "\n+\n" + qual + "\n")

    if output_singles:
        for seqid in seqs:
            if seqid not in seen:
                lu.write("@" + seqs[seqid][0] + "\n" + seqs[seqid][1] + "\n+\n" + seqs[seqid][2] + "\n")
                seqs.pop(seqid)
        lu.close()
        ru.close()
    lp.close()
    rp.close()


def _stream_fastq(fqfile: str) -> Tuple[str, str, str, str]:
    """Read a fastq file and provide an iterable of the sequence ID, the
    full header, the sequence, and the quaity scores.

    Note that the sequence ID is the header up until the first space,
    while the header is the whole header.
    """

    if fqfile.endswith('.gz'):
        qin = gzip.open(fqfile, 'rb')
    else:
        qin = open(fqfile, 'r')

    while True:
        header = qin.readline()
        if not header:
            break
        header = header.decode('utf-8').strip()
        seqidparts = header.split(' ')
        seqid = seqidparts[0]
        seq = qin.readline()
        seq = seq.decode('utf-8').strip()
        qin.readline()
        qualscores = qin.readline()
        qualscores = qualscores.decode('utf-8').strip()
        header = header.replace('@', '', 1)
        yield seqid, header, seq, qualscores


def download_SRA(sra_id: str, work_dir: str = None, threads: int = None):
    work_dir = os.getcwd() if work_dir is None else os.path.abspath(work_dir)
    threads = multiprocessing.cpu_count() if threads is None else threads

    # Params
    base_call = get_base_call(work_dir)
    tool = 'nunoagostinho/parallel-fastq-dump'
    parameters = ['parallel-fastq-dump',
                  '--sra-id', sra_id,
                  '--threads', str(threads),
                  '--outdir', '/data',
                  '--tmpdir', '/data',
                  '--gzip', '--skip-technical', '--readids', '--read-filter', 'pass',
                  '--dumpbase', '--split-files', '--clip']
    p = Popen(base_call + [tool] + parameters, stderr=PIPE, stdout=PIPE, universal_newlines=True)
    out, err = p.communicate()
    if p.returncode != 0:
        print(out, err)
    fix_permissions(tool, work_dir=work_dir)
