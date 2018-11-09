import gzip
import os
import re
import subprocess
from typing import Tuple, List

from tqdm import tqdm


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


def combine_paired(fastqs: List[str], out_dir: str = None, name: str = None):
    """
    Combine a list of paired fastqs into a single fastq pair

    Args:
        fastqs: List fastq filepaths
        out_dir: Output directory
        name: Name to prepend to output file (will be joined with a '_' automatically)
    """
    r1, r2 = [], []
    # Pattern convention: Look for "R1" / "R2" in the filename, or "_1" / "_2" before the extension
    pattern = re.compile('(?:^|[._-])(R[12]|[12]\.f)')
    for fastq in sorted(fastqs):
        match = pattern.search(os.path.basename(fastq))
        if not match:
            raise RuntimeError(f'FASTQ file name fails to meet required convention for paired reads {fastq}')
        elif '1' in match.group():
            r1.append(fastq)
        elif '2' in match.group():
            r2.append(fastq)
        else:
            assert False, match.group()
    assert len(r1) == len(r2), f'Check fastq names, uneven number of pairs found.\nr1: {r1}\nr2: {r2}'
    # Concatenate fastqs
    command = 'zcat' if r1[0].endswith('.gz') and r2[0].endswith('.gz') else 'cat'

    # If sample is already a single R1 / R2 fastq
    out_dir = os.getcwd() if out_dir is None else out_dir
    name = '' if name is None else name.rstrip('_') + '_'
    with open(os.path.join(out_dir, name + 'R1.fastq'), 'w') as f1:
        p1 = subprocess.Popen([command] + r1, stdout=f1)
    with open(os.path.join(out_dir, name + 'R2.fastq'), 'w') as f2:
        p2 = subprocess.Popen([command] + r2, stdout=f2)
    p1.wait()
    p2.wait()


def combine_single_end(fastqs: List[str], out_dir: str = None, name: str = None):
    """
    Combine a list of single-end fastqs into a single fastq

    Args:
        fastqs: List fastq filepaths
        out_dir: Output directory
        name: Name to prepend to output file (will be joined with a '_' automatically)
    """
    out_dir = os.getcwd() if out_dir is None else out_dir
    name = '' if name is None else name.rstrip('_') + '_'
    command = 'zcat' if fastqs[0].endswith('.gz') else 'cat'
    with open(os.path.join(out_dir, name + 'R1.fastq'), 'w') as f:
        subprocess.check_call([command] + fastqs, stdout=f)
