import gzip
from typing import Tuple

from tqdm import tqdm


# Adopted from: https://github.com/linsalrob/EdwardsLab/blob/master/bin/pair_fastq_fast.py


def pair_fastq(r1_path: str, r2_path: str, output_singles: bool = False) -> None:
    # read the first file into a data structure
    seqs = {}
    for seqid, header, seq, qual in tqdm(stream_fastq(r1_path)):
        seqid = seqid.replace('.1', '')
        seqs[seqid] = [header, seq, qual]

    lp = open("{}.paired.fastq".format(r1_path.replace('.fastq', '').replace('.gz', '')), 'w')
    rp = open("{}.paired.fastq".format(r2_path.replace('.fastq', '').replace('.gz', '')), 'w')
    if output_singles:
        lu = open("{}.singles.fastq".format(args.l.replace('.fastq', '').replace('.gz', '')), 'w')
        ru = open("{}.singles.fastq".format(args.r.replace('.fastq', '').replace('.gz', '')), 'w')

    # read the first file into a data structure
    seen = set()
    for seqid, header, seq, qual in tqdm(stream_fastq(r2_path), total=len(seqs)):
        seqid = seqid.replace('.2', '')
        seen.add(seqid)
        if seqid in seqs:
            lp.write("@" + seqs[seqid][0] + "\n" + seqs[seqid][1] + "\n+\n" + seqs[seqid][2] + "\n")
            rp.write("@" + header + "\n" + seq + "\n+\n" + qual + "\n")
        elif output_singles:
            ru.write("@" + header + "\n" + seq + "\n+\n" + qual + "\n")

    if output_singles:
        for seqid in seqs:
            if seqid not in seen:
                lu.write("@" + seqs[seqid][0] + "\n" + seqs[seqid][1] + "\n+\n" + seqs[seqid][2] + "\n")
        lu.close()
        ru.close()
    lp.close()
    rp.close()


def stream_fastq(fqfile: str) -> Tuple(str, str, str, str):
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
        header = header.strip()
        seqidparts = header.split(' ')
        seqid = seqidparts[0]
        seq = qin.readline()
        seq = seq.strip()
        qin.readline()
        qualscores = qin.readline()
        qualscores = qualscores.strip()
        header = header.replace('@', '', 1)
        yield seqid, header, seq, qualscores
