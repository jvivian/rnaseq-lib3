import os
import subprocess
from typing import Tuple

from rnaseq_lib3.docker import get_base_call


def download_ccle(uuid: str, work_dir: str) -> str:
    """
    Download CCLE file from GDC

    Args:
        uuid: UUID of file
        work_dir: Directory to place CCLE file

    Returns:
        Path to bam
    """
    call = get_base_call(work_dir)
    parameters = ['download',
                  '-d', '/data',
                  uuid]
    tool = 'jvivian/gdc-client'
    call = call + [tool] + parameters
    subprocess.check_call(call)

    files = [x for x in os.listdir(os.path.join(work_dir, uuid)) if x.lower().endswith('.bam')]
    assert len(files) == 1, 'More than one BAM found from GDC URL: {}'.format(files)
    bam_path = os.path.join(work_dir, uuid, files[0])
    return bam_path


def convert_bam_to_fastq(bam_path: str, ignore_validation_errors: bool = True) -> Tuple[str, str]:
    """
    Convert BAM to FASTQ pair

    Args:
        bam_path: Full path to BAM
        ignore_validation_errors: Flag to ignore validation errors in picardtools

    Returns:
        Paths to R1/R2
    """
    work_dir = os.path.dirname(os.path.abspath(bam_path))
    call = get_base_call(work_dir)
    tool = 'quay.io/ucsc_cgl/picardtools:2.10.9--23fc31175415b14dbf337216f9ae14d3acc3d1eb'
    parameters = ['SamToFastq', 'I=/data/{}'.format(os.path.basename(bam_path)), 'F=/data/R1.fq', 'F2=/data/R2.fq']
    if ignore_validation_errors:
        parameters.append('VALIDATION_STRINGENCY=SILENT')
    call = call + [tool] + parameters
    subprocess.check_call(call)
    r1 = os.path.join(work_dir, 'R1.fq')
    r2 = os.path.join(work_dir, 'R2.fq')
    return r1, r2
