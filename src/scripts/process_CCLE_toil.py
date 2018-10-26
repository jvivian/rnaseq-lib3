import os
import subprocess
import tarfile

import boto3
from toil.common import Toil
from toil.job import Job

from rnaseq_lib3.fastq import pair_fastq


def workflow(job, key, download_bucket_name, upload_bucket_name):
    # Wrap job functions
    download = job.wrapJobFn(download_sample, key, download_bucket_name, disk='50G', cores=2, memory='10G')
    pair = job.wrapJobFn(pair_fastqs, download.rv(0), download.rv(1), cores=2, memory='70G', disk='60G')
    upload = job.wrapJobFn(tar_and_upload, pair.rv(0), pair.rv(1), key, upload_bucket_name,
                           cores=1, memory='10G', disk='30G')

    # Wire
    job.addChild(download)
    download.addChild(pair)
    pair.addChild(upload)


def download_sample(job, key, download_bucket_name):
    job.log('Downloading File: ' + key)

    # Session
    session = boto3.session.Session()
    s3 = session.resource('s3')
    download_bucket = s3.Bucket(download_bucket_name)

    tar_path = os.path.join(job.tempDir, key)
    download_bucket.download_file(key, tar_path)

    job.log('Untaring Sample: ' + tar_path)
    subprocess.check_call(['tar', '-zxvf', tar_path, '-C', job.tempDir])

    job.log('Gunzipping Fastqs')
    r1 = os.path.join(job.tempDir, 'R1.fastq.gz')
    r2 = os.path.join(job.tempDir, 'R2.fastq.gz')
    p1 = subprocess.Popen(['gunzip', r1])
    p2 = subprocess.Popen(['gunzip', r2])
    p1.wait(), p2.wait()

    r1 = r1.replace('.gz', '')
    r2 = r2.replace('.gz', '')

    return job.fileStore.writeGlobalFile(r1), job.fileStore.writeGlobalFile(r2)


def pair_fastqs(job, r1_id, r2_id):
    r1_path = job.fileStore.readGlobalFile(r1_id, os.path.join(job.tempDir, 'R1.fastq'))
    r2_path = job.fileStore.readGlobalFile(r1_id, os.path.join(job.tempDir, 'R1.fastq'))

    job.log('Pairing fastqs')
    pair_fastq(r1_path, r2_path)
    job.deleteGlobalFile(r1_id), job.deleteGlobalFile(r2_id)

    job.log('Gzipping fastqs')
    r1 = os.path.join(job.tempDir, 'R1.paired.fastq')
    r2 = os.path.join(job.tempDir, 'R2.paired.fastq')
    p1 = subprocess.Popen(['gzip', r1])
    p2 = subprocess.Popen(['gzip', r2])
    p1.wait(), p2.wait()

    return job.fileStore.writeGlobalFile(r1 + '.gz'), job.fileStore.writeGlobalFile(r2 + '.gz')


def tar_and_upload(job, r1_id, r2_id, key, upload_bucket_name):
    r1 = job.fileStore.readGlobalFile(r1_id, os.path.join(job.tempDir, 'R1.fastq.gz'))
    r2 = job.fileStore.readGlobalFile(r2_id, os.path.join(job.tempDir, 'R1.fastq.gz'))

    job.log('Tar files')
    tarball_files(key, file_paths=[r1 + '.gz', r2 + '.gz'], output_dir=job.tempDir)
    tar_path = os.path.join(job.tempDir, key)

    job.log('Uploading to S3')
    session = boto3.session.Session()
    s3 = session.resource('s3')
    s3.meta.client.upload_file(tar_path, upload_bucket_name, key)


def tarball_files(tar_name, file_paths, output_dir='.', prefix=''):
    """
    Creates a tarball from a group of files

    :param str tar_name: Name of tarball
    :param list[str] file_paths: Absolute file paths to include in the tarball
    :param str output_dir: Output destination for tarball
    :param str prefix: Optional prefix for files in tarball
    """
    with tarfile.open(os.path.join(output_dir, tar_name), 'w:gz') as f_out:
        for file_path in file_paths:
            if not file_path.startswith('/'):
                raise ValueError('Path provided is relative not absolute.')
            arcname = prefix + os.path.basename(file_path)
            f_out.add(file_path, arcname=arcname)


def map_job(job, func, inputs, *args):
    """
    Spawns a tree of jobs to avoid overloading the number of jobs spawned by a single parent.
    This function is appropriate to use when batching samples greater than 1,000.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param function func: Function to spawn dynamically, passes one sample as first argument
    :param list inputs: Array of samples to be batched
    :param list args: any arguments to be passed to the function
    """
    # num_partitions isn't exposed as an argument in order to be transparent to the user.
    # The value for num_partitions is a tested value
    num_partitions = 100
    partition_size = len(inputs) / num_partitions
    if partition_size > 1:
        for partition in partitions(inputs, partition_size):
            job.addChildJobFn(map_job, func, partition, *args)
    else:
        for sample in inputs:
            job.addChildJobFn(func, sample, *args)


def partitions(l, partition_size):
    """
    >>> list(partitions([], 10))
    []
    >>> list(partitions([1,2,3,4,5], 1))
    [[1], [2], [3], [4], [5]]
    >>> list(partitions([1,2,3,4,5], 2))
    [[1, 2], [3, 4], [5]]
    >>> list(partitions([1,2,3,4,5], 5))
    [[1, 2, 3, 4, 5]]

    :param list l: List to be partitioned
    :param int partition_size: Size of partitions
    """
    for i in xrange(0, len(l), partition_size):
        yield l[i:i + partition_size]


def main():
    # Establish session
    session = boto3.session.Session()
    s3 = session.resource('s3')

    # Grab objects from upload bucket to not download duplicates
    upload_bucket_name = 'jvivian-ccle-data'
    upload_bucket = s3.Bucket(upload_bucket_name)
    processed_keys = set([obj.key for obj in upload_bucket.objects.all()])

    # Collect all keys to be processed
    download_bucket_name = 'cgl-ccle-data'
    download_bucket = s3.Bucket(download_bucket_name)
    keys = [x.key for x in download_bucket.objects.all() if x not in processed_keys]
    keys = [x for x in keys if not x.startswith('output') and x.endswith('.tar.gz')]

    # Start Toil run
    parser = Job.Runner.getDefaultArgumentParser()
    options = parser.parse_args()
    with Toil(options) as toil:
        toil.start(Job.wrapJobFn(map_job, workflow, keys, download_bucket=download_bucket_name,
                                 upload_bucket_name=upload_bucket_name))


if __name__ == '__main__':
    main()
