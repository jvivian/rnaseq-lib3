import os
import shutil
import subprocess
import tarfile

import boto3

from rnaseq_lib3.fastq import pair_fastq


def download_pair_upload(key, down_bucket, up_bucket):
    # Make tmpdir and download
    print('\tDownloading file')
    name = key.replace('.tar.gz', '')
    tmpdir = os.path.abspath(os.path.join(os.getcwd(), name))
    os.mkdir(tmpdir)
    tar_path = os.path.join(tmpdir, key)
    down_bucket.download_file(key, tar_path)

    # Untar
    print('\tUntaring file')
    subprocess.check_call(['tar', '-zxvf', tar_path, '-C', tmpdir])
    os.remove(tar_path)

    # Identify fastq pairs
    r1 = os.path.join(tmpdir, 'R1.fastq.gz')
    r2 = os.path.join(tmpdir, 'R2.fastq.gz')

    # Fastq_pair
    print('\tPairing data')
    pair_fastq(r1, r2)
    os.remove(r1)
    os.remove(r2)

    # Gzip in parallel
    r1 = os.path.join(tmpdir, 'R1.paired.fastq')
    r2 = os.path.join(tmpdir, 'R2.paired.fastq')
    print('\tZipping fastqs')
    p1 = subprocess.Popen(['gzip', r1])
    p2 = subprocess.Popen(['gzip', r2])
    p1.wait(), p2.wait()

    # Tar paired data
    print('\tCreating Tar')
    tarball_files(key, file_paths=[r1 + '.gz', r2 + '.gz'], output_dir=tmpdir)

    # Upload to S3
    print('\tUploading to S3')
    s3.meta.client.upload_file(tar_path, up_bucket, key)

    # Delete tmpdir
    shutil.rmtree(tmpdir)


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


# Establish session
session = boto3.session.Session(profile_name='nih')
s3 = session.resource('s3')
upload_bucket = 'jvivian-ccle-data'

# Identify CCLE Bucket
ccle = s3.Bucket('cgl-ccle-data')

# Grab objects from CCLE bucket to not download duplicates
ccle_up = s3.Bucket(upload_bucket)
processed_keys = set([obj.key for obj in ccle_up.objects.all()])

# Iterate through CCLE objects
failed = []
for obj in ccle.objects.all():
    if obj.key not in processed_keys:
        if not obj.key.startswith('output') and obj.key.endswith('.tar.gz'):
            print(f'Processing: {obj.key}')
            try:
                download_pair_upload(obj.key, down_bucket=ccle, up_bucket=upload_bucket)
            except:
                print(f'Failed to process {obj.key}')
                failed.append(obj.key)
    else:
        print(f'Processed {obj.key} already')

# Output failed samples
if failed:
    with open('failed_samples.txt', 'w') as f:
        f.write('\n'.join(failed))