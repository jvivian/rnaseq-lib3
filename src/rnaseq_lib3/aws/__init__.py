import boto3


# S3
def list_all_buckets(profile_name='default'):
    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')
    for bucket in s3.buckets.all():
        print(bucket)


def list_bucket(bucket_name, profile_name='default'):
    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        print(obj)


def delete_bucket(bucket_name, profile_name='default'):
    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    # Delete all objects in bucket then the bucket
    bucket = s3.Bucket(bucket_name)
    bucket_objs = bucket.objects.all()
    bucket_objs.delete()
    bucket.delete()


# EC2
# Adapted from: https://gist.github.com/mda590/679aba60ca03699d5b12a32314debdc0
def list_running_instances(profile_name='default', region_name='us-west-2'):
    session = boto3.session.Session(profile_name=profile_name)
    ec2 = session.resource('ec2', region_name=region_name)

    # Define running filters
    filters = [{'Name': 'instance-state-name',
                'Values': ['running']}]

    instances = ec2.instances.filter(Filters=filters)
    for instance in instances:
        print(instance.id)
