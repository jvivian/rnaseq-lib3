# System updates and install
sudo apt-get update
#sudo apt -y upgrade
sudo apt-get install -y python3-dev build-essential git python3-pip

# Personal library
git clone https://github.com/jvivian/rnaseq-lib3
pip3 install --user -e rnaseq-lib3
pip3 install --user boto3

# aws creds
mkdir ~/.aws

# SSD Mount (non-EBS r5d instances for example)
sudo mkfs -t xfs /dev/nvme0n1
sudo mkdir /data
sudo mount -t xfs /dev/nvme0n1 /data
sudo chown ubuntu:ubuntu /data
