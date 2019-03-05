# Install basics
sudo apt-get update && sudo apt-get install -y git build-essential

# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
sh Anaconda3-2018.12-Linux-x86_64.sh

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker ubuntu

## LOG OUT LOG BACK IN

# Create py2 env
conda create --name toil python=2.7
source activate toil
pip install toil
