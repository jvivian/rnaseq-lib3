#!/usr/bin/env bash

# Vars
ZONE=us-west-2a
CLUSTER_NAME=jvivian-ccle-processing

# Launch leader of cluster
toil launch-cluster ${CLUSTER_NAME} --keyPairName jtvivian@gmail.com --leaderNodeType t2.medium --zone ${ZONE}

# Copy workflow to leader
toil rsync-cluster --zone ${ZONE} ${CLUSTER_NAME} process_ccle_toil.py :/root

# SSH into leader
toil ssh-cluster --zone ${ZONE} jvivian-ccle-processing

# Launch once on the leader
python /root/process_ccle_toil.py aws:us-west-2:jvivian-ccle-jobstore --provisioner aws \
        --nodeTypes r4.4xlarge --maxNodes 1 --nodeStorage 150 --batchSystem mesos --restart
