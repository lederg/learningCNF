#! /bin/bash

docker-machine create --driver amazonec2 --amazonec2-instance-type t2.2xlarge --amazonec2-region "us-west-2" $1
docker-machine ssh $1 "sudo mkdir /efs; sudo apt-get install -y nfs-common"
docker-machine ssh $1 "sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-3525959c.efs.us-west-2.amazonaws.com:/ /efs"
docker-machine ssh $1 "gunzip -c /efs/image/learningcnf.tar.gz | sudo docker load"


# Our AMI already has nfs and the correct docker images, though we may want to get the latest one..
# docker-machine create --driver amazonec2  --amazonec2-ami ami-4c7dd834 --amazonec2-instance-type t2.2xlarge --amazonec2-region "us-west-2" $1
# docker-machine ssh $1 "sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-3525959c.efs.us-west-2.amazonaws.com:/ /efs"
