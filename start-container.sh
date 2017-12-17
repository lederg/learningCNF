#! /bin/bash

echo name is $1
docker-machine ssh $1 \"sudo docker run --rm -v /efs/shared-code/learningCNF:/code gilled/learningcnf:v2 ${@:2}\"

