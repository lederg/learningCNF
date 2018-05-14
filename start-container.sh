#! /bin/bash

echo name is $1
# aws01 is master machine

docker-machine ssh aws01 "cd /efs/shared-code/learningCNF; git pull"
echo docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code gilled/learningcnf:v5 ${@:2}"
docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code gilled/learningcnf:v5 ${@:2}"

