#! /bin/bash

echo name is $1
# aws01 is master machine

# docker-machine ssh aws01 "cd /efs/shared-code/learningCNF; git pull"
# echo docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code gilled/learningcnf:v5 ${@:2}"
# docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code gilled/learningcnf:v5 ${@:2}"

echo docker-machine ssh $1 "sudo docker run -v /home/ubuntu/learningCNF:/code -v /efs/runs_cadet/:/runs_cadet -v /efs/saved_models:/saved_models gilled/learningcnf:v5 ${@:2} rl_log_dir=/runs_cadet model_dir=/saved_models"
docker-machine ssh $1 "sudo docker run -v /home/ubuntu/learningCNF:/code -v /efs/runs_cadet/:/runs_cadet -v /efs/saved_models:/saved_models gilled/learningcnf:v5 ${@:2} rl_log_dir=/runs_cadet model_dir=/saved_models"

