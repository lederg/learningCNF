#! /bin/bash

echo name is $1
# aws01 is master machine

# docker-machine ssh aws01 "cd /efs/shared-code/learningCNF; git pull"
# echo docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code /learningcnf:v5 ${@:2}"
# docker-machine ssh $1 "sudo docker run --rm -v /efs/shared-code/learningCNF:/code /learningcnf:v5 ${@:2}"

echo docker-machine ssh $1 "sudo docker run -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -d -v /home/ubuntu/learningCNF:/code -v /efs/runs_cadet/:/code/runs_cadet -v /home/ubuntu/synthetic_qbf_formulas:/code/data -v /efs/saved_models:/code/saved_models -v /efs/eps:/code/eps gilled/learningcnf:latest ${@:2}"
docker-machine ssh $1 "sudo docker run -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -d -v /home/ubuntu/learningCNF:/code -v /efs/runs_cadet/:/code/runs_cadet -v /home/ubuntu/synthetic_qbf_formulas:/code/data -v /efs/saved_models:/code/saved_models -v /efs/eps:/code/eps gilled/learningcnf:latest ${@:2}"

