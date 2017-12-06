#! /bin/bash

eval "$(docker-machine env $1)"
docker ps
