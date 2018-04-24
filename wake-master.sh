#! /bin/bash

docker-machine start aws01
docker-machine regenerate-certs -f aws01
docker-machine ssh aws01 "sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-3525959c.efs.us-west-2.amazonaws.com:/ /efs"
eval $(docker-machine env aws01)
echo "Running mongo..."
docker run --rm --volumes-from mongostorage -d -p 27017:27017 mongo
echo "Master machine (aws01) ready."