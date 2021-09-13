#!/bin/bash

set -e

docker ps -a
read -rp "If a container was already created (and you wish to re-use it), enter container id: " container_id

if [ -z "$container_id" ]; then
  echo "Starting docker"
  container_id=$(docker run --name classy-docs-website -v $(pwd):/docs -p 30000:30000 -itd node:16-bullseye bash)
  echo "Started container with id $container_id"
fi

IS_CONTAINER_RUNNING=$(docker ps -q --filter "id=$container_id")
if [ -z "${IS_CONTAINER_RUNNING}" ]; then
  echo "Resuming container with id $container_id"
  docker start $container_id
fi

trap ctrl_c INT
function ctrl_c() {
  echo "Stopping container with id $container_id"
  docker stop $container_id
  echo "Resetting file permissions"
  sudo chown -R $USER:$USER ../docs
}

docker exec $container_id bash -c "cd /docs && yarn start -p 30000 -h 0.0.0.0"