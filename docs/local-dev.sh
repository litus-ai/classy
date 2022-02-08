#!/bin/bash

set -e

docker ps -a
read -rp "If a container was already created (and you wish to re-use it), enter container id: " container_id

if [ -z "$container_id" ]; then
  echo "Starting docker"
  container_id=$(docker run --name classy-docs-website -v $(pwd):/classy -p 30000:30000 -itd node:16-bullseye bash)
  docker exec $container_id bash -c "apt-get update && apt-get install -y python3-pip"
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
  sudo chown -R $USER:$USER docs
}

docker exec $container_id bash -c "
  cd /classy && \
  pip install -r requirements.txt && \
  pip install -r <(python3 classy/optional_deps.py) && \
  pdoc -f --template-dir docs/pdoc/templates -o docs/docs classy && \
  rm -rf docs/docs/api && \
  mv docs/docs/classy docs/docs/api && \
  python3 docs/pdoc/pdoc_postprocess.py && \
  cd docs && \
  yarn install && \
  yarn docusaurus clear && \
  yarn run start -p 30000 -h 0.0.0.0"
