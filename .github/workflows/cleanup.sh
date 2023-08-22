#!/bin/bash

function docker_rm()
{
    local name=$1
    [ -z $name ] && return 0
    echo "Removing docker container ${name}"
    docker rm -vf $name
}

docker_rm $NNTC_CONTAINER
docker_rm $MLIR_CONTAINER