#!/usr/bin/env bash
set -eo pipefail

ORG="williamjsmith15"
DATETIME=$(date +%Y%m%d-%H%M%S)

build_and_push() {
    local name=$1
    local dockerfile=$2

    docker build -t "$name:build" -f "$dockerfile" .

    for tag in latest "$DATETIME"; do
        docker tag "$name:build" "$ORG/$name:$tag"
        docker push "$ORG/$name:$tag"
    done

    docker rmi "$name:build"
}

build_and_push parafem-meshgen     contianers/Dockerfile.meshgen
build_and_push parafem-p123        contianers/Dockerfile.p123
build_and_push parafem-bcgen       contianers/Dockerfile.bcgen
build_and_push parafem-postprocess contianers/Dockerfile.postprocess
build_and_push parafem-vtu         contianers/Dockerfile.vtu
