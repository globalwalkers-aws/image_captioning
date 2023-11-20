#!/bin/bash

# Build docker container
docker compose -f docker-compose.yml build 

# Run docker container in background
xhost +local:root

# Download weight file
wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_base.pth

docker compose run --rm res_mplug