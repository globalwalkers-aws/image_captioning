#!/bin/bash

# Run docker container in background
# xhost +local:root

# Download weight file
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt

# Make weight file directory
mkdir -p weight_checkpoints

# Move weight file to that directory
mv caption_base_best.pt weight_checkpoints

# Run Docker
make docker-run