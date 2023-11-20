#!/bin/bash

# Build docker container
docker compose -f docker-compose.yml build 

# Run docker container in background
xhost +local:root

# Download weight file
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth

docker compose run --rm res_blip