#!/bin/bash

# Build docker container
docker compose -f docker-compose.yml build 

# Run docker container in background
xhost +local:root
docker compose run --rm res_blip