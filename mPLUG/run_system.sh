#!/bin/bash

# Build docker container
docker compose -f docker-compose.yml build 

# Run docker container in background
docker compose -f docker-compose.yml up -d

xhost +local:root
# Exec docker container
docker exec -it res_mplug /bin/bash