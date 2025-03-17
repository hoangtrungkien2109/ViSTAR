#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting docker image..."
docker-compose -f docker-compose.dev.yaml up > "$LOG_DIR/docker.log" 2>&1 &
sleep 10

echo "Docker started."