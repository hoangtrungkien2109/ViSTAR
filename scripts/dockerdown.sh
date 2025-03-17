#!/bin/bash

echo "Stopping Docker containers..."
docker-compose -f docker-compose.dev.yaml down

echo "Docker stopped."
