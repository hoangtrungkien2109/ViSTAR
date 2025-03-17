#!/bin/bash

echo "Stopping all services..."

# Kill all Python processes
taskkill //IM python.exe //F

echo "All services stopped."
