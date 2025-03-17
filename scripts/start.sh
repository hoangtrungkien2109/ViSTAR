#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting Streaming Server..."
python -m src.streaming.main > "$LOG_DIR/streaming_server.log" 2>&1 &

echo "Starting AI Client Services..."
python -m src.ai.services.text2frame_services.client > "$LOG_DIR/text2frame_client.log" 2>&1 &
python -m src.ai.services.frame2video_services.client > "$LOG_DIR/frame2video_client.log" 2>&1 &

echo "Starting FE server..."
cd src/fe && python main.py > "D:/NCKH/Text_to_Sign/ViSTAR/$LOG_DIR/fe_server.log" 2>&1 &

echo "Opening index.html..."
start D:/NCKH/Text_to_Sign/ViSTAR/src/fe/index.html
# python -m src.be.main

echo "All services started."
