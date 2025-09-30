# src/ai/services/frame2video_services/client.py

import grpc
import cv2
import numpy as np
from loguru import logger
import os
import multiprocessing as mp  # <-- IMPORTANT: Import multiprocessing

# --- All your other necessary imports ---
from src.streaming.pb import streaming_pb2, streaming_pb2_grpc
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.utils.transfrom_data import matrix_list_to_numpy

# --- Constants ---
DELAY_TIME = float(os.getenv("DELAY_TIME", 0.0))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (12, 14), (14, 16), (16, 18),
    (23, 24), (24, 26), (26, 28), (28, 32), (23, 25), (25, 27), (27, 29), (29, 31)
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
]

def visualize_landmarks_minimal(array, target_height=480, target_width=720, line_thickness=1):
    if array.shape != (1, 75, 3):
        raise ValueError(f"Expected shape (1,75,3), got {array.shape}")

    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    points = (array.squeeze(0)[:, :2] * [target_width, target_height]).astype(np.int32)

    # Replace all (0, 0) points with (target_width//2, target_height)
    zero_mask = (points[:, 0] <= 0) & (points[:, 1] <= 0)
    points[zero_mask] = [target_width // 2, target_height]

    def draw_landmarks(landmarks, color):
        valid = ~np.isnan(landmarks[:, 0])
        for x, y in landmarks[valid]:
            cv2.circle(img, (x, y), 4, color, -1)

    def draw_connections(landmarks, connections, color):
        valid_connections = [(start, end) for start, end in connections if not np.isnan(landmarks[[start, end]]).any()]
        for start_idx, end_idx in valid_connections:
            cv2.line(img, tuple(landmarks[start_idx]), tuple(landmarks[end_idx]), color, line_thickness)

    # Use multithreading for parallel execution
    draw_landmarks(points[:33], (0, 255, 0))
    draw_connections(points[:33], POSE_CONNECTIONS, (0, 255, 0))
    draw_landmarks(points[33:54], (255, 0, 0))
    draw_connections(points[33:54], HAND_CONNECTIONS, (255, 0, 0))
    draw_landmarks(points[54:], (0, 0, 255))
    draw_connections(points[54:], HAND_CONNECTIONS, (0, 0, 255))
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 1]
    success, encoded_img = cv2.imencode('.webp', img, encode_param)
    if not success:
        raise RuntimeError("Failed to encode image")
    return encoded_img.tobytes()


# --- Process 1: AI Task ---
def ai_and_receiver_task(render_queue: mp.Queue):
    """Receives frames from gRPC, processes them with AI, and puts them in the queue."""
    handler = HandleConcatFrame()
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        logger.info("[AI Process] Starting...")
        for response in pop_frame_response:
            if response.request_status == "Success":
                frames = matrix_list_to_numpy(response.frame)
                handler.push_into_process_queue(frames)
                while handler.getLen() > 20:
                    processed_frame = handler.pop()
                    if processed_frame is not None:
                        render_queue.put(processed_frame)


# --- Process 2: Renderer Task ---
def renderer_and_sender_task(render_queue: mp.Queue):
    """Gets processed frames from the queue, renders them to images, and sends them."""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel)
        batch = []
        logger.info("[Render Process] Starting...")
        while True:
            processed_frame = render_queue.get()
            if processed_frame is None:
                continue
            image_bytes = visualize_landmarks_minimal(processed_frame)
            batch.append(image_bytes)
            if len(batch) >= BATCH_SIZE:
                stub.BatchPushImage(streaming_pb2.BatchPushImageRequest(images=batch))
                batch.clear()


# -----------------------------------------------------------------
# MAIN LAUNCHER: This replaces your old run() function
# -----------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting the multiprocessing pipeline...")

    # This is a good practice for stability, especially on macOS and Windows
    mp.set_start_method('spawn')

    # 1. Create the shared queue that connects the two processes
    render_queue = mp.Queue(maxsize=200)

    # 2. Create the two processes from our task functions
    ai_process = mp.Process(target=ai_and_receiver_task, args=(render_queue,))
    render_process = mp.Process(target=renderer_and_sender_task, args=(render_queue,))

    # 3. Start both processes to run in parallel
    ai_process.start()
    render_process.start()
    logger.info("AI and Renderer processes are running.")

    # 4. Wait for them to finish before the script exits
    ai_process.join()
    render_process.join()
    logger.info("Pipeline finished.")