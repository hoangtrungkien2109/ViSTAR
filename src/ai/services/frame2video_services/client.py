import grpc
import cv2
import numpy as np
from loguru import logger
from threading import Thread
import os
from src.streaming.pb import streaming_pb2, streaming_pb2_grpc
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame
from src.ai.services.utils.transfrom_data import matrix_list_to_numpy

DELAY_TIME = float(os.getenv("DELAY_TIME", 0.0))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

POSE_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (12, 14), (14, 16), (16, 18),
    (23, 24), (24, 26), (26, 28), (28, 32), (23, 25), (25, 27), (27, 29), (29, 31)
])

HAND_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20)
])

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

    def draw_landmarks(landmarks, color):
        valid = ~np.isnan(landmarks[:, 0])
        for x, y in landmarks[valid]:
            cv2.circle(img, (x, y), 4, color, -1)

    def draw_connections(landmarks, connections, color):
        valid_connections = [(start, end) for start, end in connections if not np.isnan(landmarks[[start, end]]).any()]
        for start_idx, end_idx in valid_connections:
            cv2.line(img, tuple(landmarks[start_idx]), tuple(landmarks[end_idx]), color, line_thickness)

    # Use multithreading for parallel execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(draw_landmarks, points[:33], (0, 255, 0))
        executor.submit(draw_connections, points[:33], POSE_CONNECTIONS, (0, 255, 0))
        executor.submit(draw_landmarks, points[33:54], (255, 0, 0))
        executor.submit(draw_connections, points[33:54], HAND_CONNECTIONS, (255, 0, 0))
        executor.submit(draw_landmarks, points[54:], (0, 0, 255))
        executor.submit(draw_connections, points[54:], HAND_CONNECTIONS, (0, 0, 255))

    success, encoded_img = cv2.imencode('.png', img)  # PNG is often faster than JPEG
    if not success:
        raise RuntimeError("Failed to encode image")
    return encoded_img.tobytes()


import time
from concurrent.futures import ThreadPoolExecutor

def send_image_task(stub, data):
    try:
        image_bytes = visualize_landmarks_minimal(data)
        stub.PushImage(streaming_pb2.PushImageRequest(image=image_bytes))
    except Exception as e:
        logger.error(f"Error sending image: {e}")

def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    mem = None
    start_time = time.time()
    frame_count = 0
    batch_size = BATCH_SIZE
    batch = []

    while True:
        data = handle_concat_frame.pop()
        if data is None:
            mem = None
            continue
        if is_similar_frame(mem, data) and handle_concat_frame.getLen() > 100:
            continue

        try:
            image_bytes = visualize_landmarks_minimal(data)
            batch.append(image_bytes)
            frame_count += 1

            if len(batch) >= batch_size:
                stub.BatchPushImage(streaming_pb2.BatchPushImageRequest(images=batch))  # Send batch
                batch.clear()  # Clear batch after sending

            # Log every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                logger.info(f"Images sent in 1 second: {frame_count}")
                frame_count = 0
                start_time = time.time()
        except Exception as e:
            logger.error(f"Error sending image: {e}")

        mem = data


def is_similar_frame(frame1, frame2, threshold=0.1):
    return False if frame1 is None or frame2 is None else np.linalg.norm(frame1 - frame2) < threshold

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        send_image_thread = Thread(target=send_image_into_streaming, args=(stub, handle_concat_frame))
        send_image_thread.start()
        for response in pop_frame_response:
            if response.request_status == "Success":
                try:
                    frames = matrix_list_to_numpy(response.frame)
                    handle_concat_frame.push_into_process_queue(frames)
                except Exception as e:
                    logger.error(f"Error processing received frame: {e}")

if __name__ == "__main__":
    run()