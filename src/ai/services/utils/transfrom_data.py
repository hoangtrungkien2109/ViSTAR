import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import numpy as np
def numpy_to_matrix_list(array):
    matrix_list = streaming_pb2.MatrixList()
    
    print(f"Converting numpy array of shape {array.shape} to MatrixList...")

    for matrix in array:  # Loop over batch dimension (N)
        matrix_pb = matrix_list.matrix.add()  # Add a new matrix to the list
        for row in matrix:  # Loop over rows
            row_pb = matrix_pb.rows.add()  # Add a new row
            row_pb.elements.extend(row.tolist())  # Convert row to list and extend
    
    return matrix_list

def matrix_list_to_numpy(matrix_list):
    numpy_array = np.array([
        [[element for element in row.elements] for row in matrix.rows]
        for matrix in matrix_list.matrix
    ])

    print(f"Converted back to numpy array of shape {numpy_array.shape}")
    return numpy_array

import cv2

# Correct connections for pose landmarks in MediaPipe (total 33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # Right eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),          # Left eye to ear
    (9, 10), (11, 12),                       # Shoulders
    (11, 13), (13, 15), (15, 17),            # Left arm
    (12, 14), (14, 16), (16, 18),            # Right arm
    (23, 24),                                # Hips
    (24, 26), (26, 28), (28, 32),            # Right leg
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (5, 6), (6, 7), (7, 8),          # Index finger
    (9, 10), (10, 11), (11, 12),     # Middle finger
    (13, 14), (14, 15), (15, 16),    # Ring finger
    (17, 18), (18, 19), (19, 20)     # Pinky finger
]

def draw_landmarks(image, frame_landmarks, line_thickness=2):
    pose_landmarks = frame_landmarks[:33]
    for lm in pose_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
    # Draw lines between connected pose landmarks
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = pose_landmarks[start_idx]
        end_point = pose_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)

    right_hand_landmarks = frame_landmarks[33:33 + 21]
    right_hand_present = False
    for lm in right_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            right_hand_present = True
            
    # Draw lines between connected right hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = right_hand_landmarks[start_idx]
        end_point = right_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), line_thickness)
            
    left_hand_landmarks = frame_landmarks[33 + 21:]
    left_hand_present = False
    for lm in left_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            left_hand_present = True
            
    # Draw lines between connected left hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = left_hand_landmarks[start_idx]
        end_point = left_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), line_thickness)
             
    if not right_hand_present and not left_hand_present:
        return None  # Return None if no hands are detected