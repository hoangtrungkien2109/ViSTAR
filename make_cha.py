import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# with open("character_dict_old2.json", "r") as f1:
#     d = json.load(f1)
    
#     d["default"] = [d["a"][20]]
    
#     with open("character_dict.json", 'w') as f2:
#         json.dump(d, f2)
    
    
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

    return img




if __name__ == "__main__":
    
    default = np.load("/Users/trHien/frames/landmarks_D0001B.npy")[37]
    
    default = default.reshape(1,75,3)
    
    default = default * 1000
    default = default.astype(np.int16)

    with open("character_dict.json", 'r') as f1:
        d = json.load(f1)
        d["default"] = default.tolist()
        with open("character_dict.json", 'w') as f2:
            json.dump(d, f2)

    img = visualize_landmarks_minimal(default/1000.0)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    