import os
from dotenv import load_dotenv
from collections import deque
import numpy as np
from loguru import logger
from src.ai.services.frame2video_services.lstm_model import load_model, predict
load_dotenv()

lstm_model = load_model(os.getenv("CUTTING_MODEL_PATH"))


def concatenate_frame(prev_frame, post_frame, rest):
    """
    Args:
    prev_frame: The last frame of a previous word, shape (75, 3).
    post_frame: The first frame of the next word, shape (75, 3).
    rest: The remaining frames of the next word.
    """
    prev_frame = np.array(prev_frame).reshape(75, 3)
    post_frame = np.array(post_frame).reshape(75, 3)


    # Create copies
    prev_frame_processed = np.copy(prev_frame)
    post_frame_processed = np.copy(post_frame)

    # Identify landmarks with (0,0) coordinates, which we treat as missing.
    # We check only x and y, as z can sometimes be non-zero.
    prev_missing_mask = np.all(prev_frame[:, :2] == 0, axis=1)
    post_missing_mask = np.all(post_frame[:, :2] == 0, axis=1)

    # Make it "hold" its last position during interpolation.
    disappearing_landmarks = ~prev_missing_mask & post_missing_mask
    post_frame_processed[disappearing_landmarks] = prev_frame[disappearing_landmarks]

    # Make it start at its final destination.
    appearing_landmarks = prev_missing_mask & ~post_missing_mask
    prev_frame_processed[appearing_landmarks] = post_frame[appearing_landmarks]


    distance = np.linalg.norm(prev_frame_processed - post_frame_processed)

    if distance <= 1:
        num_frame_concat = 5
    elif distance <= 2:
        num_frame_concat = 7
    else:
        num_frame_concat = 10

    middle = np.linspace(prev_frame_processed, post_frame_processed, num=num_frame_concat)

    logger.info(f"{middle.shape} - {post_frame.shape}")

    if rest is None:
        concatenated_frame = np.concatenate((middle, [post_frame]), axis=0)
    else:
        concatenated_frame = np.concatenate((middle, [post_frame], rest), axis=0)

    return (concatenated_frame, num_frame_concat)


import json
default_frame = None
with open(r"/home/chucky/ViSTAR/data/character_dict.json", 'r') as f:
        default_frame = np.array(json.load(f)["default"])
        default_frame = default_frame.reshape(1, 75, 3)/1000.0

class HandleConcatFrame:
    def __init__(self, default_frame=default_frame):
        self.processed_frame_queue = deque()
        self.default_frame = default_frame

        self.processed_frame_queue.extend(default_frame)
        self.num_default_frames = 1

    def remove_default_frame(self, num_frame=1):
        if len(self.processed_frame_queue) <= (num_frame + 2): # Delay frames preving sending task
            return

        for _ in range(num_frame):
            try:
                self.processed_frame_queue.pop()
            except IndexError:
                logger.error("Default frames is showing")
                break

    def push_into_process_queue(self, frames):
        try:
            if len(frames) != 1:
                frames = frames / 1000.0
                p = predict(lstm_model, frames)
                frames = frames[p.flatten() == 1]
                if len(frames) == 0:  # Early exit if no frames survive filtering
                    logger.warning("No frames left after prediction filtering.")
                    # We still add the default tail in the finally block
                    return

            # Check if the last item in the deque is a numpy array and then compare
            if self.processed_frame_queue:
                last_item = self.processed_frame_queue[-1]
                # Ensure it's a numpy array before calling np.array_equal
                if isinstance(last_item, np.ndarray) and np.array_equal(last_item, self.default_frame):
                    self.remove_default_frame(self.num_default_frames)

            if self.processed_frame_queue:
                prev_frame = self.processed_frame_queue[-1]
                post_frame = frames[0]
                # concatenate_frame returns numpy arrays
                result, self.num_default_frames = concatenate_frame(
                    prev_frame=prev_frame, post_frame=post_frame, rest=frames[1:]
                )
                # Extend the deque with the numpy arrays directly
                self.processed_frame_queue.extend(result)
            else:
                # Handle the case where the queue was empty
                self.processed_frame_queue.extend(frames)

        except IndexError as e:
            logger.error("IndexError: Likely no frames passed the filter. Extending with raw frames.")
            self.processed_frame_queue.extend(frames)  # Add frames directly if processing fails
        except Exception as e:
            logger.error(e)
        finally:
            if self.processed_frame_queue:
                last_frame = self.processed_frame_queue[-1]
                # concatenate_frame returns a numpy array
                new_tail, self.num_default_frames = concatenate_frame(
                    prev_frame=last_frame, post_frame=self.default_frame.squeeze(), rest=None
                )
                # Extend with the new numpy arrays
                self.processed_frame_queue.extend(new_tail)

    def pop(self):
        try:
            # The item is already a numpy array, just needs reshaping
            frame = self.processed_frame_queue.popleft()
            return frame.reshape(1, 75, 3)
        except IndexError:
            return None

    def getLen(self):
        return len(self.processed_frame_queue)