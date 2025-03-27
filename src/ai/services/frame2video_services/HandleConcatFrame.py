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
    prev_frame: only frame (75,3) which is that last frame of a previous word
    post_frame: only frame (75,3) which is frist frame of post word
    """
    prev_frame = np.array(prev_frame)
    post_frame = np.array(post_frame)
    prev_frame = prev_frame.reshape(75, 3)
    post_frame = post_frame.reshape(75, 3)
    num_frame_concat = 0
    if np.linalg.norm(prev_frame - post_frame) <= 1:
        num_frame_concat = 5
        middle = np.linspace(prev_frame, post_frame, num=5)
    elif np.linalg.norm(prev_frame - post_frame) <= 2:
        num_frame_concat = 7
        middle = np.linspace(prev_frame, post_frame, num=7)
    else:
        num_frame_concat = 10
        middle = np.linspace(prev_frame, post_frame, num=10)
    
    logger.info(f"{middle.shape} - {post_frame.shape}")
    if rest is None:
        concatenated_frame = np.concatenate((middle, [post_frame]),axis=0)
    else:
        concatenated_frame = np.concatenate((middle, [post_frame], rest),axis=0)
    
    return (concatenated_frame, num_frame_concat)


import json
default_frame = None
with open("character_dict.json", 'r') as f:
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
            
            # Remove the default tail if it exists.
            if len(self.processed_frame_queue) > 0 and np.allclose(np.array(self.processed_frame_queue[-1]), self.default_frame):
                self.remove_default_frame(self.num_default_frames)
            
            if len(self.processed_frame_queue) > 0:
                prev_frame = self.processed_frame_queue[-1]
                post_frame = frames[0]
                result, self.num_default_frames = concatenate_frame(prev_frame=prev_frame, post_frame=post_frame, rest=frames[1:])

                result = result.tolist()
                
                self.processed_frame_queue.extend(result)
            else:
                new_tail, self.num_default_frames = concatenate_frame(prev_frame=self.default_frame, post_frame=frames[0], rest=frames[1:])
                self.processed_frame_queue.extend(new_tail.tolist())
        
        except IndexError as e:
            logger.error("Queue is empty")
            self.processed_frame_queue.extend(frames.tolist())
        except Exception as e:
            logger.error(e)
        finally:
            if self.processed_frame_queue:
                last_frame = np.array(self.processed_frame_queue[-1])
                # logger.error(f"Last frame shape: {last_frame.shape}")
                new_tail, self.num_default_frames = concatenate_frame(prev_frame=last_frame, post_frame=self.default_frame, rest=None)
                # logger.error(f"New tail shape: {new_tail.shape}")
                new_tail = new_tail.tolist()
                self.processed_frame_queue.extend(new_tail)
    
    def pop(self):
        try:
            frame = np.array([self.processed_frame_queue.popleft()])
            # logger.info(f"Queue length: {len(self.processed_frame_queue)}")
            return frame
        except IndexError:
            return None
        
    def getLen(self):
        return len(self.processed_frame_queue)