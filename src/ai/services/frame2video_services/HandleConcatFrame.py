from collections import deque
import numpy as np
from loguru import logger
from src.ai.services.frame2video_services.lstm_model import load_model, predict

lstm_model = load_model("/Users/trHien/DoAnTotNghiep/ViSTAR/src/ai/services/frame2video_services/cut.pth")

def concatenate_frame(prev_frame, post_frame, rest):
    """
    Args:
    prev_frame: only frame (75,3) which is that last frame of a previous word
    post_frame: only frame (75,3) which is frist frame of post word
    """
    if np.linalg.norm(prev_frame - post_frame) <= 1:
        middle = np.linspace(prev_frame, post_frame, num=5)
    elif np.linalg.norm(prev_frame - post_frame) <= 2:
        middle = np.linspace(prev_frame, post_frame, num=7)
    else:
        middle = np.linspace(prev_frame, post_frame, num=10)
    
    logger.info(f"{middle.shape} - {post_frame.shape}")
    concatenated_frame = np.concatenate((middle, [post_frame], rest),axis=0)
    return concatenated_frame
  
class HandleConcatFrame:
    def __init__(self):
        """
        INIT
        """
        self.processed_frame_queue = deque()
        
    def push_into_process_queue(self, frames):
        """
        push frame into queue with process progress
        """
        try:
            frames = frames/1000
            p = predict(lstm_model, frames)
            frames = frames[p.flatten() == 1]
            if (self.processed_frame_queue[-1] is not None):
                
                prev_frame = self.processed_frame_queue[-1]
                post_frame = frames[0]
                
                # Concat 2 frame
                result = concatenate_frame(prev_frame=prev_frame, post_frame=post_frame, rest=frames[1:])
                
                result = result.tolist()
                
                self.processed_frame_queue.extend(result)
                
                # logger.success("Processed successfuly")
        except IndexError:
            self.processed_frame_queue.extend(frames)
        except Exception as e:
            print(e)
    
    def pop(self):
        try:
            a = np.array([self.processed_frame_queue.popleft()])
            logger.info(f"len: {len(self.processed_frame_queue)}")
            return a
        except IndexError:
            return None
        
    def getLen(self):
        return len(self.processed_frame_queue)