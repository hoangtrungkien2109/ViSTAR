import threading
from queue import Queue
from loguru import logger
import os
import time

from dotenv import load_dotenv
from grpc import StatusCode
from grpc_interceptor.exceptions import GrpcException

import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import time

load_dotenv(override=True)
DELAY_TIME = float(os.getenv("DELAY_TIME"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
class StreamingBaseService(streaming_pb2_grpc.StreamingServicer):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(StreamingBaseService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True

            self.text_queue = Queue()
            self.frame_queue = Queue()
            self.image_queue = Queue()

    def PushText(self, request, context):
        try:
            self.text_queue.put(request.text)
            return streaming_pb2.PushTextResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopText(self, request, context):
        while True:
            if self.text_queue.empty():
                yield streaming_pb2.PopTextResponse(request_status="Empty", text=None)
            else:
                yield streaming_pb2.PopTextResponse(request_status="Success", text=self.text_queue.get())
            time.sleep(DELAY_TIME)

    def PushFrame(self, request, context):
        try:
            self.frame_queue.put(request.frame)
            return streaming_pb2.PushFrameResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopFrame(self, request, context):
        while True:
            if not self.frame_queue.empty():
                yield streaming_pb2.PopFrameResponse(request_status="Success", frame=self.frame_queue.get())

    def PushImage(self, request, context):
        try:
            for i in range(BATCH_SIZE):
                self.image_queue.put(request.image[i])
            return streaming_pb2.BatchPushImageResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopImage(self, request, context):
        frame_count = 0
        start_time = time.time()
        while True:
            if self.image_queue.empty():
                yield streaming_pb2.BatchPopImageResponse(request_status="Empty")
            else:
                images = []
                for _ in range(BATCH_SIZE):
                    images.append(self.image_queue.get())

                yield streaming_pb2.BatchPopImageResponse(request_status="Success", images=images)
                
    def BatchPushImage(self, request, context):
        try:
            for img in request.images:  # Process batch images
                self.image_queue.put(img)
            return streaming_pb2.BatchPushImageResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def BatchPopImage(self, request, context):
        while True:
            if self.image_queue.empty():
                yield streaming_pb2.BatchPopImageResponse(request_status="Empty", images=[])
            else:
                images = []
                for _ in range(min(BATCH_SIZE, self.image_queue.qsize())):
                    img = self.image_queue.get()
                    if not isinstance(img, bytes):
                        logger.error(f"Invalid data type in queue: {type(img)} (expected bytes)")
                        continue  
                    images.append(img)

                if images:  # Only send response if images exist
                    yield streaming_pb2.BatchPopImageResponse(request_status="Success", images=images)


