import asyncio
import base64
import grpc.aio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from streaming_pb2 import PushTextRequest, PopImageRequest, BatchPopImageResponse
from streaming_pb2_grpc import StreamingStub
import time
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def get_grpc_stub():
    """Initialize and return a gRPC stub."""
    channel = grpc.aio.insecure_channel('localhost:50051')
    return StreamingStub(channel), channel

async def handle_text(websocket: WebSocket, stub: StreamingStub):
    """Handle text messages sent by client."""
    try:
        while True:
            text = await websocket.receive_text()
            await stub.PushText(PushTextRequest(text=text, time_stamp=""))
    except WebSocketDisconnect:
        print("Client disconnected from text handling.")
    except Exception as e:
        print(f"Text handling error: {e}")

async def handle_images(websocket: WebSocket, stub: StreamingStub, queue: asyncio.Queue):
    """Process image stream from gRPC and send via WebSocket."""
    try:
        async for response in stub.BatchPopImage(PopImageRequest(time_stamp="")):
            if len(response.images) == 0:
                continue
            for image in response.images:
                base64_image = base64.b64encode(image).decode('utf-8')
                await queue.put(f"data:image/jpeg;base64,{base64_image}")
    except WebSocketDisconnect:
        print("Client disconnected from image handling.")
    except Exception as e:
        print(f"Image handling error: {e}")


async def send_images(websocket: WebSocket, queue: asyncio.Queue):
    """Send images from queue to WebSocket in a controlled manner (30 FPS)."""
    try:
        while True:
            image_data = await queue.get()
            await websocket.send_text(image_data)
            await asyncio.sleep(1 / 60)  # Maintain ~30 FPS
    except WebSocketDisconnect:
        print("Client disconnected from send_images.")
    except Exception as e:
        print(f"Error in send_images: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time text and image streaming."""
    await websocket.accept()
    stub, channel = await get_grpc_stub()
    queue = asyncio.Queue(maxsize=100)

    text_task = asyncio.create_task(handle_text(websocket, stub))
    image_task = asyncio.create_task(handle_images(websocket, stub, queue))
    send_task = asyncio.create_task(send_images(websocket, queue))

    try:
        done, pending = await asyncio.wait(
            [text_task, image_task, send_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        await channel.close()

@app.get("/")
async def get_index():
    """Serve the HTML page."""
    with open("./index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
