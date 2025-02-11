import streamlit as st
import streamlit.components.v1 as components
import grpc
import io
import base64
import datetime
from PIL import Image
import streaming_pb2 as streaming_pb2
import streaming_pb2_grpc as streaming_pb2_grpc

def create_grpc_channel():
    # Replace with your gRPC server address
    channel = grpc.insecure_channel('localhost:50051')
    return channel

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

def push_text(stub, text):
    try:
        timestamp = get_timestamp()
        request = streaming_pb2.PushTextRequest(text=text, time_stamp=timestamp)
        response = stub.PushText(request)
        return response.request_status, timestamp
    except grpc.RpcError as e:
        return f"Error: {str(e)}", None

def stream_images(stub):
    try:
        # Assuming your gRPC method doesn't require any request data
        response_stream = stub.PopImage(streaming_pb2.PopImageRequest(time_stamp=""))  # Adjust request as needed
        for response in response_stream:
            yield response.text  # Adjust based on your proto definition
    except grpc.RpcError as e:
        st.error(f"Error streaming images: {e}")

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Failed to decode image: {e}")
        return None
    

def render_speech_to_text():
    speech_to_text_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Speech to Text</title>
    </head>
    <body>
        <button id="start-btn" style="padding: 10px 20px; font-size: 16px;">🎤 Start Listening</button>
        <p id="status" style="margin-top: 20px;">Press the button and speak...</p>
        <textarea id="transcript" rows="4" style="width: 100%; margin-top: 20px;"></textarea>
        <script>
            const startBtn = document.getElementById("start-btn");
            const status = document.getElementById("status");
            const transcriptTextarea = document.getElementById("transcript");

            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();

                recognition.lang = 'vi-VN'; // Đặt ngôn ngữ nhận diện là tiếng Việt
                recognition.continuous = false;
                recognition.interimResults = true;

                startBtn.addEventListener('click', () => {
                    recognition.start();
                });

                recognition.onstart = () => {
                    status.innerText = "Listening...";
                    startBtn.disabled = true;
                };

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    transcriptTextarea.value = transcript; // Hiển thị kết quả trong textarea
                    status.innerText = "Speech captured!";
                    // Truyền giá trị qua Streamlit (gửi transcript về Python)
                    window.parent.postMessage({ transcript: transcript }, "*");
                };

                recognition.onerror = (event) => {
                    status.innerText = `Error: ${event.error}`;
                    startBtn.disabled = false;
                };

                recognition.onend = () => {
                    status.innerText = "Press the button and speak...";
                    startBtn.disabled = false;
                };
            } else {
                status.innerText = "Your browser does not support Web Speech API.";
                startBtn.disabled = true;
            }
        </script>
    </body>
    </html>
    """
    return speech_to_text_code


def main():
    st.title("Streaming Text Service")
    
    # Initialize session state for storing message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Create gRPC channel and stub
    try:
        channel = create_grpc_channel()
        stub = streaming_pb2_grpc.StreamingStub(channel)
        st.sidebar.success("Connected to gRPC server")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to gRPC server: {str(e)}")
        return

    # Create tabs for Push and Pop operations
    tab1, tab2 = st.tabs(["Push Text", "Pop Text"])
    
    # Push Text Tab
    with tab1:
        st.header("Push Text")
        components.html(render_speech_to_text(), height=400)
        
        text_input = st.text_area("Enter your text:", height=100)
        
        if st.button("Send Text"):
            if text_input and text_input.strip():
                status, timestamp = push_text(stub, text_input)
                
                if "Error" not in status:
                    st.success(f"Text pushed successfully! Status: {status}")
                    # Store message in history
                    st.session_state.messages.append({
                        'text': text_input,
                        'timestamp': timestamp,
                        'status': status
                    })
                else:
                    st.error(status)
            else:
                st.warning("Please enter some text before sending.")
        st.subheader("Streamed Images")
        start_streaming = st.button("Start Image Stream")
        if start_streaming:
            # Placeholder for dynamic image updates
            
            # Stream images from the server
            for base64_image in stream_images(stub):
                image_placeholder = st.empty()  # This will allow us to update the image dynamically
                if base64_image:
                    # Decode and display the image
                    image = decode_base64_image(base64_image)
                    if image:
                        # Replace the current image in the placeholder
                        image_placeholder.image(image, caption="Streaming Image", use_container_width=True)
                # else:
                #     # Clear the image when no data is received
                #     image_placeholder.empty()
                    # st.warning("No image data received from the stream.")

if __name__ == "__main__":
    main()