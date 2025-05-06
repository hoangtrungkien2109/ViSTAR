# import os
# import cv2
import mediapipe as mp
import numpy as np
# import math
from src.be.tool import *

# from gtts import gTTS
# from playsound import playsound
# from tensorflow.keras.models import load_model
# import os
# import time
import torch
cap = None
# model = load_model('model/30hope3.keras',safe_mode=False)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
LEFT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
RIGHT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
sequence = []
sentence = []
predictions = []
sign_buffer = []
confidence_history = []
EMA_option = True  # Set to False to use the old averaging method
alpha = 0.7  # Smoothing factor for EMA (0 < alpha < 1)
ema_predictions = None

threshold = 0.7
actions = np.array(
    ["Xin chao", "Tu Choi", "Le Halloween", "Ruc Ro", "May Man", "Nhan Vien", "Dia Chi", "San Truong", "Thay", "Toi",
     "Khong quen", "Nghi Hoc", "Tiep tan", "Ngay nay", "Cam on"
        , "Xin loi", "Ky nang", "Hap dan", "Thuong Xuyen"])
# Set mediapipe model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
import math


# Update Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings for the maximum sequence length
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor of shape (seq_len, batch_size, d_model)
        """
        # Dynamically select positional encodings based on the sequence length
        seq_len = x.size(0)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers,
                 dim_feedforward, num_actions, dropout, max_seq_length=40):
        super(TransformerClassifier, self).__init__()

        # 1D Convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=input_size,  # Number of input features (457)
            out_channels=d_model,  # Output channels (same as d_model for compatibility)
            kernel_size=3,  # Kernel size (adjust as needed)
            padding=1  # Padding to maintain sequence length
        )
        self.conv_residual = nn.Linear(input_size, d_model)
        self.conv_norm = nn.LayerNorm(d_model)
        # Linear layer to project input to d_model
        self.input_linear = nn.Linear(d_model, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_norm = nn.LayerNorm(d_model)
        # Classifier
        self.classifier = nn.Linear(d_model, num_actions)
        self.dropout = nn.Dropout(dropout)
        self.cache = None
        self.max_seq_length = max_seq_length

    def reset_cache(self):
        self.cache = None

    def forward(self, src, update_cache=True):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            out: Tensor of shape (batch_size, num_actions)
        """
        # Apply 1D Convolution
        # Input shape: (batch_size, seq_len, input_size)
        # Conv1d expects (batch_size, input_size, seq_len), so we transpose
        src_residual = self.conv_residual(src)
        src = src.transpose(1, 2)  # (batch_size, input_size, seq_len)
        src = self.conv1d(src)  # (batch_size, d_model, seq_len)
        src = src.transpose(1, 2)  # (batch_size, seq_len, d_model)
        src = self.conv_norm(src + src_residual)
        # Project to d_model
        src = self.input_linear(src)  # (batch_size, seq_len, d_model)

        # Transpose for Transformer: (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)

        # Update cache (if enabled)
        if self.cache is None:
            self.cache = src
        else:
            if update_cache:
                self.cache = torch.cat([self.cache, src], dim=0)
                # Truncate cache if it exceeds max_seq_length
                if self.cache.size(0) > self.max_seq_length:
                    self.cache = self.cache[-self.max_seq_length:, :, :]

        # Add positional encoding
        src = self.positional_encoding(src)

        # Pass through Transformer encoder
        memory = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        memory = self.transformer_norm(memory + src)
        # Aggregate the outputs (e.g., take the mean over the sequence)
        memory = memory.mean(dim=0)  # (batch_size, d_model)
        memory = self.dropout(memory)

        # Classifier
        out = self.classifier(memory)  # (batch_size, num_actions)
        return out


batch_size = 32
seq_len = 40
input_size = 457
# Initialize the model
num_actions = actions.shape[0]  # Replace with your actual number of actions
model = TransformerClassifier(num_actions=num_actions, input_size=input_size, d_model=256, nhead=4,
                              num_encoder_layers=4, dim_feedforward=1024, dropout=0.2, max_seq_length=seq_len)

# Load the trained weights
model.load_state_dict(torch.load('src/be/n2_dict.pth', map_location=torch.device('cpu')))  # Use 'cuda' if on GPU

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Set the model to evaluation mode
model.eval()
# def predict_stt():
#     global sequence,sentence,ema_predictions,cap
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#
#             ret, frame = cap.read()
#
#             if not ret:
#                 break
#
#             image, results = mediapipe_detection(frame, holistic)
#             draw_styled_landmarks(image, results)
#
#             left_hand_orientation_encoded = [0, 0, 0, 0, 0]
#             right_hand_orientation_encoded = [0, 0, 0, 0, 0]
#             calculate_left_rotation_encoded = [0] * 8
#             calculate_right_rotation_encoded = [0] * 8
#             determine_lhand_shape_encoded = [0] * 7
#             determine_rhand_shape_encoded = [0] * 7
#             finger_lencoded = [0] * 40
#             finger_rencoded = [0] * 40
#             # Determine hand orientation (if detected)
#             if results.pose_landmarks or (results.left_hand_landmarks or results.right_hand_landmarks):
#                 pose_landmarks = np.array(
#                     [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
#                 left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in
#                                                 results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None
#                 right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in
#                                                  results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else None
#
#                 # Normalize holistic landmarks
#                 pose_landmarks, left_hand_landmarks, right_hand_landmarks = PoseNormalizer.normalize_holistic(
#                     pose_landmarks, left_hand_landmarks, right_hand_landmarks
#                 )
#                 keypoints = extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks)
#                 if left_hand_landmarks is not None:
#                     normal_vector_left = calculate_palm_normal(left_hand_landmarks, 'Left')
#                     left_hand_orientation_encoded = classify_hand_view(normal_vector_left, 'Left')
#                     calculate_left_rotation_encoded = calculate_hand_rotation(left_hand_landmarks)
#                     determine_lhand_shape_encoded = determine_hand_shape(left_hand_landmarks)
#                     finger_lencoded = one_hot_finger(left_hand_landmarks)
#                 if right_hand_landmarks is not None:
#                     normal_vector_right = calculate_palm_normal(right_hand_landmarks, 'Right')
#                     right_hand_orientation_encoded = classify_hand_view(normal_vector_right, 'Right')
#                     calculate_right_rotation_encoded = calculate_hand_rotation(right_hand_landmarks)
#                     determine_rhand_shape_encoded = determine_hand_shape(right_hand_landmarks)
#                     finger_rencoded = one_hot_finger(right_hand_landmarks)
#                 keypoints = np.concatenate([keypoints, left_hand_orientation_encoded, right_hand_orientation_encoded,
#                                             calculate_left_rotation_encoded, calculate_right_rotation_encoded,
#                                             determine_lhand_shape_encoded, determine_rhand_shape_encoded, finger_lencoded,
#                                             finger_rencoded])
#
#                 sequence.append(keypoints)
#                 sequence = sequence[-40:]
#                 if len(sequence) == 40:
#                     input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 30, 369)
#                     input_tensor = input_tensor.to(device)
#                     with torch.no_grad():
#                         output = model(input_tensor, update_cache=False)
#                         res = torch.softmax(output, dim=1).cpu().numpy()[0]
#
#                     # Dynamic threshold adjustment
#                     if EMA_option:
#                         if ema_predictions is None:
#                             ema_predictions = res
#                         else:
#                             ema_predictions = alpha * res + (1 - alpha) * ema_predictions
#                         avg_predictions = ema_predictions
#                     else:
#                         predictions.append(res)
#                         predictions = predictions[-10:]  # Keep the last 10 predictions
#                         avg_predictions = np.mean(predictions, axis=0)
#                     print(np.max(avg_predictions))
#                     print(np.argmax(avg_predictions))
#                     avg_pred_class = np.argmax(avg_predictions)
#
#                     if avg_predictions[avg_pred_class] > threshold:
#                         if actions[avg_pred_class] != sentence:
#                             sentence = actions[avg_pred_class]
#                             print(f"New prediction: {sentence}")
#
#                         # Reset for the next prediction
#                         sequence = []
#                         if EMA_option:
#                             ema_predictions = None
#                         else:
#                             predictions = []
#
#                 y_offset = 100
#                 cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#                 cv2.putText(image, ' '.join(sentence), (3, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#
#                 # Encode the frame as JPEG
#                 ret, buffer = cv2.imencode('.jpg', image)
#                 if not ret:
#                     continue  # If encoding fails, skip this frame
#
#                 # Build a 'multipart/x-mixed-replace' response
#                 frame_bytes = buffer.tobytes()
#                 yield (
#                         b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
#                 )

