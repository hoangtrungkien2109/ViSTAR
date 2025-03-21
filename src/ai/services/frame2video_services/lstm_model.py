import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

def pad_tensor(tensor, target_length=300):
    current_length = tensor.size(1)  # Get the current length of the second dimension
    padding_size = target_length - current_length
    
    if padding_size > 0:
        # Apply padding if the current length is less than the target length
        return F.pad(tensor, (0, 0, 0, 0, 0, padding_size))
    else:
        # If the tensor is already the correct size or larger, return it as is
        return tensor

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.bn = nn.BatchNorm1d(300)  # BatchNorm1d for sequence length

    def forward(self, x):
        batch_size, seq_len, _, _ = x.size()
        x = x.view(batch_size, seq_len, -1)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        out = out.squeeze(-1)  # Remove last dimension
        out = self.bn(out)  # Apply batch normalization
        return out

def load_model(filename):
    # Hyperparameters (should match the ones used during training)
    input_size = 225
    hidden_size = 128
    num_layers = 1
    output_size = 1
    # Initialize the model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filename}")
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, x):
    x = torch.FloatTensor(x).unsqueeze(0).to(device)  # Add batch dimension and move to device
    l = x.shape[1]
    x = pad_tensor(x)
    
    # Make prediction
    with torch.no_grad():
        output = model(x)
        prediction = torch.sigmoid(output)
        prediction = prediction.cpu().numpy()  # Move to CPU before converting to NumPy
        prediction = np.where(prediction > 0.8, 1, 0)
    prediction = prediction.reshape(300, 1)
    return prediction[:l, :]

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model('../cut/model_final.pth')
    model.eval()  # Set the model to evaluation mode

    # Prepare your input
    x = np.load("./train_landmarks/landmarks_D0125.npy")

    x = torch.FloatTensor(x).unsqueeze(0).to(device)  # Add batch dimension and move to device
    x = pad_tensor(x)
    print(x.shape)


    # Make prediction
    with torch.no_grad():
        output = model(x)
        prediction = torch.sigmoid(output)
        prediction = np.array(prediction)
        prediction = np.where(prediction > 0.5, 1, 0)

    print(f"Prediction: {prediction}")