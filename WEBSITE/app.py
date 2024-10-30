from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data handling
import numpy as np
import pandas as pd
import math

import multiprocessing
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the necessary classes from your main model file if applicable
# from your_model_file import Model, Embedding

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import emoji  
import re  
import contractions  

def text_cleaner(text):
    
    # Convert text to lowercase
    text = text.lower()

    # Replace emojis with a text description to capture their sentiment
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Expand contractions
    text = contractions.fix(text)

    # Replace non-word characters with a space to separate words better
    text = re.sub(r'[^\w\s]', ' ', text)

    # Reduce characters repeated more than twice to two to reduce exaggeration
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove consecutive repeated words
    text = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text)

    # Normalize white spaces
    text = re.sub(r'\s+', ' ', text).strip()  

    return text

# Load the tokenizer
tokenizer = joblib.load('tokenizer.pkl')  # Load the tokenizer

# Define or import the model architecture
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model=50,dropout=0.1,max_len=500):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pe = self.pos_encoding(max_len, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(pe, freeze=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        pos_mat = torch.arange(X.size(dim=-1)).to(device)
        X = self.input_embedding(X) * math.sqrt(self.d_model)
        X += self.pos_embedding(pos_mat)
        return self.dropout(X)

    def pos_encoding(self, max_len, d_model):
        dividend = torch.arange(max_len).unsqueeze(0).T
        divisor = torch.pow(10000.0, torch.arange(0, d_model, 2)/d_model)
        epsilon = 1e-8
        angles = dividend / (divisor+epsilon)
        pe = torch.zeros((max_len, d_model))
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

class Model(nn.Module):
    def __init__(self, embeddings, nhead, num_layers, nbr_classes, d_model=50, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embeddings = embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward, activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, nbr_classes)

    def forward(self, X):
        padding_mask = (X == 0)
        X = self.embeddings(X)
        X = self.encoder(X, src_key_padding_mask=padding_mask)
        X = torch.mean(X, dim=1)
        return self.fc(X)

# Load the model
vocab_size = 39783  # Set this according to your tokenizer's vocab size
embedding = Embedding(vocab_size).to('cpu')  # Adjust device as necessary
model = Model(embedding, nhead=5, num_layers=5, nbr_classes=1)  # Instantiate the model
model.load_state_dict(torch.load('model.pth', map_location='cpu'))  # Load model weights
model.eval()  # Set model to evaluation mode

# Load the scaler
scaler = joblib.load('minmax_scaler.pkl')  # Load the minmax scaler

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form['text']
    
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    data = text_cleaner(data) # Pre-process of the data

    # Convert the input text to sequences
    sequences = tokenizer.texts_to_sequences([data])  # Wrap in a list to maintain 2D structure
    max_sequence_length = 140  # Match with your model's expected input length

    # Pad sequences to ensure consistent input size
    X_new_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Convert to torch tensor for model input
    X_new_tensor = torch.tensor(X_new_padded, dtype=torch.long)  # Ensure type matches model input

    # Predict the sentiment score
    with torch.no_grad():
        y_pred = model(X_new_tensor)

    # Convert predictions to numpy and clip if necessary
    y_pred_numpy = y_pred.numpy().flatten()  # Convert to a 1D array

    # Clip predictions to stay within the 1-5 range if your model output needs it
    y_pred_clipped = np.clip(y_pred_numpy, 1, 5)

    # Return the predicted score directly
    return jsonify({'score': float(y_pred_clipped[0])})  # Convert to a standard Python float

if __name__ == '__main__':
    app.run(debug=True)
