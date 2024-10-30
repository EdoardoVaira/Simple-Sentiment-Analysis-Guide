from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model, scaler, and tokenizer
model = torch.load('model.pth', map_location=torch.device('cpu'))
scaler = joblib.load('minmax_scaler.pkl')  # Load the minmax scaler
tokenizer = joblib.load('tokenizer.pkl')  # Load the tokenizer

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form['text']
    
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Convert the input text to sequences
    sequences = tokenizer.texts_to_sequences([data])  # Wrap in a list to maintain 2D structure
    max_sequence_length = 100  # Adjust this to your model's expected input length

    # Pad sequences to ensure consistent input size
    X_new_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Convert to torch tensor for model input
    X_new_tensor = torch.tensor(X_new_padded, dtype=torch.float32)

    # Predict the sentiment score
    model.eval()  # Set model to evaluation mode
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
