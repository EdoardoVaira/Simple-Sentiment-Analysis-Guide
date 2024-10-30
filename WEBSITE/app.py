from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import torch

app = Flask(__name__)

# Load the model and scalers
model = torch.load('model.pth', map_location=torch.device('cpu'))
scaler = joblib.load('minmax_scaler.pkl')  # Load only the minmax scaler

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form['text']
    
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Directly transform the input text if it no longer needs vectorization
    # Ensure the data is in a format that `scaler` can process
    # For example, if `data` needs to be numeric, pre-process it accordingly

    # Scale the input text if needed for your specific input requirements
    X_new_scaled = scaler.transform([[float(data)]])  # Example assuming `data` is numeric
    
    # Convert to torch tensor for model input
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
    
    # Predict the sentiment score
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_pred = model(X_new_tensor)
    
    # Inverse transform to get the predicted score in the original range
    y_pred_original = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
    
    # Clip predictions to stay within the 1-5 range
    y_pred_original_clipped = np.clip(y_pred_original, 1, 5)
    
    # Return the predicted score directly
    return jsonify({'score': float(y_pred_original_clipped[0])})

if __name__ == '__main__':
    app.run(debug=True)
