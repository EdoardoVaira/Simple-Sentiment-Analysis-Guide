from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scalers
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('maxabs_scaler.pkl')
y_scaler = joblib.load('minmax_scaler.pkl')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.form['text']
    
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Vectorize and scale the input text
    X_new_bow = vectorizer.transform([data])
    X_new_scaled = scaler.transform(X_new_bow)
    
    # Predict the sentiment score
    y_pred = model.predict(X_new_scaled)
    
    # Inverse transform to get the predicted score in the original range
    y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Ensure correct shape for inverse_transform
    
    # Clip predictions to stay within the 1-5 range
    y_pred_original_clipped = np.clip(y_pred_original, 1, 5)
    
    # Return the predicted score directly
    return jsonify({'score': float(y_pred_original_clipped[0])})  # Convert to a standard Python float

if __name__ == '__main__':
    app.run(debug=True)
