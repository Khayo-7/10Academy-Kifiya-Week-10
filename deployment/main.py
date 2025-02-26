from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
from lstm_model import LSTMPredictor

app = Flask(__name__)

# Load trained LSTM model
lstm_model = LSTMPredictor("brent_oil_prices_economic_indicators.csv")
lstm_model.prepare_data(seq_length=10)
lstm_model.load_trained_model("lstm_final_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict future Brent oil prices."""
    data = request.json
    steps = data.get('steps', 30)  # Default to 30-day prediction

    try:
        predictions = lstm_model.forecast(steps=steps)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
