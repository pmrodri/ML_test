from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
model = LinearRegression()

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    X = np.array(data['X'])
    y = np.array(data['y'])
    model.fit(X, y)
    return jsonify({"message": "Modelo treinado com sucesso!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['X'])
    predictions = model.predict(X)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)