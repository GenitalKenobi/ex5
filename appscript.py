from flask import Flask, jsonify, request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model.joblib')

dictionary = {0:'Iris Virginica', 1:'Iris Setosa', 2:'Iris Versicolor'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1,-1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': dictionary[prediction]})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
