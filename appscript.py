from flask import Flask, jsonify, request
from joblib import load

app = Flask(__name__)
model = load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
