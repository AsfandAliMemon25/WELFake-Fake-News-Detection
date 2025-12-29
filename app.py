from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
dt_model = joblib.load('decision_tree_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return "Decision Tree & Logistic Regression Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' key"}), 400
    
    X_input = np.array(data['features']).reshape(1, -1)
    dt_pred = dt_model.predict(X_input).tolist()
    lr_pred = lr_model.predict(X_input).tolist()
    
    return jsonify({'DecisionTree': dt_pred[0], 'LogisticRegression': lr_pred[0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

