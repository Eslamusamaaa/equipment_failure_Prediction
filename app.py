import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from features import add_derived

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'equipment_failure_Prediction', 'model_final.pkl')

# Bundle produced by train_final.py: ensemble of models, per-class weights,
# class names and the tool-wear alert threshold.
bundle = None
try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Model not found at {MODEL_PATH} ({e}). Run train_final.py first.")


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if bundle is None:
            return jsonify({'error': 'Model not loaded', 'status': 'error'})

        row = pd.DataFrame([{
            'Type': int(request.form['type']),
            'Air temperature [K]': float(request.form['air_temperature']),
            'Process temperature [K]': float(request.form['process_temperature']),
            'Rotational speed [rpm]': float(request.form['rotational_speed']),
            'Torque [Nm]': float(request.form['torque']),
            'Tool wear [min]': float(request.form['tool_wear']),
        }])
        X = add_derived(row)[bundle['features']]

        classes = bundle['classes']
        proba = np.mean([m.predict_proba(X)[0] for m in bundle['models'].values()], axis=0)
        pred = int(np.argmax(proba * bundle['weights']))

        # Tool-wear alert: model says "No Failure" but the tool is in the risky wear zone.
        alert = False
        if classes[pred] == 'No Failure' and X['wear'].iloc[0] >= bundle['wear_alert']:
            pred = classes.index('Tool Wear Failure')
            alert = True

        return jsonify({
            'prediction': classes[pred],
            # P(No Failure): the UI shows it as "stability"
            'probability': float(proba[classes.index('No Failure')]),
            'probabilities': {c: float(p) for c, p in zip(classes, proba)},
            'tool_wear_alert': alert,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})


if __name__ == '__main__':
    app.run(debug=True)
