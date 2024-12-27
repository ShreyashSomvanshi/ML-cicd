from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np

app = Flask(__name__)
model = joblib.load('rf_model.joblib')

@app.route('/', methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	data = request.form
	features = [float(data['sepal_length']),
				float(data['sepal_width']),
				float(data['petal_length']),
				float(data['petal_width'])]
	prediction = model.predict(np.array(features).reshape(1,-1))
	return render_template('index.html', prediction = prediction[0])

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)
