from flask import Flask, request, jsonify, render_template
import joblib
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
	app.run(debug=True)

# rnd_czX5LhGAjoa0xrRGSmQ8SwCcssQi