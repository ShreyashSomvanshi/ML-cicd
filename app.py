from flask import Flask, request, jsonify
import joblib
import numpy as numpy

app = Flask(__name__)
model = joblib.load('rf_model.joblib')

@app.route('/', methods=['GET'])
def home():
	return "Hello World!"
	# return {'message':'Iris Model API'}


@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json(force=True)
	prediction = model.predict(np.array(data['features']).reshape(1,-1))
	return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
	app.run(debug=True)

