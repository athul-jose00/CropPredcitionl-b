from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def yield_predict():
    if request.method == 'POST':
        state_names = request.form['state_names']
        district_names = request.form['district_names']
        crop_names = request.form['crop_names']
        area = float(request.form['area'])
        temperature = float(request.form['temperature'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        humidity = float(request.form['humidity'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        features = np.array([[state_names, district_names, crop_names, area, temperature, wind_speed, precipitation, humidity, N, P, K]])
        transformed_input_data = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_input_data).reshape(1, -1)
        return render_template('index.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
