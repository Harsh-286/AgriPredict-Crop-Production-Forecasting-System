from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'State_Name': request.form['state'],
        'District_Name': request.form['district'],
        'Crop_Year': int(request.form['year']),
        'Season': request.form['season'],
        'Crop': request.form['crop'],
        'Temperature': float(request.form['temperature']),
        'Humidity': float(request.form['humidity']),
        'Soil_Moisture': float(request.form['soil']),
        'Area': float(request.form['area'])
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    return render_template(
        'index.html',
        prediction_text=f"Predicted Production: {prediction:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)