from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Define features
FEATURES = [
    'Age', 'Gender', 'Occupation', 'Days_Indoors', 'Growing_Stress',
    'Quarantine_Frustrations', 'Changes_Habits', 'Weight_Change',
    'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness'
]

# Define options
OPTIONS = {
    'Age': ['16-20', '20-25', '25-30', '30-Above'],
    'Gender': ['Male', 'Female'],
    'Occupation': ['Student', 'Corporate', 'Others'],
    'Days_Indoors': ['1-14 days', '15-31 days', '31-60 days', 'More than 2 months', 'Go out Every day'],
    'Growing_Stress': ['Yes', 'No'],
    'Quarantine_Frustrations': ['Yes', 'No'],
    'Changes_Habits': ['Yes', 'No'],
    'Weight_Change': ['Yes', 'No'],
    'Mood_Swings': ['Low', 'Medium', 'High'],
    'Coping_Struggles': ['Yes', 'No'],
    'Work_Interest': ['Yes', 'No'],
    'Social_Weakness': ['Yes', 'No']
}

# Load model and preprocessor with error handling
try:
    model = joblib.load("mental_health_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    model = None
    preprocessor = None
    print(f"[ERROR] Failed to load model or preprocessor: {e}")

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES, options=OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return "Model or preprocessor not loaded. Please check the server logs."

    try:
        # Get input data
        input_data = [request.form[feature] for feature in FEATURES]
        df = pd.DataFrame([input_data], columns=FEATURES)

        # Debug: print input
        print("[DEBUG] User input:", input_data)

        # Transform the input
        X_transformed = preprocessor.transform(df)

        # Debug: print transformed input
        print("[DEBUG] Transformed input:", X_transformed)

        # Predict
        raw_prediction = model.predict(X_transformed)[0]

        # If regressor, threshold to binary
        prediction_binary = 1 if raw_prediction >= 0.5 else 0

        print("[DEBUG] Raw prediction:", raw_prediction)
        print("[DEBUG] Final prediction:", prediction_binary)

        result = "Yes" if prediction_binary == 1 else "No"
        return render_template('index.html', features=FEATURES, options=OPTIONS,
                               prediction_text=f"Predicted Mental Health Status: {result}")
    except Exception as e:
        return f"[ERROR] Prediction failed: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
