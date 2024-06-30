from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
import category_encoders as ce
from custom_label_encoder import CustomLabelEncoder  # Import CustomLabelEncoder from custom_label_encoder.py

app = Flask(__name__)

print("Loading pipeline...")
pipeline = joblib.load('pipeline_1.pkl')
print("Pipeline loaded successfully.")

print("Loading unique values...")
with open('unique_values.pkl', 'rb') as file:
    unique_values = pickle.load(file)
print("Unique values loaded successfully.")

@app.route('/')
def index():
    print("Rendering index.html...")
    return render_template('index.html', unique_values=unique_values)

@app.route('/about-us')
def about_us():
    return render_template('about_us.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received. Parsing JSON data...")
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])  # Create DataFrame from JSON data
        print("Input data:")
        print(input_data)
        
        new_column_names = {
            'days_shipping_real': 'Days for shipping (real)',
            'days_shipment_scheduled': 'Days for shipment (scheduled)'
        }
        input_data.rename(columns=new_column_names, inplace=True)
        print("Renamed input data:")
        print(input_data)

        ans = pipeline.predict(input_data)
        print("Prediction:", ans)
        return jsonify({'prediction': int(ans[0])})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
