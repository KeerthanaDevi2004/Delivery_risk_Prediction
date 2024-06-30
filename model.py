import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
import category_encoders as ce
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from custom_label_encoder import CustomLabelEncoder  # Import CustomLabelEncoder from custom_label_encoder.py

def main():
    data = pd.read_csv("C:Users/ALMIGHTY/Downloads/late/cleaned4_late.csv")
    y = data['Late_delivery_risk']
    X = data.drop(columns=['Late_delivery_risk'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline with CustomLabelEncoder
    pipeline = Pipeline([
        ('preprocessing', ColumnTransformer([
            ('label_encoding', CustomLabelEncoder(['Type', 'Delivery Status', 'Order Status', 'Shipping Mode']), ['Type', 'Delivery Status', 'Order Status', 'Shipping Mode']),
            ('target_encoding', ce.TargetEncoder(cols=['Category Name', 'Customer City', 'Customer State', 'Market', 'Order City', 'Order Country', 'Order Region', 'Order State', 'Product Name']), ['Category Name', 'Customer City', 'Customer State', 'Market', 'Order City', 'Order Country', 'Order Region', 'Order State', 'Product Name']),
            ('scaling', MinMaxScaler(), ['Days for shipping (real)', 'Days for shipment (scheduled)'])
        ])),
        ('feature_selection', SelectKBest(score_func=chi2, k=15)),
        ('classification', DecisionTreeClassifier(random_state=42))
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Save the pipeline
    joblib.dump(pipeline, 'pipeline_1.pkl')

    # Save unique values
    unique_values = {}
    categorical_columns = ['Type', 'Delivery Status', 'Category Name', 'Customer City', 'Customer State',
                           'Market', 'Order City', 'Order Country', 'Order Region', 'Order State',
                           'Order Status', 'Product Name', 'Shipping Mode']
    for col in categorical_columns:
        unique_values[col] = X_train[col].unique().tolist()
    with open('unique_values.pkl', 'wb') as file:
        pickle.dump(unique_values, file)

if __name__ == '__main__':
    main()
