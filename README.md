# 📦 Predicting Late Delivery Risk in Supply Chain Management

This project focuses on building a predictive model to identify the **risk of late deliveries** in a supply chain. Late deliveries can severely impact customer satisfaction, production schedules, and overall business efficiency. Our goal is to enable supply chain managers to take **proactive measures** to mitigate risks using data-driven insights.

## 🎯 Problem Statement

In supply chain management, predicting late deliveries is crucial to avoid disruptions, manage costs, and maintain customer trust. This project develops a **machine learning-based solution** that flags high-risk orders based on historical data.

---

## 🔍 Project Objectives

- Identify key features that contribute to late deliveries.
- Build and evaluate multiple ML models for accuracy and reliability.
- Develop a user-friendly web interface to make predictions in real-time.
- Deploy the model using a scalable web platform.

---

## 🧾 Dataset Description

The dataset contains **52 attributes** and over **180,000 records** about supply chain transactions, including:

- Order details (date, region, country, item info)
- Customer details (location, segment)
- Shipping details (mode, actual vs scheduled dates)
- Product details (name, price, category)
- Target variable: `Late_delivery_risk` (0 = On time, 1 = At risk)

---

## 🛠️ Data Preprocessing

- Dropped irrelevant or duplicate columns.
- Handled missing values (e.g., ZIP code imputed; product description dropped).
- Removed outliers using IQR method.
- Encoded categorical features using **Label Encoding** and **Target Encoding**.
- Scaled numerical features using **MinMaxScaler**.
- Final cleaned dataset: `cleaned4_late.csv`.

---

## 📊 Exploratory Data Analysis (EDA)

- Balanced target variable (approx. 55% late, 45% on-time).
- Most orders placed in LATAM and Europe.
- 51% of customers are Consumers; 30% Corporates.
- Majority of orders were **delivered late (54.8%)**.

> Insights from EDA were used to guide feature selection and model training.

---

## 🤖 Model Building

We trained and compared several machine learning models:

| Model                | AUC Score |
|---------------------|-----------|
| Logistic Regression | 0.65      |
| Naive Bayes         | 0.97      |
| XGBoost             | 1.00      |
| AdaBoost            | 1.00      |
| Decision Tree       | 1.00      |
| KNN                 | 0.56      |

- **Decision Tree** was selected as the final model based on high performance and interpretability.
- Hyperparameter tuning and cross-validation were applied for optimization.
- Final test accuracy: **99.5%**

---

## 🌐 Web Application

A Flask-based web interface allows users to:

- Input order details via form or API
- View delivery risk predictions
- Learn about the model and contributors

### 🔗 Live Demo

👉 [Visit Web App](https://delivery-risk-prediction.onrender.com)

---

## 🧩 Tech Stack

- **Frontend**: HTML, CSS (Bootstrap)
- **Backend**: Python (Flask)
- **ML Models**: Scikit-learn, XGBoost, Naive Bayes
- **Deployment**: Render.com

---


## ✅ Conclusion

This project successfully demonstrates how data analytics and machine learning can help anticipate late deliveries and improve supply chain performance. With real-time predictions and a user-friendly web interface, this tool provides actionable insights to optimize logistics.

---

## 📁 Files in This Repo

- `app.py` – Flask application logic
- `model.py` – Model training pipeline
- `custom_label_encoder.py` – Custom transformer for label encoding
- `pipeline_1.pkl` – Trained model pipeline
- `unique_values.pkl` – Encoded values for dropdowns
- `cleaned4_late.csv` – Final cleaned dataset
- `/templates` – HTML files for the frontend

---

## 📬 Contact

For any queries or collaborations, feel free to reach out!

---

