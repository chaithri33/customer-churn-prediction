# 📊 Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using machine learning.

## 📂 Project Structure

- `data/`: Contains the Telco-Customer-Churn dataset.
- `churn_notebook.ipynb`: Jupyter notebook for EDA, preprocessing, model training, and saving the model.
- `app.py`: Streamlit app for interactive churn prediction.
- `churn_model.pkl` & `scaler.pkl`: Saved trained model and scaler.
- `requirements.txt`: Python dependencies to run the project.

## 🚀 How It Works

- **Input:** Customer details like tenure, contract type, charges, services used, etc.
- **Output:** Predicts whether the customer is likely to churn and shows the probability.

## 📊 Model Performance

- **Algorithm:** Random Forest Classifier
- **ROC AUC Score:** ~0.81
- **Accuracy:** ~79%

## ✅ How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer_churn_prediction.git
   cd customer_churn_prediction
