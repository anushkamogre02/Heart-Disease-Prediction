Heart Disease Prediction System

End-to-end Flask + ML + MySQL project to predict heart disease risk.

SETUP

1. Create environment & install dependencies
   python -m venv venv
   venv\Scripts\activate
   pip install -r backend/requirements.txt

2. Place dataset
   Download Kaggle heart disease dataset and save as:
   dataset/heart.csv

3. Train model
   Open model_training.ipynb in Jupyter and run all cells.
   This will generate:
   backend/model/heart_model.pkl
   backend/model/scaler.pkl

4. Configure MySQL
   Edit backend/config.py with your database credentials.

5. Run app
   python backend/app.py
   Open in browser:
   http://127.0.0.1:5000

NOTES

Input fields order:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

The history page shows recent predictions stored in MySQL.
