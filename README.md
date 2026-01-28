
# Heart Disease Prediction System

End-to-end Flask + ML + MySQL project to predict heart disease risk.

## Setup

1. **Create environment & install deps**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r backend/requirements.txt
   ```

2. **Place dataset**
   Download Kaggle heart disease dataset and save as `dataset/heart.csv`.

3. **Train model**
   - Open `model_training.ipynb` in Jupyter and run all cells **or**
   - Adapt cells to a script if you prefer.

   This will produce `backend/model/heart_model.pkl` and `backend/model/scaler.pkl`.

4. **Configure MySQL**
   Edit `backend/config.py` with your credentials.

5. **Run app**
   ```bash
   python backend/app.py
   ```
   Visit http://127.0.0.1:5000

## Notes
- The form expects inputs in this order:
  `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.
- History page shows recent predictions saved to MySQL.
