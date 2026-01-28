
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
from database import get_db, init_db, insert_prediction, fetch_predictions
from utils.helper import validate_input_order, allowed_keys

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

# Try to load model & scaler, handle nicely if missing
try:
    model = joblib.load("backend/model/heart_model.pkl")
    scaler = joblib.load("backend/model/scaler.pkl")
except Exception as e:
    model = None
    scaler = None
    print(f"[WARN] Could not load model/scaler: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    rows = fetch_predictions(limit=50)
    return render_template("history.html", rows=rows)
@app.route("/predict", methods=["POST"])
def predict():
    # Expect form keys matching 'allowed_keys' in correct order
    form = request.form.to_dict()
    missing = [k for k in allowed_keys if k not in form or form[k] == ""]
    if missing:
        flash(f"Missing fields: {', '.join(missing)}")
        return redirect(url_for("home"))

    try:
        values_ordered = validate_input_order(form)
        values_float = [float(v) for v in values_ordered]

        if scaler is None or model is None:
            flash("Model or scaler not found. Train and export them first.")
            return redirect(url_for("home"))

        # ✅ Scale input data
        scaled = scaler.transform([values_float])

        # ✅ Predict probabilities for both classes (0 = No Disease, 1 = Disease)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(scaled)[0]
            prob_no_disease = float(probas[0])
            prob_disease = float(probas[1])
        else:
            prob_disease = None

        # ⚖️ Apply custom decision threshold (e.g., 0.4 instead of 0.5)
        threshold = 0.4
        pred = 1 if prob_disease is not None and prob_disease >= threshold else 0

        # ✅ Interpret prediction
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

        # ✅ Save to DB (store actual probability of disease)
        insert_prediction(values_float, pred, prob_disease)

        # ✅ Pass both probabilities to template
        return render_template(
            "result.html",
            prediction=result,
            probability=prob_disease,
            threshold=threshold,
            prob_no_disease=prob_no_disease
        )

    except Exception as e:
        flash(f"Error during prediction: {e}")
        return redirect(url_for("home"))

if __name__ == "__main__":
    # Ensure DB exists and table is ready
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
