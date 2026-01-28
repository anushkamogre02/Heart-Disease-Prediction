import os, sys

# ✅ Always add the project root to sys.path dynamically
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

print("✅ Project root added to path:", ROOT_DIR)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Now this import will work
from backend.model.preprocessing import FEATURE_COLUMNS
# Add root path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from backend.model.preprocessing import FEATURE_COLUMNS

# Load dataset
df = pd.read_csv(r"D:\PRO_Cdac\HeartDiseasePrediction\dataset\heart.csv")

X = df[FEATURE_COLUMNS]
y = df["target"]

# Split with stratification (preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 🧠 Model 1: Logistic Regression (good for probabilities)
# ==============================
log_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    penalty="l2"
)
log_model.fit(X_train_scaled, y_train)

# Evaluate Logistic Regression
y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_proba_log))

# ==============================
# 🧠 Model 2: RandomForest (improved tuning)
# ==============================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=4,
    class_weight="balanced_subsample",
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Evaluate RandomForest
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

# ==============================
# ✅ Pick best performing model
# ==============================
log_auc = roc_auc_score(y_test, y_proba_log)
rf_auc = roc_auc_score(y_test, y_proba_rf)
best_model = rf if rf_auc > log_auc else log_model
best_name = "Random Forest" if rf_auc > log_auc else "Logistic Regression"

print(f"\n✅ Selected Model: {best_name}")
print("Saving model and scaler...")

# Save both
model_path = os.path.abspath(os.path.join(os.getcwd(), "../heart_model.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "../scaler.pkl"))

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ Model saved at: {model_path}")
print(f"✅ Scaler saved at: {scaler_path}")
