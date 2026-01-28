
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

def evaluate_clf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        # For ROC-AUC we need probabilities; fallback gracefully
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:,1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = None
        auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
