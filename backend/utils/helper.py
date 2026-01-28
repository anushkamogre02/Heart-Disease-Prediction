
# Utilities shared across app
# Define the exact order of features expected by the model
allowed_keys = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

def validate_input_order(form_dict):
    # Returns values in the expected order as strings (to be cast later)
    return [form_dict[k] for k in allowed_keys]
