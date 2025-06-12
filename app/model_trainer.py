import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

def train_model(df):
    df = df.copy()

    # --- SMART TARGET DETECTION ---
    possible_targets = []
    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        binary_like = set(unique_vals) <= {'0', '1', 'true', 'false', 'yes', 'no'}
        if binary_like:
            possible_targets.append(col)

    if not possible_targets:
        raise ValueError("❌ No binary column found. Please include a fraud label column with values like 0/1 or yes/no.")

    known_names = ['fraud', 'is_fraud', 'fraudulent', 'target', 'class', 'label']
    selected_col = next((col for col in possible_targets if any(name in col.lower() for name in known_names)), possible_targets[0])

    print(f"✅ Detected fraud column: {selected_col}")

    # --- CLEAN TARGET COLUMN ---
    df[selected_col] = df[selected_col].astype(str).str.lower().map({
        '1': 1, '0': 0, 'true': 1, 'false': 0, 'yes': 1, 'no': 0
    })
    df = df.dropna(subset=[selected_col])
    df[selected_col] = df[selected_col].astype(int)

    # --- REMOVE NON-NUMERIC COLUMNS BEFORE ENCODING ---
    for col in df.columns:
        if df[col].dtype == 'object' and col != selected_col:
            if df[col].nunique() > 50:  # Avoid exploding features from high-cardinality
                df.drop(columns=[col], inplace=True)

    # --- SPLIT FEATURES AND TARGET ---
    y = df[selected_col]
    X = df.drop(columns=[selected_col])

    # --- ONE-HOT ENCODE & SANITIZE ---
    X = pd.get_dummies(X)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.astype('float32')  # Force float32 to match XGBoost expectations

    # --- TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- TRAIN MODEL ---
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # --- SAVE MODEL ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.pkl")

    return model, X_test, selected_col
