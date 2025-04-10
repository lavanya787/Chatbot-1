import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import os

def load_saved_model(model_path):
    model_bundle = joblib.load(model_path)
    return model_bundle['model'], model_bundle['scaler']

def predict_from_model(model, scaler, X_new):
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)
def preprocess_dataframe(df):
    print(f"[🔍] Preprocessing {df.shape[0]} rows and {df.shape[1]} columns...\n")

    # 1. Show initial null counts
    print("[🔎] Initial missing values:\n", df.isnull().sum())

    # 2. Convert date columns safely
    for col in ['Date', 'Value Dt']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"[📅] Converted '{col}' to datetime")

    # 3. Convert numeric columns safely
    numeric_cols = ['Withdrawal Amt.', 'Deposit Amt.', 'Closing Balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"[🔢] Converted '{col}' to numeric")

    # 4. Debugging: show how many rows are NaN after conversion
    for col in numeric_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"[⚠️] {null_count} nulls in '{col}' after conversion")

    # 5. Drop rows only if ALL numeric columns are missing
    df = df.dropna(subset=numeric_cols, how='all')

    # 6. Fill any remaining NaN values with fallback
    df.fillna({'Narration': 'Unknown', 'Chq./Ref.No.': 'Unknown'}, inplace=True)

    # 7. Final check
    print(f"[✅] Data shape after cleaning: {df.shape}")
    if df.empty:
        print("[❌] ERROR: DataFrame is empty after preprocessing!")
    else:
        print("[✅] Preprocessing completed successfully!")

    return df


def train_and_save_model(csv_path, model_path):
    print(f"[📥] Reading file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Preprocess raw data (your custom logic)
    df = preprocess_dataframe(df)

    if df.empty or df.shape[0] < 5:
        raise ValueError(f"❌ DataFrame is too small or empty (shape: {df.shape}). Check the CSV content.")

    print(f"[📋] Columns after preprocessing: {df.columns.tolist()}")
    print(f"[🔍] Data types:\n{df.dtypes}")

    # Automatically select target column (last non-numeric column)
    non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
    if not non_numeric_cols:
        raise ValueError("❌ No non-numeric column found to use as target.")
    
    target_column = non_numeric_cols[-1]  # Pick last non-numeric column as target
    print(f"[🎯] Target column detected: '{target_column}'")

    if target_column not in df.columns:
        raise ValueError(f"❌ Target column '{target_column}' not found.")

    # Drop rows with nulls in target
    df = df.dropna(subset=[target_column])

    # Separate features and label
    X = df.drop(columns=[target_column])
    y = df[target_column]

# Keep only columns that are numeric (int or float)
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

    if X.empty:
        raise ValueError("❌ No numeric columns found for training.")

    print(f"[📐] Feature shape: {X.shape}")

    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump({'model': model, 'scaler': scaler, 'target': target_column}, model_path)
    print(f"[💾] Model saved at: {model_path}")
