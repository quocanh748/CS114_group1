"""
Script retrain Logistic Regression và lưu cả scaler.
Chạy 1 lần: python save_scaler.py
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ── Đường dẫn ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "Diab_pyth_data_clean.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "..", "Model")

# ── 1. Đọc dữ liệu ────────────────────────────────────────────────────────────
print("Dang doc du lieu...")
df = pd.read_csv(DATA_PATH)
print(f"  -> {len(df)} dong, {len(df.columns)} cot")

# ── 2. Chuẩn bị feature / target ─────────────────────────────────────────────
y_class = df['glyhb'].apply(lambda x: 1 if x >= 6.5 else 0)

cols_drop = ['glyhb']
if 'diabetes_status' in df.columns: cols_drop.append('diabetes_status')
if 'diabetes'        in df.columns: cols_drop.append('diabetes')

X = df.drop(columns=cols_drop)
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
print(f"  -> Features sau encoding: {list(X_encoded.columns)}")

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

# ── 4. Train/Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# ── 5. Train Logistic Regression ──────────────────────────────────────────────
print("\nDang train Logistic Regression...")
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"  -> Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# ── 6. Lưu model + scaler ────────────────────────────────────────────────────
model_path  = os.path.join(MODEL_DIR, "diabetes_logistic_classifier.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler_logistic.pkl")
cols_path   = os.path.join(MODEL_DIR, "logistic_feature_cols.pkl")

joblib.dump(log_reg,             model_path)
joblib.dump(scaler,              scaler_path)
joblib.dump(X_encoded.columns.tolist(), cols_path)

print(f"\n[OK] Da luu model : {model_path}")
print(f"[OK] Da luu scaler: {scaler_path}")
print(f"[OK] Da luu cols  : {cols_path}")
print("\n=== XONG! Restart app_web.py de ap dung ===")
