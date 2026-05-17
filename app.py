from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os
import numpy as np

app = FastAPI(title="Diabetes Prediction API - Mix Models")

model_dir = "Model"

# Load models
xgb_model = joblib.load(os.path.join(model_dir, "diabetes_xgb_model.pkl"))
log_model = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
rf_model = joblib.load(os.path.join(model_dir, "random_forest_diabetes_model.pkl"))

print("All 3 models loaded successfully.")

# Expected columns for each model
cols_xgb = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'age', 'height', 'weight', 
    'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn', 'BMI', 
    'location_Louisa', 'gender_male', 'frame_medium', 'frame_small'
]

cols_logistic = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'glyhb', 'age', 'height', 'weight', 
    'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn', 'BMI', 'location', 'gender', 'frame'
]

cols_rf = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'glyhb', 'location', 'age', 'gender', 
    'height', 'weight', 'frame', 'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn', 'BMI'
]

class PatientData(BaseModel):
    chol: float
    stab_glu: float
    hdl: float
    ratio: float
    age: float
    height: float
    weight: float
    bp_1s: float
    bp_1d: float
    waist: float
    hip: float
    time_ppn: float
    bmi: float
    location: str
    gender: str
    frame: str
    glyhb: float = 5.0 # Optional with default

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/style.css")
def read_css():
    return FileResponse("style.css")

@app.get("/script.js")
def read_js():
    return FileResponse("script.js")

@app.post("/predict")
def predict(data: PatientData):
    def to_int(x):
        if isinstance(x, str):
            return 1 if 'diabetes' in x.lower() else 0
        return int(x)
        
    results = {'status': 'success'}
    
    # --- Prepare data for XGBoost (Manual Encoding) ---
    features_xgb = {
        'chol': data.chol,
        'stab.glu': data.stab_glu,
        'hdl': data.hdl,
        'ratio': data.ratio,
        'age': data.age,
        'height': data.height,
        'weight': data.weight,
        'bp.1s': data.bp_1s,
        'bp.1d': data.bp_1d,
        'waist': data.waist,
        'hip': data.hip,
        'time.ppn': data.time_ppn,
        'BMI': data.bmi,
        'location_Louisa': 1 if data.location == 'Louisa' else 0,
        'gender_male': 1 if data.gender == 'male' else 0,
        'frame_medium': 1 if data.frame == 'medium' else 0,
        'frame_small': 1 if data.frame == 'small' else 0
    }
    df_xgb = pd.DataFrame([features_xgb])[cols_xgb]
    
    # --- Prepare data for Pipelines (Raw Features) ---
    features_raw = {
        'chol': data.chol,
        'stab.glu': data.stab_glu,
        'hdl': data.hdl,
        'ratio': data.ratio,
        'glyhb': data.glyhb,
        'location': data.location,
        'age': data.age,
        'gender': data.gender,
        'height': data.height,
        'weight': data.weight,
        'frame': data.frame,
        'bp.1s': data.bp_1s,
        'bp.1d': data.bp_1d,
        'waist': data.waist,
        'hip': data.hip,
        'time.ppn': data.time_ppn,
        'BMI': data.bmi
    }
    
    df_logistic = pd.DataFrame([features_raw])[cols_logistic]
    df_rf = pd.DataFrame([features_raw])[cols_rf]
    
    # --- Predictions ---
    # 1. XGBoost
    results['xgb_prediction'] = to_int(xgb_model.predict(df_xgb)[0])
    results['xgb_probability'] = float(xgb_model.predict_proba(df_xgb)[0][1])
    
    # 2. Logistic (Pipeline)
    results['logistic_prediction'] = to_int(log_model.predict(df_logistic)[0])
    
    # 3. Random Forest (Pipeline)
    results['rf_prediction'] = to_int(rf_model.predict(df_rf)[0])
    results['rf_probability'] = float(rf_model.predict_proba(df_rf)[0][1])
    
    return results

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)