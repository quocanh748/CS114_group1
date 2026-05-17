import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv('Diab_pyth_data_clean.csv')
X = df.drop(columns=['diabetes_status'])
y = df['diabetes_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_features = ['chol', 'stab.glu', 'hdl', 'ratio', 'glyhb', 'age', 'height','weight', 'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn', 'BMI']
cat_features = ['location', 'gender', 'frame']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),('cat', cat_transformer, cat_features)
])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
joblib.dump(model_pipeline, 'random_forest_diabetes_model.pkl')
