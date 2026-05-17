# ==================================================
# 1. Imports
# ==================================================

import inspect
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

try:
    from IPython.display import display
except ImportError:
    display = print


# ==================================================
# 2. Config
# ==================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_FILE = "Diab_pyth_data_clean.csv"
LOGISTIC_MODEL_FILE = "logistic_regression.pkl"

TARGET_COL = "diabetes_status"
SCALER_TYPE = "standard"  # Options: "standard" or "robust"

CONTINUOUS_FEATURES = [
    "chol",
    "stab.glu",
    "hdl",
    "ratio",
    "glyhb",
    "age",
    "height",
    "weight",
    "bp.1s",
    "bp.1d",
    "waist",
    "hip",
    "time.ppn",
    "BMI",
]

CATEGORICAL_FEATURES = [
    "location",
    "gender",
    "frame",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


# ==================================================
# 3. Load Data
# ==================================================

def resolve_path(filename):
    """Find a file in local, Colab, or Kaggle input paths."""
    direct_paths = [
        Path(filename),
        Path("/content") / filename,
        Path("/kaggle/working") / filename,
    ]

    for path in direct_paths:
        if path.exists():
            return path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find {filename}. Upload it to the notebook working directory "
        "or update DATA_FILE."
    )


data_path = resolve_path(DATA_FILE)
df = pd.read_csv(data_path)

required_columns = ALL_FEATURES + [TARGET_COL]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert numeric columns safely; imputers will handle missing values.
for col in CONTINUOUS_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Strip categorical labels while preserving real missing values.
for col in CATEGORICAL_FEATURES:
    df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

X = df[ALL_FEATURES]
y = df[TARGET_COL]

print("Data path:", data_path)
print("Shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ==================================================
# 4. Preprocessing
# ==================================================

def make_scaler():
    if SCALER_TYPE == "robust":
        return RobustScaler()
    if SCALER_TYPE == "standard":
        return StandardScaler()
    raise ValueError("SCALER_TYPE must be 'standard' or 'robust'.")


def make_onehot_encoder():
    # sklearn >= 1.2 uses sparse_output; older versions use sparse.
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor(numeric_features=None, categorical_features=None):
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", make_scaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ==================================================
# 5. Train Logistic Regression
# ==================================================

def make_logistic_regression():
    # multi_class was removed in sklearn 1.8. With lbfgs and n_classes >= 3,
    # sklearn uses multinomial loss automatically.
    params = {
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        params["multi_class"] = "multinomial"
    return LogisticRegression(**params)


def build_logistic_pipeline(numeric_features=None, categorical_features=None):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("classifier", make_logistic_regression()),
        ]
    )


logistic_models = {
    "continuous": {
        "feature_type": "Continuous variables",
        "model": build_logistic_pipeline(
            numeric_features=CONTINUOUS_FEATURES,
            categorical_features=None,
        ),
    },
    "categorical": {
        "feature_type": "Categorical variables",
        "model": build_logistic_pipeline(
            numeric_features=None,
            categorical_features=CATEGORICAL_FEATURES,
        ),
    },
    "all_features": {
        "feature_type": "All variables",
        "model": build_logistic_pipeline(
            numeric_features=CONTINUOUS_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        ),
    },
}

for key, info in logistic_models.items():
    print(f"\nTraining Logistic Regression - {info['feature_type']}...")
    info["model"].fit(X_train, y_train)


# ==================================================
# 6. Validation Metrics For Selecting Saved Model
# ==================================================

metric_rows = []

for key, info in logistic_models.items():
    model = info["model"]
    y_pred = model.predict(X_test)

    metric_rows.append(
        {
            "Key": key,
            "Feature Type": info["feature_type"],
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
    )

metrics_df = pd.DataFrame(metric_rows)

# If metrics tie, prefer the simpler continuous-only model, then all features.
tie_breaker = {
    "Continuous variables": 0,
    "All variables": 1,
    "Categorical variables": 2,
}
metrics_df["Tie Breaker"] = metrics_df["Feature Type"].map(tie_breaker)

metrics_sorted = metrics_df.sort_values(
    ["Macro F1", "Weighted F1", "Tie Breaker"],
    ascending=[False, False, True],
)

print("\nLogistic Regression validation metrics:")
display(
    metrics_sorted[
        ["Feature Type", "Accuracy", "Macro F1", "Weighted F1"]
    ].round(4)
)


# ==================================================
# 7. Save Best Logistic Regression Model
# ==================================================

best_row = metrics_sorted.iloc[0]
best_key = best_row["Key"]
best_model = logistic_models[best_key]["model"]

joblib.dump(best_model, LOGISTIC_MODEL_FILE)

print("\nSaved Logistic Regression model to:", LOGISTIC_MODEL_FILE)
print("Saved feature group:", best_row["Feature Type"])
print(f"Validation macro F1-score: {best_row['Macro F1']:.4f}")
