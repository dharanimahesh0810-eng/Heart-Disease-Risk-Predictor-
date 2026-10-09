"""
train_model.py
----------------
Reproduces the training pipeline from the notebook (Refinedcode.ipynb)
and saves the fitted model to disk as 'heart_disease_model.pkl'.

Run this ONCE (or whenever you want to retrain) before running the
Streamlit app. Make sure 'framingham.csv' is in the same folder.
"""

import pandas as pd
import joblib

from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("framingham.csv")

X = df.drop(columns="TenYearCHD")
y = df.TenYearCHD

# Same split ratios as the notebook: 64% train / 20% test / 16% val
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Oversample the training data only (fix class imbalance)
# ---------------------------------------------------------
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

print("Original train shape:", X_train.shape)
print("Oversampled train shape:", X_train_over.shape)

# ---------------------------------------------------------
# 3. Build pipeline + hyperparameter grid search
# ---------------------------------------------------------
base_model = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "simpleimputer__strategy": ["mean", "median"],
    "randomforestclassifier__n_estimators": [10, 100],
    "randomforestclassifier__max_depth": [2, 10, 30],
}

grid_search = GridSearchCV(
    base_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring="roc_auc",
)

grid_search.fit(X_train_over, y_train_over)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)

# ---------------------------------------------------------
# 4. Evaluate on the held-out test set
# ---------------------------------------------------------
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ---------------------------------------------------------
# 5. Save the trained model to disk
# ---------------------------------------------------------
joblib.dump(best_model, "heart_disease_model.pkl")
print("Model saved to heart_disease_model.pkl")
