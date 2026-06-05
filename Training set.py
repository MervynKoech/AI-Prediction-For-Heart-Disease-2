#!/usr/bin/env python
# coding: utf-8

# In[2]:


# train_save_models.py
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

print("1. Loading dataset...")
df = pd.read_csv("Heart.csv", na_values=["NA", "?", " "])
df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1})

x = df.drop(columns=["AHD", "HD"])
y = df["AHD"]

num_features = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak", "Ca"]
cat_features = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Thal"]

print("2. Configuring preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
], remainder="drop")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=SEED, stratify=y)
pos_weight_train = (y_train == 0).sum() / (y_train == 1).sum()

print("3. Building model pipelines...")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
])

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", xgb.XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, scale_pos_weight=pos_weight_train))
])

# Simplified param grids for faster execution (increase n_iter for production)
param_dist_rf = {
    "rf__n_estimators": [100, 200], "rf__max_depth": [None, 10, 15],
    "rf__min_samples_split": [2, 5], "rf__class_weight": ["balanced"]
}
param_dist_xgb = {
    "xgb__n_estimators": [100, 200], "xgb__max_depth": [3, 4, 5],
    "xgb__learning_rate": [0.05, 0.1], "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0], "xgb__gamma": [0, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("4. Training Random Forest...")
search_rf = RandomizedSearchCV(estimator=rf_pipeline, param_distributions=param_dist_rf, n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=SEED)
search_rf.fit(x_train, y_train)
best_rf = search_rf.best_estimator_

print("5. Training XGBoost...")
search_xgb = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=param_dist_xgb, n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=SEED)
search_xgb.fit(x_train, y_train)
best_xgb = search_xgb.best_estimator_

print("6. Evaluating models...")
y_prob_rf = best_rf.predict_proba(x_test)[:, 1]
y_pred_rf = best_rf.predict(x_test)
y_prob_xgb = best_xgb.predict_proba(x_test)[:, 1]
y_pred_xgb = best_xgb.predict(x_test)

metrics = {
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1-Score": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    },
    "XGBoost": {
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "Precision": precision_score(y_test, y_pred_xgb),
        "Recall": recall_score(y_test, y_pred_xgb),
        "F1-Score": f1_score(y_test, y_pred_xgb),
        "ROC-AUC": roc_auc_score(y_test, y_prob_xgb)
    }
}

print("7. Saving artifacts...")
joblib.dump(best_rf, "rf_model.pkl")
joblib.dump(best_xgb, "xgb_model.pkl")
joblib.dump(best_rf.named_steps['preprocessor'], "preprocessor.pkl")
joblib.dump(metrics, "metrics.pkl")

test_artifacts = {
    "y_test": y_test, "y_prob_rf": y_prob_rf, "y_prob_xgb": y_prob_xgb,
    "y_pred_rf": y_pred_rf, "y_pred_xgb": y_pred_xgb, "x_test": x_test
}
joblib.dump(test_artifacts, "test_artifacts.pkl")
print("✅ Done! Models and artifacts saved successfully.")


# In[ ]:




