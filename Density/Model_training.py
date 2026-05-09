
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV,
)
from xgboost import XGBRegressor


def build_group_keys(df):
    s1 = df["Smiles#1"].astype(str).fillna("")
    s2 = df["Smiles#2"].astype(str).fillna("")
    x1 = pd.to_numeric(df["X#1"], errors="coerce").round(6).astype(str)
    x2 = pd.to_numeric(df["X#2"], errors="coerce").round(6).astype(str)
    return (s1 + "||" + s2 + "||" + x1 + "||" + x2).values


def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    return {
        "train_R2": r2_score(y_train, y_train_pred),
        "test_R2": r2_score(y_test, y_test_pred),
        "train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }


def main():
    target_col = "Density, g/cm3"
    exp_feature_cols = ["X#1", "X#2", "T, K"]

    df = pd.read_csv("Density_features.csv")

    desc_cols = [c for c in df.columns if c.endswith("_1") or c.endswith("_2")]
    feature_cols = desc_cols + exp_feature_cols

    group_key = build_group_keys(df)
    X = df[feature_cols]
    y = df[target_col]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=group_key))

    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    groups_train = group_key[train_idx]

    cv = GroupKFold(n_splits=5)

    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42, tree_method="hist"),
        param_distributions={
            "n_estimators": [300, 500, 800],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        n_iter=30, scoring="neg_mean_squared_error",
        cv=cv, random_state=42, n_jobs=-1, verbose=1,
    )
    xgb_search.fit(X_train, y_train, groups=groups_train)
    xgb_best = xgb_search.best_estimator_

    y_train_pred = xgb_best.predict(X_train)
    y_test_pred = xgb_best.predict(X_test)
    results = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)

    pd.DataFrame([results]).to_csv("Dens_performance.csv", index=False)
    joblib.dump({"model": xgb_best, "feature_cols": feature_cols, "target_col": target_col}, "Dens_model_xgb.pkl")


if __name__ == "__main__":
    main()
