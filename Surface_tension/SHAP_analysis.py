
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


def main():
    bundle = joblib.load("ST_model_xgb.pkl")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df_train = pd.read_csv("Train_set.csv")
    X_train = df_train[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train, check_additivity=False)

    shap.summary_plot(shap_values, X_train, max_display=15, show=False)
    plt.savefig("ST_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
