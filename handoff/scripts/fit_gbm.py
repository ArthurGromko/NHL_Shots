import pandas as pd
import numpy as np
import json, pickle
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb

with open("final_features.json") as f:
    features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

X_train, y_train = train[features].values, train["target_shots"].values
X_test, y_test = test[features].values, test["target_shots"].values

# --- XGBoost, objectif Poisson (compte de tirs) ---
xgb_model = xgb.XGBRegressor(
    objective="count:poisson",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
mae_xgb = np.mean(np.abs(pred_xgb - y_test))
print(f"XGBoost  - MAE test: {mae_xgb:.4f}")

print("\n=== Importance des features (XGBoost, gain) ===")
importances = xgb_model.get_booster().get_score(importance_type="gain")
for name, score in sorted(importances.items(), key=lambda x: -x[1]):
    idx = int(name[1:])
    print(f"{features[idx]:40s} {score:.1f}")

# --- LightGBM, objectif Poisson ---
lgb_model = lgb.LGBMRegressor(
    objective="poisson",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_test)
mae_lgb = np.mean(np.abs(pred_lgb - y_test))
print(f"\nLightGBM - MAE test: {mae_lgb:.4f}")

# --- Comparaison Brier score Over/Under 2.5, en supposant Poisson autour de mu prédit ---
def prob_over(mu, line):
    threshold = int(np.floor(line)) + 1
    return 1 - poisson.cdf(threshold - 1, mu)

actual_over = (y_test > 2.5).astype(int)
brier_xgb = np.mean((prob_over(pred_xgb, 2.5) - actual_over) ** 2)
brier_lgb = np.mean((prob_over(pred_lgb, 2.5) - actual_over) ** 2)
print(f"\nBrier score XGBoost:  {brier_xgb:.4f}")
print(f"Brier score LightGBM: {brier_lgb:.4f}")

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("lgb_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)
np.save("pred_test_xgb.npy", pred_xgb)
np.save("pred_test_lgb.npy", pred_lgb)
