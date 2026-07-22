import pandas as pd
import numpy as np
import json
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

with open("final_features.json") as f:
    features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

X_train, y_train = train[features].values, train["target_shots"].values
X_test, y_test = test[features].values, test["target_shots"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = PoissonRegressor(alpha=1e-4, max_iter=1000)
model.fit(X_train_s, y_train)

pred_train = model.predict(X_train_s)
pred_test = model.predict(X_test_s)

print("=== Coefficients (standardisés) ===")
for name, coef in sorted(zip(features, model.coef_), key=lambda x: -abs(x[1])):
    print(f"{name:40s} {coef:+.4f}")

print(f"\nMAE train: {np.mean(np.abs(pred_train - y_train)):.4f}")
print(f"MAE test:  {np.mean(np.abs(pred_test - y_test)):.4f}")

# Baseline naif: juste la moyenne de carrière (shots_career_avg) comme prédiction directe
naive_pred = test["shots_career_avg"].values
print(f"\nMAE baseline naive (juste career_avg): {np.mean(np.abs(naive_pred - y_test)):.4f}")

# Baseline encore plus simple: moyenne globale de la ligue
league_mean = train["target_shots"].mean()
print(f"MAE baseline moyenne de ligue ({league_mean:.3f} pour tous): {np.mean(np.abs(league_mean - y_test)):.4f}")

np.save("pred_test_poisson.npy", pred_test)
np.save("y_test.npy", y_test)
import pickle
with open("scaler.pkl","wb") as f: pickle.dump(scaler, f)
with open("poisson_model.pkl","wb") as f: pickle.dump(model, f)
