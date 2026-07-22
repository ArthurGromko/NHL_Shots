import pandas as pd
import numpy as np
import json, pickle
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

with open("final_features.json") as f:
    pooled_features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

# Features numériques communes, sans les dummies de position (le groupe remplace l'info)
numeric_features = [f for f in pooled_features if not f.startswith("position_")]

# --- Baseline: modèle poolé (déjà entraîné) appliqué séparément à chaque groupe ---
pooled_pred = np.load("pred_test_poisson.npy")
pooled_y = np.load("y_test.npy")
assert len(pooled_pred) == len(test), "ordre de test.csv différent de celui utilisé pour le modèle poolé"

is_defense_test = test["position_D"].values.astype(bool)

def prob_over(mu, line=2.5):
    threshold = int(np.floor(line)) + 1
    return 1 - poisson.cdf(threshold - 1, mu)

def report(name, y, pred):
    mae = np.mean(np.abs(pred - y))
    brier = np.mean((prob_over(pred) - (y > 2.5).astype(int)) ** 2)
    print(f"{name:30s} n={len(y):6d}  MAE={mae:.4f}  Brier={brier:.4f}")
    return mae, brier

print("=== Modèle poolé, évalué séparément par groupe ===")
report("Poolé - Attaquants", pooled_y[~is_defense_test], pooled_pred[~is_defense_test])
report("Poolé - Défenseurs", pooled_y[is_defense_test], pooled_pred[is_defense_test])

# --- Modèles segmentés: un Poisson par groupe ---
groups = {
    "forwards": (train["position_D"] == 0, test["position_D"] == 0),
    "defense": (train["position_D"] == 1, test["position_D"] == 1),
}

print("\n=== Modèles segmentés (un par groupe) ===")
results = {}
for label, (train_mask, test_mask) in groups.items():
    tr, te = train[train_mask], test[test_mask]
    X_train, y_train = tr[numeric_features].values, tr["target_shots"].values
    X_test, y_test = te[numeric_features].values, te["target_shots"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = PoissonRegressor(alpha=1e-4, max_iter=1000)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    mae, brier = report(f"Segmenté - {label}", y_test, pred)
    results[label] = {"mae": mae, "brier": brier}

    print(f"  Coefficients ({label}):")
    for fname, coef in sorted(zip(numeric_features, model.coef_), key=lambda x: -abs(x[1])):
        print(f"    {fname:35s} {coef:+.4f}")

    with open(f"poisson_{label}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"scaler_{label}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.save(f"pred_test_{label}.npy", pred)
    np.save(f"y_test_{label}.npy", y_test)

with open("segmented_features.json", "w") as f:
    json.dump(numeric_features, f)
