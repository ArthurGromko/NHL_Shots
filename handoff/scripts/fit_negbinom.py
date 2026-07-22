import pandas as pd
import numpy as np
import json, pickle
from scipy.special import gammaln
from scipy.optimize import minimize

with open("final_features.json") as f:
    features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

with open("scaler.pkl","rb") as f: scaler = pickle.load(f)

X_train = scaler.transform(train[features].values)
X_test = scaler.transform(test[features].values)
y_train = train["target_shots"].values
y_test = test["target_shots"].values

# Ajouter une colonne d'intercept
X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
X_test_i = np.column_stack([np.ones(len(X_test)), X_test])
n_params = X_train_i.shape[1]

def negloglik(params, X, y):
    beta = params[:-1]
    log_alpha = params[-1]
    alpha = np.exp(log_alpha)
    eta = X @ beta
    eta = np.clip(eta, -20, 20)  # éviter l'overflow numérique
    mu = np.exp(eta)
    r = 1.0 / alpha
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
          + r * np.log(r / (r + mu))
          + y * np.log(mu / (r + mu)))
    return -np.sum(ll)

# warm start: intercept=log(moyenne), reste à 0, alpha initial ~0.25 (estimé plus tôt)
with open("poisson_model.pkl","rb") as f: poisson_model = pickle.load(f)
init_beta = np.concatenate([[poisson_model.intercept_], poisson_model.coef_])
init_params = np.concatenate([init_beta, [np.log(0.25)]])

print("Optimisation en cours (MLE binomiale négative)...")
result = minimize(
    negloglik, init_params, args=(X_train_i, y_train),
    method="L-BFGS-B",
    options={"maxiter": 300, "disp": True}
)

print("\nConvergé:", result.success, "-", result.message)
beta_hat = result.x[:-1]
alpha_hat = np.exp(result.x[-1])
print(f"\nAlpha (dispersion) estimé: {alpha_hat:.4f}")

print("\n=== Coefficients binomiale négative (standardisés) ===")
names = ["intercept"] + features
for name, coef in sorted(zip(names, beta_hat), key=lambda x: -abs(x[1])):
    print(f"{name:40s} {coef:+.4f}")

# Prédictions
mu_test = np.exp(np.clip(X_test_i @ beta_hat, -20, 20))
mae_nb = np.mean(np.abs(mu_test - y_test))
print(f"\nMAE test (binomiale négative): {mae_nb:.4f}")

np.save("pred_test_nb.npy", mu_test)
with open("nb_params.pkl","wb") as f:
    pickle.dump({"beta": beta_hat, "alpha": alpha_hat, "features": names}, f)
