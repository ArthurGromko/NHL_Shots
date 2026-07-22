import pandas as pd
import numpy as np
import json, pickle
import statsmodels.api as sm

with open("final_features.json") as f:
    features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_train = scaler.transform(train[features].values)
X_test = scaler.transform(test[features].values)
y_train = train["target_shots"].values
y_test = test["target_shots"].values

X_train_i = sm.add_constant(X_train)
X_test_i = sm.add_constant(X_test)
names = ["intercept"] + features

print("Fit en cours (statsmodels NegativeBinomial, MLE natif)...")
model = sm.NegativeBinomial(y_train, X_train_i, loglike_method="nb2")
result = model.fit(method="bfgs", maxiter=300, disp=True)

print(result.summary(xname=names + ["alpha"]))

alpha_hat = result.params[-1]
beta_hat = result.params[:-1]

mu_test = np.exp(np.clip(X_test_i @ beta_hat, -20, 20))
mae_nb = np.mean(np.abs(mu_test - y_test))
print(f"\nMAE test (binomiale négative, statsmodels): {mae_nb:.4f}")
print(f"Alpha (dispersion) estimé: {alpha_hat:.4f}")

np.save("pred_test_nb_sm.npy", mu_test)
with open("nb_params_sm.pkl", "wb") as f:
    pickle.dump({"beta": beta_hat, "alpha": alpha_hat, "features": names}, f)
