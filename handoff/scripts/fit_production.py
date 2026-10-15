import pandas as pd
import numpy as np
import json, pickle
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Modèle de production: même NB que fit_negbinom_statsmodels.py, mais entraîné sur
# tous les matchs joués (toutes saisons, y compris la saison en cours jusqu'à la
# dernière mise à jour MoneyPuck). Pas d'ensemble de test ici: l'évaluation se fait
# avec le split train/test de prepare_model_data.py.

with open("final_features.json") as f:
    features = json.load(f)

df = pd.read_csv("model_all_clean.csv", low_memory=False)
print(f"Fit production: {len(df)} lignes, saisons {df['season'].min()}-{df['season'].max()}, "
      f"dernier match {df['gameDate'].max()}")

scaler = StandardScaler()
X = sm.add_constant(scaler.fit_transform(df[features].values))
y = df["target_shots"].values

result = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(method="bfgs", maxiter=300, disp=False)
print(f"Convergé: {result.mle_retvals['converged']}")

beta, alpha = result.params[:-1], result.params[-1]
for name, coef in zip(["intercept"] + features, beta):
    print(f"{name:40s} {coef:+.4f}")
print(f"\nAlpha (dispersion) estimé: {alpha:.4f}")

with open("prod_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("prod_nb_params.pkl", "wb") as f:
    pickle.dump({"beta": beta, "alpha": alpha, "features": features,
                 "last_game_date": str(df["gameDate"].max())}, f)
