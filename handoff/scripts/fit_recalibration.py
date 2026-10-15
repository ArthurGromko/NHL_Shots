import pandas as pd
import numpy as np
import json, pickle
import statsmodels.api as sm
from scipy.stats import nbinom
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# Les prédictions NB sont trop resserrées autour de la moyenne (déciles bas
# surestimés, déciles hauts sous-estimés). La variance autour de mu colle bien
# à la NB, donc on recalibre mu lui-même (une seule correction, cohérente pour
# toutes les lignes Over/Under).
#
# Pour ne pas toucher au test : on ré-entraîne le même modèle NB sur
# 2008-2022, on prédit la saison 2023 (validation), et on apprend la
# correction mu -> mu' sur cette saison. La correction est ensuite appliquée
# aux prédictions du modèle principal (entraîné jusqu'à 2023) sur le test.

with open("final_features.json") as f:
    features = json.load(f)

train = pd.read_csv("model_train_clean.csv", low_memory=False)
test = pd.read_csv("model_test_clean.csv", low_memory=False)

VALID_SEASON = train["season"].max()
inner = train[train["season"] < VALID_SEASON]
valid = train[train["season"] == VALID_SEASON]
print(f"Fit interne: {len(inner)} lignes (<= {VALID_SEASON - 1}), validation: {len(valid)} lignes ({VALID_SEASON})")

scaler = StandardScaler()
X_inner = sm.add_constant(scaler.fit_transform(inner[features].values))
X_valid = sm.add_constant(scaler.transform(valid[features].values))
y_inner = inner["target_shots"].values
y_valid = valid["target_shots"].values

print("Fit NB interne...")
result = sm.NegativeBinomial(y_inner, X_inner, loglike_method="nb2").fit(method="bfgs", maxiter=300, disp=False)
beta, alpha_inner = result.params[:-1], result.params[-1]
mu_valid = np.exp(np.clip(X_valid @ beta, -20, 20))

# --- Correction 1: log-linéaire, log mu' = a + b log mu (GLM Poisson) ---
glm = sm.GLM(y_valid, sm.add_constant(np.log(mu_valid)), family=sm.families.Poisson()).fit()
a, b = glm.params
print(f"Log-linéaire: a={a:+.4f}, b={b:.4f} (b>1 => prédictions trop resserrées)")

# --- Correction 2: isotonique (monotone, non paramétrique) ---
iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso.fit(mu_valid, y_valid)

# --- Évaluation sur le test, avec les prédictions du modèle principal ---
with open("nb_params_sm.pkl", "rb") as f:
    alpha = pickle.load(f)["alpha"]
y_test = np.load("y_test.npy")
mu_raw = np.load("pred_test_nb_sm.npy")

variants = {
    "brut": mu_raw,
    "log-lineaire": np.exp(a + b * np.log(mu_raw)),
    "isotonique": np.clip(iso.predict(mu_raw), 1e-3, None),
}

r = 1.0 / alpha

def prob_over(mu, line):
    return 1 - nbinom.cdf(int(np.floor(line)), r, r / (r + mu))

def decile_error(pred, actual):
    """Écart moyen absolu prédit/réel sur les déciles de probabilité prédite."""
    d = pd.DataFrame({"pred": pred, "actual": actual})
    d["decile"] = pd.qcut(d["pred"], 10, duplicates="drop")
    g = d.groupby("decile", observed=True)[["pred", "actual"]].mean()
    return (g["pred"] - g["actual"]).abs().mean(), (g["pred"] - g["actual"]).abs().max()

lines = [0.5, 1.5, 2.5, 3.5, 4.5]
rows = []
for name, mu in variants.items():
    row = {"variante": name, "MAE": np.mean(np.abs(mu - y_test))}
    for line in lines:
        actual = (y_test > line).astype(int)
        p = prob_over(mu, line)
        row[f"Brier {line}"] = np.mean((p - actual) ** 2)
        row[f"ErrDec {line}"], row[f"ErrDecMax {line}"] = decile_error(p, actual)
    rows.append(row)
res = pd.DataFrame(rows).set_index("variante")

pd.set_option("display.width", 200)
print(f"\n=== Test (n={len(y_test)}) — MAE et Brier par ligne ===")
print(res[["MAE"] + [f"Brier {l}" for l in lines]].round(4))
print("\n=== Erreur de calibration par déciles (moyenne absolue, en points de probabilité) ===")
print((res[[f"ErrDec {l}" for l in lines]] * 100).round(2))
print("\n=== Pire décile (écart absolu max, en points de probabilité) ===")
print((res[[f"ErrDecMax {l}" for l in lines]] * 100).round(2))

print("\n=== Calibration par décile, Over 2.5 ===")
actual = (y_test > 2.5).astype(int)
for name, mu in variants.items():
    d = pd.DataFrame({"pred": prob_over(mu, 2.5), "actual": actual})
    d["decile"] = pd.qcut(d["pred"], 10)
    print(f"\n--- {name} ---")
    print(d.groupby("decile", observed=True).agg(pred_moyenne=("pred", "mean"), reel_moyen=("actual", "mean")).round(4))

with open("recalibration.pkl", "wb") as f:
    pickle.dump({"loglinear": {"a": a, "b": b}, "isotonic": iso, "valid_season": int(VALID_SEASON)}, f)
for name, mu in variants.items():
    if name != "brut":
        np.save(f"pred_test_nb_recal_{name.replace('-', '_')}.npy", mu)
