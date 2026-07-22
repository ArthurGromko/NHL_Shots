import pandas as pd
import numpy as np
import pickle
from scipy.stats import nbinom, poisson

with open("nb_params.pkl","rb") as f:
    nb = pickle.load(f)

y_test = np.load("y_test.npy")
mu_test_nb = np.load("pred_test_nb.npy")
mu_test_poisson = np.load("pred_test_poisson.npy")
alpha = nb["alpha"]

# Paramétrisation scipy.stats.nbinom: n (nombre de succès), p
# Relation avec NB2: r = 1/alpha, p = r/(r+mu)
r = 1.0 / alpha

def prob_over(mu, line, dist="nb"):
    """P(shots > line), ex: line=2.5 -> P(X>=3)"""
    threshold = int(np.floor(line)) + 1  # ex: 2.5 -> X>=3
    if dist == "nb":
        p = r / (r + mu)
        return 1 - nbinom.cdf(threshold - 1, r, p)
    else:
        return 1 - poisson.cdf(threshold - 1, mu)

# --- Test de calibration sur une ligne courante: Over/Under 2.5 tirs ---
line = 2.5
actual_over = (y_test > line).astype(int)

pred_prob_nb = prob_over(mu_test_nb, line, "nb")
pred_prob_poisson = prob_over(mu_test_poisson, line, "poisson")

print(f"=== Calibration pour la ligne Over {line} tirs ===")
print(f"Taux réel de 'Over' dans le test: {actual_over.mean():.4f}")
print(f"Probabilité moyenne prédite (NB): {pred_prob_nb.mean():.4f}")
print(f"Probabilité moyenne prédite (Poisson): {pred_prob_poisson.mean():.4f}")

# Calibration par décile de probabilité prédite (NB)
df_cal = pd.DataFrame({"pred": pred_prob_nb, "actual": actual_over})
df_cal["decile"] = pd.qcut(df_cal["pred"], 10, duplicates="drop")
cal_table = df_cal.groupby("decile", observed=True).agg(
    pred_moyenne=("pred","mean"),
    reel_moyen=("actual","mean"),
    n=("actual","count")
)
print("\n=== Calibration par décile (NB) ===")
print(cal_table)

# Brier score (plus bas = mieux)
brier_nb = np.mean((pred_prob_nb - actual_over)**2)
brier_poisson = np.mean((pred_prob_poisson - actual_over)**2)
brier_naive = np.mean((actual_over.mean() - actual_over)**2)  # baseline = juste le taux moyen
print(f"\nBrier score NB: {brier_nb:.4f}")
print(f"Brier score Poisson: {brier_poisson:.4f}")
print(f"Brier score baseline (taux moyen constant): {brier_naive:.4f}")
