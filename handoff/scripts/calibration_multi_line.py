import pandas as pd
import numpy as np
import pickle
from scipy.stats import nbinom, poisson

with open("nb_params.pkl", "rb") as f:
    nb = pickle.load(f)

y_test = np.load("y_test.npy")
mu_test_nb = np.load("pred_test_nb.npy")
mu_test_poisson = np.load("pred_test_poisson.npy")
alpha = nb["alpha"]
r = 1.0 / alpha


def prob_over(mu, line, dist="nb"):
    """P(shots > line), ex: line=2.5 -> P(X>=3)"""
    threshold = int(np.floor(line)) + 1
    if dist == "nb":
        p = r / (r + mu)
        return 1 - nbinom.cdf(threshold - 1, r, p)
    else:
        return 1 - poisson.cdf(threshold - 1, mu)


lines = [0.5, 1.5, 2.5, 3.5, 4.5]

print(f"{'Ligne':>6} {'n':>7} {'Taux réel':>10} {'Pred NB':>9} {'Pred Poi':>9} "
      f"{'Brier NB':>9} {'Brier Poi':>10} {'Brier naif':>11}")

for line in lines:
    actual_over = (y_test > line).astype(int)
    pred_prob_nb = prob_over(mu_test_nb, line, "nb")
    pred_prob_poisson = prob_over(mu_test_poisson, line, "poisson")

    brier_nb = np.mean((pred_prob_nb - actual_over) ** 2)
    brier_poisson = np.mean((pred_prob_poisson - actual_over) ** 2)
    brier_naive = np.mean((actual_over.mean() - actual_over) ** 2)

    print(f"{line:>6} {len(y_test):>7} {actual_over.mean():>10.4f} "
          f"{pred_prob_nb.mean():>9.4f} {pred_prob_poisson.mean():>9.4f} "
          f"{brier_nb:>9.4f} {brier_poisson:>10.4f} {brier_naive:>11.4f}")

print("\n=== Calibration par décile (NB), pour chaque ligne ===")
for line in lines:
    actual_over = (y_test > line).astype(int)
    pred_prob_nb = prob_over(mu_test_nb, line, "nb")

    df_cal = pd.DataFrame({"pred": pred_prob_nb, "actual": actual_over})
    try:
        df_cal["decile"] = pd.qcut(df_cal["pred"], 10, duplicates="drop")
    except ValueError:
        print(f"\n--- Ligne {line}: distribution trop concentrée pour 10 déciles, skip ---")
        continue
    cal_table = df_cal.groupby("decile", observed=True).agg(
        pred_moyenne=("pred", "mean"),
        reel_moyen=("actual", "mean"),
        n=("actual", "count"),
    )
    print(f"\n--- Ligne {line} ---")
    print(cal_table)
