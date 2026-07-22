import pandas as pd

df = pd.read_csv("reduced_shots_2008_2024.csv", low_memory=False)
df_all = df[df["situation"] == "all"].copy()
df_all = df_all[df_all["icetime"] > 0]  # exclure les joueurs qui n'ont pas joué

mean_shots = df_all["I_F_shotsOnGoal"].mean()
var_shots = df_all["I_F_shotsOnGoal"].var()

print(f"Moyenne: {mean_shots:.3f}")
print(f"Variance: {var_shots:.3f}")
print(f"Ratio variance/moyenne: {var_shots/mean_shots:.3f}  (=1.0 => Poisson pur, >1.0 => surdispersion => binomiale négative)")

# Regarder par groupe de TOI pour voir si la surdispersion vient du TOI variable
df_all["toi_min"] = df_all["icetime"] / 60
print("\nPar tranche de TOI (minutes):")
bins = [0, 8, 12, 16, 20, 30]
df_all["toi_bin"] = pd.cut(df_all["toi_min"], bins=bins)
grouped = df_all.groupby("toi_bin", observed=True)["I_F_shotsOnGoal"].agg(["mean","var","count"])
grouped["ratio"] = grouped["var"] / grouped["mean"]
print(grouped)
