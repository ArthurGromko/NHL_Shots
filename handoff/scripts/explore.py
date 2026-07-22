import pandas as pd

df = pd.read_csv("reduced_shots_2008_2024.csv", low_memory=False)
print("Shape:", df.shape)
print("\nSituations:", df["situation"].unique())
print("\nSaisons:", sorted(df["season"].unique()))
print("\nJoueurs uniques:", df["playerId"].nunique())
print("\nMatchs uniques:", df["gameId"].nunique())

# Filtrer sur "all" (agrégat complet du match, toutes situations confondues) pour un premier coup d'oeil
df_all = df[df["situation"] == "all"].copy()
print("\n--- Lignes avec situation='all' ---")
print("Shape:", df_all.shape)
print("\nDistribution de I_F_shotsOnGoal (tirs au but par match-joueur):")
print(df_all["I_F_shotsOnGoal"].describe())

print("\nExemple - Connor McDavid quelques matchs:")
mcdavid = df_all[df_all["name"].str.contains("McDavid", na=False)]
print(mcdavid[["name","season","gameDate","icetime","I_F_shotsOnGoal"]].tail(10))
