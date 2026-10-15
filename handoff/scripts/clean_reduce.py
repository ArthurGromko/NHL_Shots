import glob
import pandas as pd

cols_needed = [
    "playerId", "name", "gameId", "season", "playerTeam", "opposingTeam",
    "home_or_away", "gameDate", "position", "situation", "icetime", "shifts",
    "I_F_shotsOnGoal", "I_F_missedShots", "I_F_blockedShotAttempts",
    "I_F_shotAttempts", "I_F_goals", "I_F_xGoals", "I_F_rebounds",
    "I_F_highDangerShots", "I_F_mediumDangerShots", "I_F_lowDangerShots",
    "OnIce_F_shotAttempts", "OnIce_A_shotAttempts"
]

chunksize = 200_000
# tous les CSV MoneyPuck de extracted/ (2008_to_2024.csv + une saison par fichier ensuite)
input_files = sorted(glob.glob("extracted/*.csv"))
print("Fichiers source:", input_files)

def read_chunks():
    for path in input_files:
        yield from pd.read_csv(path, usecols=cols_needed, chunksize=chunksize, low_memory=False)

reader = read_chunks()

out_path = "reduced_shots.csv"
first = True
total_rows = 0
bad_rows = 0

for i, chunk in enumerate(reader):
    # enlever les lignes d'en-tête dupliquées (season == 'season' etc.)
    before = len(chunk)
    chunk = chunk[chunk["season"] != "season"]
    bad_rows += before - len(chunk)

    # convertir les types
    chunk["season"] = pd.to_numeric(chunk["season"], errors="coerce")
    chunk["icetime"] = pd.to_numeric(chunk["icetime"], errors="coerce")
    chunk["I_F_shotsOnGoal"] = pd.to_numeric(chunk["I_F_shotsOnGoal"], errors="coerce")

    chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
    first = False
    total_rows += len(chunk)
    print(f"Chunk {i+1}: {len(chunk)} lignes valides, total cumulé: {total_rows}")

print(f"\nTerminé. Lignes valides: {total_rows}, lignes d'en-tête ignorées: {bad_rows}")
