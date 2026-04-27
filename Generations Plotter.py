from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================

ROOT_DIR = Path(r"C:\Users\GlenA\Documents\GitHub\serotonin\Model RUNS\SEEDED LOGS runtime 40seconds")  # change this
FILE_PATTERN = "*generations*.jsonl"

# For exact filename only:
# FILE_PATTERN = "generations.jsonl"


# =========================
# FUNCTIONS
# =========================

def read_generations_jsonl(file_path):
    """
    Reads a JSONL optimisation history file containing:
    generation, best_fitness, mean_fitness, etc.
    """
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    required_cols = {"generation", "best_fitness"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"{file_path} is missing columns: {missing}")

    df = df.sort_values("generation")

    # Running minimum of best_fitness
    # Example: 10, 8, 9 becomes 10, 8, 8
    df["min_best_fitness_so_far"] = df["best_fitness"].cummin()

    return df


# =========================
# FIND FILES
# =========================

files = sorted(ROOT_DIR.rglob(FILE_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No files matching '{FILE_PATTERN}' found in {ROOT_DIR}"
    )

print(f"Found {len(files)} file(s):")
for f in files:
    print(" -", f)


# =========================
# PLOT 1: RAW BEST FITNESS
# =========================

case_labels = [f"Case {chr(65 + i)}" for i in range(len(files))]

plt.figure(figsize=(12, 7))

for file_path, label in zip(files, case_labels):
    df = read_generations_jsonl(file_path)

    plt.plot(
        df["generation"],
        df["best_fitness"],
        label=label,
        linewidth=1.8
    )

plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.title("Best fitness across generations")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# =========================
# PLOT 2: MIN BEST FITNESS SO FAR
# =========================

plt.figure(figsize=(12, 7))

for file_path, label in zip(files, case_labels):
    df = read_generations_jsonl(file_path)

    plt.plot(
        df["generation"],
        df["min_best_fitness_so_far"],
        label=label,
        linewidth=1.8
    )

plt.xlabel("Generation")
plt.ylabel("Minimum best fitness so far")
plt.title("Running minimum best fitness across generations")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
