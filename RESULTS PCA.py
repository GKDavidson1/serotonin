# pca_seed_cosine_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# =========================
# 1. Load the Excel file
# =========================
file_path = "THESIS/fI paramters.xlsx"   # change path if needed
sheet_name = 0                    # first sheet

df = pd.read_excel(file_path, sheet_name=sheet_name)

print("\nColumns found:")
print(df.columns.tolist())


# =========================
# 2. Select variables
# =========================
# Use only columns beginning with 'seed_' as PCA inputs
seed_cols = [col for col in df.columns if col.startswith("seed_")]

# Target variable
target_col = "best_difference_vector_cosine"

# Check columns exist
if len(seed_cols) == 0:
    raise ValueError("No columns starting with 'seed_' were found.")

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' was not found.")

# Keep only needed columns and drop missing rows
analysis_df = df[seed_cols + [target_col]].dropna().copy()

print("\nSeed columns used for PCA:")
print(seed_cols)

print(f"\nNumber of rows used: {len(analysis_df)}")


# =========================
# 3. Standardise predictors
# =========================
X = analysis_df[seed_cols].values
y = analysis_df[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# 4. Run PCA
# =========================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

# Loadings: contribution of each original variable to each PC
loadings = pd.DataFrame(
    pca.components_.T,
    index=seed_cols,
    columns=[f"PC{i+1}" for i in range(len(seed_cols))]
)

# PCA scores dataframe
pc_scores_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

print("\nExplained variance ratio by principal component:")
for i, var in enumerate(explained_var, start=1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

print("\nCumulative explained variance:")
for i, var in enumerate(cum_explained_var, start=1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

print("\nPCA loadings:")
print(loadings.round(4))


# =========================
# 5. Relationship between PCs and cosine
# =========================
# Pearson correlations between each PC and the cosine target
pc_target_corrs = {}
for col in pc_scores_df.columns:
    r = np.corrcoef(pc_scores_df[col], y)[0, 1]
    pc_target_corrs[col] = r

pc_target_corrs_df = pd.DataFrame({
    "PC": list(pc_target_corrs.keys()),
    "Correlation_with_cosine": list(pc_target_corrs.values())
})

print("\nCorrelation of each PC with best_difference_vector_cosine:")
print(pc_target_corrs_df.round(4))


# =========================
# 6. Optional regression using first few PCs
# =========================
# Use enough PCs to explain ~80-90% variance, or just first 2-3 as a start
n_pcs_for_regression = min(3, X_pca.shape[1])

X_reg = pc_scores_df.iloc[:, :n_pcs_for_regression]
reg = LinearRegression()
reg.fit(X_reg, y)

regression_summary = pd.DataFrame({
    "PC": X_reg.columns,
    "Regression_coefficient": reg.coef_
})

print(f"\nLinear regression using first {n_pcs_for_regression} PCs:")
print(regression_summary.round(4))
print(f"Intercept: {reg.intercept_:.4f}")
print(f"R^2: {reg.score(X_reg, y):.4f}")


# =========================
# 7. Approximate original-variable influence on cosine
# =========================
# A simple way to map PC->target relationship back to the seed variables:
# weighted sum of loadings by PC-target correlation.
#
# This is not a formal causal coefficient, but it is a useful interpretation aid.
weighted_influence = loadings.copy()

for pc in weighted_influence.columns:
    weighted_influence[pc] = weighted_influence[pc] * pc_target_corrs[pc]

seed_influence = weighted_influence.sum(axis=1).sort_values(key=np.abs, ascending=False)

seed_influence_df = pd.DataFrame({
    "seed_parameter": seed_influence.index,
    "approx_influence_on_cosine": seed_influence.values
})

print("\nApproximate seed parameter influence on cosine")
print("(larger absolute values suggest stronger contribution through the PCs):")
print(seed_influence_df.round(4))


# =========================
# 8. Save outputs
# =========================
with pd.ExcelWriter("pca_seed_cosine_results.xlsx", engine="openpyxl") as writer:
    pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained_var))],
        "Explained_variance_ratio": explained_var,
        "Cumulative_explained_variance": cum_explained_var
    }).to_excel(writer, sheet_name="explained_variance", index=False)

    loadings.to_excel(writer, sheet_name="loadings")
    pc_scores_df.assign(best_difference_vector_cosine=y).to_excel(writer, sheet_name="pc_scores", index=False)
    pc_target_corrs_df.to_excel(writer, sheet_name="pc_target_correlations", index=False)
    regression_summary.to_excel(writer, sheet_name="pc_regression", index=False)
    seed_influence_df.to_excel(writer, sheet_name="approx_seed_influence", index=False)

print("\nResults saved to: pca_seed_cosine_results.xlsx")


# =========================
# 9. Plots
# =========================
# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.xticks(range(1, len(explained_var) + 1))
plt.tight_layout()
plt.show()

# Cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_explained_var) + 1), cum_explained_var, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.xticks(range(1, len(cum_explained_var) + 1))
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# Bar plot of approximate influence
plt.figure(figsize=(10, 6))
plt.bar(seed_influence_df["seed_parameter"], seed_influence_df["approx_influence_on_cosine"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Approximate Influence on Cosine")
plt.title("Approximate Influence of Seed Parameters on best_difference_vector_cosine")
plt.tight_layout()
plt.show()