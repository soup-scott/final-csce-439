#!/usr/bin/env python3
import gzip
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# 1) Load cached EMBER features
# -----------------------------
data = joblib.load("ember18_train_optimize.joblib")

X_tr = data["X_train"]
y_tr = data["y_train"]

data2 = joblib.load("ember18_challenge_optimize.joblib")

X_val = data2["X_challenge"]
y_val = data2["y_challenge"]

print("Full train shape:", X_tr.shape)

# -----------------------------
# 2) Create train/validation split
# -----------------------------
# X_tr, X_val, y_tr, y_val = train_test_split(
#     X_train_full,
#     y_train_full,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train_full,
# )

print("Train split shape:     ", X_tr.shape)
print("Validation split shape:", X_val.shape)

# -----------------------------
# 3) Define hyperparameter grid
#    (small & fast to start; expand later)
# -----------------------------
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [2, 3, 5],
    "max_features": ["sqrt", 0.3],
}

# Utility to iterate over grid
def iter_param_grid(grid):
    from itertools import product
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        yield params

results = []

# -----------------------------
# 4) Grid search loop
# -----------------------------
print("\nStarting hyperparameter search...\n")

for i, params in enumerate(iter_param_grid(param_grid), start=1):
    print(f"=== Config {i} ===")
    print(params)

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=12,              # keep from your current best
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=9,
        bootstrap=True,
        # you can try enabling this later:
        # max_samples=0.3,
    )

    clf.fit(X_tr, y_tr)

    # Validation predictions
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    try:
        val_auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        val_auc = np.nan

    print(f"  -> val_acc = {val_acc:.4f}, val_auc = {val_auc:.4f}\n")

    row = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"] if params["max_depth"] is not None else -1,
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": str(params["max_features"]),
        "val_acc": val_acc,
        "val_auc": val_auc,
    }
    results.append(row)

# -----------------------------
# 5) Collect results into DataFrame
# -----------------------------
results_df = pd.DataFrame(results)
results_df.sort_values(by="val_acc", ascending=False, inplace=True)

print("\nTop 10 configs by validation accuracy:\n")
print(results_df.head(10))

results_df.to_csv("rf_tuning_results.csv", index=False)
print("\nSaved all tuning results to rf_tuning_results.csv")

# -----------------------------
# 6) Simple plots
# -----------------------------

# a) val_acc vs n_estimators (averaged over other params)
acc_by_n = results_df.groupby("n_estimators")["val_acc"].mean().reset_index()

plt.figure()
plt.plot(acc_by_n["n_estimators"], acc_by_n["val_acc"], marker="o")
plt.xlabel("n_estimators")
plt.ylabel("Mean validation accuracy")
plt.title("RF: val accuracy vs n_estimators")
plt.grid(True)
plt.tight_layout()
plt.savefig("val_accuracy_vs_n_estimators.png")
plt.close()
print("Saved plot: val_accuracy_vs_n_estimators.png")

# b) val_acc vs max_depth (averaged over other params)
acc_by_depth = results_df.groupby("max_depth")["val_acc"].mean().reset_index()

plt.figure()
plt.plot(acc_by_depth["max_depth"], acc_by_depth["val_acc"], marker="o")
plt.xlabel("max_depth (-1 means None)")
plt.ylabel("Mean validation accuracy")
plt.title("RF: val accuracy vs max_depth")
plt.grid(True)
plt.tight_layout()
plt.savefig("val_accuracy_vs_max_depth.png")
plt.close()
print("Saved plot: val_accuracy_vs_max_depth.png")

# -----------------------------
# 7) Save the best model
# -----------------------------
best_row = results_df.iloc[0]
best_params = {
    "n_estimators": int(best_row["n_estimators"]),
    "max_depth": None if int(best_row["max_depth"]) == -1 else int(best_row["max_depth"]),
    "min_samples_leaf": int(best_row["min_samples_leaf"]),
    "max_features": 0.3 if best_row["max_features"] == "0.3" else "sqrt",
}

print("\nBest params from search:")
print(best_params)

best_clf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    min_samples_split=12,
    max_features=best_params["max_features"],
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=9,
    bootstrap=True,
)

print("\nRetraining best RF on full training set...")
best_clf.fit(X_train_full, y_train_full)
print("Done.")

save_obj = {
    "model": best_clf,
    "features": list(X_train_full.columns),
}

out_path = "RF_ember18_tuned.gz"
with gzip.open(out_path, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\nSaved tuned RF model to {out_path}")
