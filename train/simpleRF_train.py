# train_rf.py
import joblib
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier

from loader import load_ember_jsonl  # only if you ever want to rebuild the cache

# -------------------------------------------------------------------
# 1) Load precomputed features (fast path)
#    Assumes you already created ember24_features.joblib with:
#    { "X_train", "y_train", "X_test", "y_test", "X_challenge", "y_challenge" }
# -------------------------------------------------------------------
data = joblib.load("ember18_train_optimize.joblib")

X_train = data["X_train"]
y_train = data["y_train"]
# X_test = data["X_test"]
# y_test = data["y_test"]
# X_challenge = data["X_challenge"]
# y_challenge = data["y_challenge"]

print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)
# print("Challenge shape:", X_challenge.shape)

# -------------------------------------------------------------------
# 2) Train Random Forest
# -------------------------------------------------------------------
# clf = RandomForestClassifier(
#     # you can turn these back on as you tune:
#     n_estimators=300,
#     # criterion="gini",
#     max_depth=20,
#     min_samples_leaf=5,
#     min_samples_split=10,
#     max_features='sqrt',
#     class_weight='balanced',
#     n_jobs=-1,
#     random_state=9,
#     bootstrap=True,
#     oob_score=True,
#     max_samples=0.01,
# )
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=12,
    max_features=0.3,                # try 0.3; also try "sqrt"
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=9,
    bootstrap=True,
    # max_samples=0.8,               # optional: try enabling this
)


print("Training...")
clf.fit(X_train, y_train)
print("Training complete.")

# -------------------------------------------------------------------
# 3) Save model + feature columns together
# -------------------------------------------------------------------
model_out_path = "RF_ember18_optimize.gz"
to_save = {
    "model": clf,
    "features": list(X_train.columns),  # keep feature order for API input / eval
}

with gzip.open(model_out_path, "wb") as f:
    pickle.dump(to_save, f)

print(f"Saved gzipped model to {model_out_path}")
