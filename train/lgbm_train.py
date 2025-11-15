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
data = joblib.load("lgbm_train_val.joblib")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
# X_challenge = data["X_challenge"]
# y_challenge = data["y_challenge"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
# print("Challenge shape:", X_challenge.shape)

# -------------------------------------------------------------------
# 2) Train Random Forest
# -------------------------------------------------------------------
# clf = RandomForestClassifier(
#     # you can turn these back on as you tune:
#     n_estimators=201,
#     # criterion="gini",
#     max_depth=20,
#     min_samples_leaf=5,
#     min_samples_split=10,
#     max_features='sqrt',
#     class_weight="balanced",
#     n_jobs=-1,
#     random_state=9,
#     bootstrap=True,
#     oob_score=True,
#     # max_samples=0.33,
# )

import lightgbm as lgb

clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    n_jobs=-1,
)




print("Training...")
# clf.fit(X_train, y_train)
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50),
    ]
)
print("Training complete.")

# -------------------------------------------------------------------
# 3) Save model + feature columns together
# -------------------------------------------------------------------
model_out_path = "lgbm.gz"
to_save = {
    "model": clf,
    "features": list(X_train.columns),  # keep feature order for API input / eval
}

with gzip.open(model_out_path, "wb") as f:
    pickle.dump(to_save, f)

print(f"Saved gzipped model to {model_out_path}")
