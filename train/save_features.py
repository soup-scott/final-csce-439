from loader18 import load_ember_jsonl
import joblib  # or use pickle/gzip if you prefer

# Load from JSONL *once*
# X_train, y_train = load_ember_jsonl("../ember24set/train", show_progress=True)
# X_train, y_train = load_ember_jsonl("../ember24set/total", show_progress=True)
X_train, y_train = load_ember_jsonl("../ember18set", show_progress=True)
# X_test, y_test   = load_ember_jsonl("../ember24set/test",  show_progress=True)
# X_challenge, y_challenge = load_ember_jsonl("../ember24set/challenge",  show_progress=True)

# Save to disk in a fast binary format
joblib.dump(
    {
        "X_train": X_train,
        "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        # "X_challenge": X_challenge,
        # "y_challenge": y_challenge,
    },
    "ember18_train_optimize.joblib",
    compress=3,  # tweak compression if you like
)

print("Saved cached features to ember24_features.joblib")
