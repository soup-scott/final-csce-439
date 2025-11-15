# eval_rf.py

import gzip
import pickle

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)


# -------------------------------------------------------------------
# 1) Load cached features
# -------------------------------------------------------------------
data = joblib.load("ember18_challenge_optimize.joblib")

# X_train = data["X_train"]
# y_train = data["y_train"]
# X_test = data["X_test"]
# y_test = data["y_test"]
X_challenge = data["X_challenge"]
y_challenge = data["y_challenge"]

print("Loaded cached features:")
# print("  Train:", X_train.shape)
# print("  Test:", X_test.shape)
print("  Challenge:", X_challenge.shape)

# -------------------------------------------------------------------
# 2) Load model + feature order
# -------------------------------------------------------------------
model_path = "RF_ember18_optimize.gz"
with gzip.open(model_path, "rb") as f:
    saved = pickle.load(f)

clf = saved["model"]
feature_order = saved["features"]

print(f"\nLoaded model from {model_path}")
print("Number of features expected by model:", len(feature_order))

# Ensure column order matches training
# X_test = X_test[feature_order]
X_challenge = X_challenge[feature_order]


# -------------------------------------------------------------------
# 3) Helper: evaluate one split (test or challenge)
# -------------------------------------------------------------------
def evaluate_split(name, X, y):
    """
    name: str, e.g. "Test" or "Challenge"
    X, y: features and labels
    """
    print(f"\n================= {name.upper()} SET EVALUATION =================")

    # Classification report at default threshold (0.5)
    probs = clf.predict_proba(X)[:, 1]
    y_pred_default = (probs >= 0.2076).astype(int)

    print("\nClassification report (thr=0.5):")
    print(classification_report(y, y_pred_default, digits=4))

    # ROC / PR metrics
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    print(f"ROC AUC ({name}): {roc_auc:.4f}")
    print(f"PR  AUC ({name}): {pr_auc:.4f}")

    # Confusion matrix at 0.5
    cm = confusion_matrix(y, y_pred_default)
    print("\nConfusion matrix (thr=0.5):")
    print(cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        specificity = float("nan")

    bal_acc = balanced_accuracy_score(y, y_pred_default)
    print(f"Specificity (TNR, thr=0.5): {specificity:.4f}")
    print(f"Balanced accuracy (thr=0.5): {bal_acc:.4f}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y, probs)

    # Threshold sweep: accuracy, TPR, TNR vs threshold
    accuracies = []
    tnrs = []
    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        acc_t = (y_pred_t == y).mean()
        accuracies.append(acc_t)

        cm_t = confusion_matrix(y, y_pred_t, labels=[0, 1])
        if cm_t.shape == (2, 2):
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            tnr_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else np.nan
        else:
            tnr_t = np.nan
        tnrs.append(tnr_t)

    accuracies = np.array(accuracies)
    tnrs = np.array(tnrs)

    # Best accuracy across thresholds
    best_idx_acc = int(np.argmax(accuracies))
    print(
        f"\nMax accuracy on {name} set: {accuracies[best_idx_acc]:.4f} "
        f"at threshold {thresholds[best_idx_acc]:.4f}"
    )

    # Best point with TPR >= 0.9 (if exists)
    idxs_tpr = np.where(tpr >= 0.9)[0]
    if len(idxs_tpr):
        best_idx_tpr = idxs_tpr[np.argmin(fpr[idxs_tpr])]
        chosen_thr = thresholds[best_idx_tpr]
        print(
            f"{name} set: Best threshold with TPR>=0.95: {chosen_thr:.4f}, "
            f"TPR={tpr[best_idx_tpr]:.4f}, FPR={fpr[best_idx_tpr]:.4f}"
        )
    else:
        print(f"{name} set: No threshold achieves TPR >= 0.95")

    # ---------------------- PLOTS ---------------------- #

    # ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} Set")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    roc_path = f"roc_{name.lower()}.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    # Threshold vs metrics plot (Accuracy, TPR, TNR)
    plt.figure()
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, tpr, label="TPR (Recall)")
    plt.plot(thresholds, tnrs, label="TNR (Specificity)")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title(f"Threshold vs Metrics - {name} Set")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    thr_path = f"threshold_metrics_{name.lower()}.png"
    plt.savefig(thr_path)
    plt.close()
    print(f"Saved threshold metrics plot to {thr_path}")

    # Score distribution plot
    plt.figure()
    plt.hist(
        probs[y == 0],
        bins=50,
        alpha=0.5,
        label="goodware",
        density=False,
    )
    plt.hist(
        probs[y == 1],
        bins=50,
        alpha=0.5,
        label="malware",
        density=False,
    )
    plt.xlabel("Predicted probability (malware)")
    plt.ylabel("Count")
    plt.title(f"Score Distribution - {name} Set")
    plt.legend()
    plt.tight_layout()
    hist_path = f"score_hist_{name.lower()}.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved score histogram to {hist_path}")


# -------------------------------------------------------------------
# 4) Run evaluation on TEST and CHALLENGE sets
# -------------------------------------------------------------------
# evaluate_split("Test", X_test, y_test)
evaluate_split(model_path, X_challenge, y_challenge)

# -------------------------------------------------------------------
# 5) Feature importance summary
# -------------------------------------------------------------------
importances = clf.feature_importances_
feat_imp = sorted(
    zip(feature_order, importances),
    key=lambda x: x[1],
    reverse=True,
)

print("\nTop features by importance:")
for name, val in feat_imp[:]:
    print(f"{name:30s} {val:.4f}")
