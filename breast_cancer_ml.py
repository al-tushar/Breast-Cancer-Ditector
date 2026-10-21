import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Dataset ──────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: Malignant={sum(y==0)}, Benign={sum(y==1)}")

# ── 2. Split & Standardize ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 3. Train SVM ─────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm = CalibratedClassifierCV(svm_base, ensemble=False)
svm.fit(X_train, y_train)

y_pred_svm  = svm.predict(X_test)
y_prob_svm  = svm.predict_proba(X_test)[:, 1]
cv_svm      = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy')

# ── 4. Train Random Forest ───────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf   = rf.predict(X_test)
y_prob_rf   = rf.predict_proba(X_test)[:, 1]
cv_rf       = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

# ── 5. Print Results ─────────────────────────────────────────────
for name, y_pred, y_prob, cv_scores in [
    ("SVM",           y_pred_svm, y_prob_svm, cv_svm),
    ("Random Forest", y_pred_rf,  y_prob_rf,  cv_rf)]:

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  CV Acc    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 6. Plot Confusion Matrices ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, y_pred, title in zip(axes,
    [y_pred_svm, y_pred_rf],
    ["SVM", "Random Forest"]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("e:/AI/BreastCancerProject/confusion_matrices.png", dpi=150)
plt.show()

# ── 7. Plot ROC Curves ───────────────────────────────────────────
plt.figure(figsize=(6, 5))
for y_prob, label, color in [
    (y_prob_svm, f"SVM (AUC={roc_auc_score(y_test, y_prob_svm):.3f})", "steelblue"),
    (y_prob_rf,  f"RF  (AUC={roc_auc_score(y_test, y_prob_rf):.3f})",  "darkorange")]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=label, lw=2, color=color)
plt.plot([0,1],[0,1],'k--', lw=1, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — SVM vs Random Forest")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("e:/AI/BreastCancerProject/roc_curves.png", dpi=150)
plt.show()

# ── 8. Plot Feature Importances (RF only) ────────────────────────
importances = rf.feature_importances_
top10_idx   = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(8, 4))
plt.bar(range(10), importances[top10_idx], color="steelblue")
plt.xticks(range(10), [data.feature_names[i] for i in top10_idx],
           rotation=35, ha='right', fontsize=8)
plt.title("Top 10 Feature Importances — Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("e:/AI/BreastCancerProject/feature_importances.png", dpi=150)
plt.show()