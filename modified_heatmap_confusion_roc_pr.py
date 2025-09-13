import os, time, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ========= Config =========
CSV_PATH = r"dataset_final.csv"
OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

def savefig_all(stem: str):
    pdf_path = OUTDIR / f"{stem}.pdf"
    svg_path = OUTDIR / f"{stem}.svg"
    png_path = OUTDIR / f"{stem}.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=400)
    plt.close()
    return pdf_path, svg_path, png_path

# ========= Load data =========
df = pd.read_csv(CSV_PATH)
label_col = "fault"
assert label_col in df.columns, f"Label column '{label_col}' not found. Got: {df.columns.tolist()}"
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols].values.astype(np.float32)
y = df[label_col].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= Models =========
models = {
    "sgd_logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000,
                              early_stopping=True, n_iter_no_change=10))
    ]),
    "linear_svc": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC())
    ]),
    "rbf_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, C=2.0, gamma="scale"))
    ]),
    "rf": RandomForestClassifier(n_estimators=150, max_depth=12,
                                 max_features="sqrt", random_state=42),
    "hgb": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.1,
                                          max_iter=120, random_state=42),
}

# ========= Train/evaluate =========
metrics = []
prob_cache, fitted = {}, {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        y_prob = None
    prob_cache[name] = y_prob

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    pr  = average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    # single-sample latency
    x1 = X_test[:1]
    _ = model.predict(x1)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model.predict(x1)
    latency_ms = (time.perf_counter() - t0) / N * 1000.0

    # model size
    tmp = OUTDIR / f"tmp_{name}.joblib"
    joblib.dump(model, tmp)
    size_mb = tmp.stat().st_size / (1024 * 1024)
    tmp.unlink(missing_ok=True)

    metrics.append(dict(model=name, accuracy=acc, f1=f1, pr_auc=pr,
                        auroc=auc, latency_ms=latency_ms, size_mb=size_mb))

metrics_df = pd.DataFrame(metrics).sort_values("f1", ascending=False)
metrics_df.to_csv(OUTDIR / "metrics_summary.csv", index=False)

best_name = metrics_df.iloc[0]["model"]
best_model = fitted[best_name]
print("Best model by F1:", best_name)
print(metrics_df)

# ========= Figures =========

# A) Sensor distributions
rep_cols = []
for base in ["current", "temp", "vibration"]:
    reps = [c for c in feature_cols if c.startswith(base)]
    if reps:
        rep_cols.append(reps[0])

if rep_cols:
    plt.figure(figsize=(10, 4))
    for i, col in enumerate(rep_cols[:3]):
        plt.subplot(1, len(rep_cols[:3]), i + 1)
        n_vals = df.loc[df[label_col] == 0, col]
        f_vals = df.loc[df[label_col] == 1, col]
        plt.hist(n_vals, bins=30, alpha=0.6, label="Normal", density=True)
        plt.hist(f_vals, bins=30, alpha=0.6, label="Fault", density=True)
        plt.title(col); plt.xlabel("Value"); plt.ylabel("Density")
        if i == 0: plt.legend()
    plt.suptitle("Sensor Value Distributions by Class (Normal vs Fault)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_all("fig_sensor_distributions")

# B) Feature correlation heatmap (with numbers)
corr_df = pd.DataFrame(X, columns=feature_cols).corr()
plt.figure(figsize=(7.5, 6.5))
im = plt.imshow(corr_df, interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=7)
plt.yticks(range(len(feature_cols)), feature_cols, fontsize=7)

for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        plt.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=6, color="black")

plt.colorbar(im, shrink=0.8)
plt.tight_layout()
savefig_all("Heatmap")

# C) Latency comparison
plt.figure(figsize=(8.5, 4.5))
plt.bar(metrics_df["model"].values, metrics_df["latency_ms"].values)
plt.ylabel("Latency (ms) @ batch=1")
plt.title("Inference Latency")
plt.grid(axis='y', linewidth=0.3)
plt.tight_layout()
savefig_all("fig_latency")

# D) ROC Curves (all models)
plt.figure(figsize=(7, 6))
for name, y_prob in prob_cache.items():
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=plt.gca())
plt.title("ROC Curves (All Models)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
savefig_all("fig_all_roc")

# E) Precision-Recall Curves (all models)
plt.figure(figsize=(7, 6))
for name, y_prob in prob_cache.items():
    if y_prob is not None:
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=plt.gca())
plt.title("Precision-Recall Curves (All Models)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
savefig_all("fig_all_pr")

# F) Confusion matrix (best model only)
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(4.6, 4.6))
plt.imshow(cm, interpolation='nearest', cmap="Blues")
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks([0, 1], ["Normal", "Fault"]); plt.yticks([0, 1], ["Normal", "Fault"])
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, f"{val}", ha="center", va="center", fontsize=10, color="black")
plt.colorbar(shrink=0.8)
plt.tight_layout()
savefig_all("fig_confusion_matrix")

# Save summary
summary = {
    "best_model": best_name,
    "metrics": metrics_df.to_dict(orient="records"),
    "figures": [p.name for p in sorted(OUTDIR.glob("*.pdf"))]
}
with open(OUTDIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFigures saved to:", OUTDIR.resolve())
for p in sorted(OUTDIR.glob("*.*")):
    if p.suffix.lower() in {".pdf", ".svg", ".png"}:
        print(" -", p.name)
print("\nMetrics table:", (OUTDIR / "metrics_summary.csv").resolve())
print("Summary JSON  :", (OUTDIR / "summary.json").resolve())
