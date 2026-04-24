"""
=============================================================================
RESEARCH QUESTION:
Sous des contraintes de latence stricte (< 50 ms), quelle famille de modèles ML
(RF vs SVM) maximise le PR-AUC et le Recall lors de la détection d'intrusions ?

HYPOTHESES:
H1 - RF surpasse SVM sur grands volumes (CIC-IDS2017)
H2 - Hybride RF+SVM supérieur à chacun seul (feature selection RF → SVM classifier)
H3 - SVM supérieur dans des contextes spécifiques (features réduits / Host-based IDS)
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample

import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = Path(".")           # folder with all CSV files
LATENCY_THRESHOLD_MS = 50      # strict latency constraint
N_SAMPLES = 100_000            # rows to sample per file (set None for all)
RANDOM_STATE = 42
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_cicids2017(data_dir: Path, n_samples=None):
    """Load and merge CIC-IDS2017 CSV files."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"[DATA] Found {len(csv_files)} CSV files")
    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name} ...", end=" ")
        try:
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()
            if n_samples:
                df = df.sample(min(n_samples, len(df)), random_state=RANDOM_STATE)
            dfs.append(df)
            print(f"({len(df):,} rows)")
        except Exception as e:
            print(f"ERROR: {e}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"[DATA] Total rows: {len(data):,}")
    return data


def preprocess(data: pd.DataFrame):
    """Clean, encode labels, split features/target."""
    # Identify label column
    label_col = None
    for candidate in [" Label", "Label", "label"]:
        if candidate in data.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Label column not found")

    print(f"[PREP] Label column: '{label_col}'")
    print(f"[PREP] Class distribution:\n{data[label_col].value_counts()}\n")

    # Binary label: BENIGN vs ATTACK
    data["binary_label"] = (data[label_col].str.strip() != "BENIGN").astype(int)

    # Drop non-numeric / label cols
    drop_cols = [label_col, "binary_label", "Flow ID", " Source IP", " Destination IP",
                 " Timestamp", "Source IP", "Destination IP", "Timestamp"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = data["binary_label"]

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])

    # Replace inf / NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    print(f"[PREP] Features: {X.shape[1]}  |  Samples: {len(X):,}")
    print(f"[PREP] Attack ratio: {y.mean():.2%}\n")
    return X, y


# ─────────────────────────────────────────────
# 2. LATENCY MEASUREMENT HELPER
# ─────────────────────────────────────────────
def measure_inference_latency(model, X_test, n_runs=5):
    """
    Return mean per-sample inference latency in milliseconds.
    Use a single sample to simulate real-time IDS.
    """
    single = X_test.iloc[[0]]
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(single)
        latencies.append((time.perf_counter() - t0) * 1000)
    return np.mean(latencies)


# ─────────────────────────────────────────────
# 3. METRIC HELPER
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    # Probability or decision function for PR-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    pr_auc  = average_precision_score(y_test, y_score)
    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    latency = measure_inference_latency(model, X_test)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Latency  : {latency:.3f} ms  {'✓ < 50ms' if latency < 50 else '✗ > 50ms'}")

    return {
        "model": model_name,
        "PR-AUC": pr_auc,
        "Recall": recall,
        "Precision": prec,
        "F1": f1,
        "Latency_ms": latency,
        "meets_latency": latency < LATENCY_THRESHOLD_MS,
        "y_pred": y_pred,
        "y_score": y_score,
    }


# ─────────────────────────────────────────────
# H1 — RF vs SVM on full large-scale dataset
# ─────────────────────────────────────────────
def hypothesis_1(X_train, X_test, y_train, y_test):
    print("\n" + "█"*60)
    print("  H1: RF surpasses SVM on large-scale CIC-IDS2017")
    print("█"*60)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # RF
    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    )
    t0 = time.time()
    rf.fit(X_train_sc, y_train)
    rf_train_time = time.time() - t0
    print(f"\n[H1] RF training time: {rf_train_time:.1f}s")

    # SVM (LinearSVC is much faster for large N)
    svm = LinearSVC(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    t0 = time.time()
    svm.fit(X_train_sc, y_train)
    svm_train_time = time.time() - t0
    print(f"[H1] SVM training time: {svm_train_time:.1f}s")

    rf_metrics  = evaluate(rf,  pd.DataFrame(X_test_sc),  y_test, "RF (H1)")
    svm_metrics = evaluate(svm, pd.DataFrame(X_test_sc),  y_test, "LinearSVM (H1)")

    rf_metrics["train_time_s"]  = rf_train_time
    svm_metrics["train_time_s"] = svm_train_time

    # Save models
    joblib.dump(rf,     OUTPUT_DIR / "rf_h1.pkl")
    joblib.dump(svm,    OUTPUT_DIR / "svm_h1.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "scaler_h1.pkl")

    return rf_metrics, svm_metrics, scaler


# ─────────────────────────────────────────────
# H2 — Hybrid RF (feature selection) + SVM (classifier)
# ─────────────────────────────────────────────
def hypothesis_2(X_train, X_test, y_train, y_test, scaler, top_k=14):
    print("\n" + "█"*60)
    print(f"  H2: Hybrid RF→SVM (top-{top_k} features)")
    print("█"*60)

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Step 1: RF for feature importance
    rf_selector = RandomForestClassifier(
        n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf_selector.fit(X_train_sc, y_train)
    importances = rf_selector.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_k]

    print(f"[H2] Top-{top_k} feature indices: {top_indices.tolist()}")

    X_train_top = X_train_sc[:, top_indices]
    X_test_top  = X_test_sc[:, top_indices]

    # Step 2: SVM on selected features
    svm_hybrid = SVC(
        C=1.0, kernel="rbf", probability=True,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    # SVC with RBF on 14 features is tractable
    max_train = min(30_000, len(X_train_top))  # cap for speed
    idx = np.random.choice(len(X_train_top), max_train, replace=False)
    t0 = time.time()
    svm_hybrid.fit(X_train_top[idx], y_train.iloc[idx])
    print(f"[H2] Hybrid SVM training time: {time.time()-t0:.1f}s")

    hybrid_metrics = evaluate(
        svm_hybrid, pd.DataFrame(X_test_top), y_test, "Hybrid RF→SVM (H2)"
    )
    joblib.dump(svm_hybrid, OUTPUT_DIR / "svm_hybrid_h2.pkl")

    return hybrid_metrics, top_indices


# ─────────────────────────────────────────────
# H3 — SVM vs RF on reduced feature set (Host-based IDS simulation)
# ─────────────────────────────────────────────
def hypothesis_3(X_train, X_test, y_train, y_test, scaler, n_features=10):
    print("\n" + "█"*60)
    print(f"  H3: SVM vs RF on reduced features ({n_features}) — Host-based IDS")
    print("█"*60)

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Use only the first n_features (simulating host-based reduced feature set)
    X_train_red = X_train_sc[:, :n_features]
    X_test_red  = X_test_sc[:, :n_features]

    # SVM with RBF on small feature set
    svm_h3 = SVC(
        C=10.0, kernel="rbf", probability=True,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    max_train = min(30_000, len(X_train_red))
    idx = np.random.choice(len(X_train_red), max_train, replace=False)
    svm_h3.fit(X_train_red[idx], y_train.iloc[idx])
    svm_h3_metrics = evaluate(
        svm_h3, pd.DataFrame(X_test_red), y_test, f"SVM-RBF top-{n_features} (H3)"
    )

    # RF on same reduced set
    rf_h3 = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf_h3.fit(X_train_red, y_train)
    rf_h3_metrics = evaluate(
        rf_h3, pd.DataFrame(X_test_red), y_test, f"RF top-{n_features} (H3)"
    )

    joblib.dump(svm_h3, OUTPUT_DIR / "svm_h3.pkl")
    joblib.dump(rf_h3,  OUTPUT_DIR / "rf_h3.pkl")

    return svm_h3_metrics, rf_h3_metrics


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_pr_curves(results, X_test_dict, y_test):
    """Plot Precision-Recall curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Precision-Recall Curves per Hypothesis", fontsize=14, fontweight="bold")

    hypothesis_groups = [
        ("H1: Large-Scale RF vs SVM", ["RF (H1)", "LinearSVM (H1)"]),
        ("H2: Hybrid Approach",        ["Hybrid RF→SVM (H2)"]),
        ("H3: Reduced Features",       ["SVM-RBF top-10 (H3)", "RF top-10 (H3)"]),
    ]

    result_map = {r["model"]: r for r in results}

    for ax, (title, model_names) in zip(axes, hypothesis_groups):
        for name in model_names:
            if name not in result_map:
                continue
            r = result_map[name]
            y_score = r["y_score"]
            prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_score)
            ax.plot(rec_vals, prec_vals, lw=2,
                    label=f"{name} (AUC={r['PR-AUC']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved pr_curves.png")


def plot_summary_bar(results):
    """Bar chart comparing PR-AUC, Recall, and Latency."""
    df = pd.DataFrame(results)[["model", "PR-AUC", "Recall", "F1", "Latency_ms"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold")

    metrics = ["PR-AUC", "Recall", "Latency_ms"]
    colors  = ["steelblue", "seagreen", "tomato"]
    ylabels = ["PR-AUC", "Recall", "Latency (ms)"]

    for ax, metric, color, ylabel in zip(axes, metrics, colors, ylabels):
        bars = ax.bar(df["model"], df[metric], color=color, alpha=0.8, edgecolor="black")
        if metric == "Latency_ms":
            ax.axhline(LATENCY_THRESHOLD_MS, color="red", linestyle="--",
                       linewidth=1.5, label=f"Threshold {LATENCY_THRESHOLD_MS}ms")
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(f"Comparison: {ylabel}")
        ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved summary_comparison.png")


def plot_feature_importance(rf_model, feature_names, top_k=20):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_k), importances[indices], color="steelblue", alpha=0.8)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"RF Feature Importances — Top {top_k}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved feature_importance.png")


def save_results_csv(results):
    df = pd.DataFrame(results).drop(columns=["y_pred", "y_score"], errors="ignore")
    df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print(f"\n[SAVE] Results saved to {OUTPUT_DIR / 'results_summary.csv'}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  CIC-IDS2017 — RF vs SVM Intrusion Detection Experiment")
    print("="*60)

    # Load data
    data = load_cicids2017(DATA_DIR, n_samples=N_SAMPLES)
    X, y = preprocess(data)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[SPLIT] Train: {len(X_train):,}  Test: {len(X_test):,}")

    all_results = []

    # ── H1 ──────────────────────────────────
    rf_m, svm_m, scaler = hypothesis_1(X_train, X_test, y_train, y_test)
    all_results += [rf_m, svm_m]

    # ── H2 ──────────────────────────────────
    hybrid_m, top_idx = hypothesis_2(X_train, X_test, y_train, y_test, scaler, top_k=14)
    all_results.append(hybrid_m)

    # ── H3 ──────────────────────────────────
    svm_h3_m, rf_h3_m = hypothesis_3(X_train, X_test, y_train, y_test, scaler, n_features=10)
    all_results += [svm_h3_m, rf_h3_m]

    # ── Plots ───────────────────────────────
    plot_pr_curves(all_results, {}, y_test)
    plot_summary_bar(all_results)

    # Feature importance from H1 RF
    rf_h1 = joblib.load(OUTPUT_DIR / "rf_h1.pkl")
    plot_feature_importance(rf_h1, list(X.columns))

    # ── CSV summary ─────────────────────────
    save_results_csv(all_results)

    # ── Conclusion ──────────────────────────
    print("\n" + "="*60)
    print("  LATENCY CONSTRAINT SUMMARY (< 50 ms)")
    print("="*60)
    for r in all_results:
        status = "✓ VALID" if r["meets_latency"] else "✗ EXCEEDS"
        print(f"  {r['model']:<30}  {r['Latency_ms']:.2f} ms  [{status}]")

    print("\n" + "="*60)
    print("  PR-AUC RANKING (highest = best detection)")
    print("="*60)
    for r in sorted(all_results, key=lambda x: x["PR-AUC"], reverse=True):
        print(f"  {r['model']:<30}  PR-AUC={r['PR-AUC']:.4f}  Recall={r['Recall']:.4f}")

    print(f"\n[DONE] All outputs saved to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
