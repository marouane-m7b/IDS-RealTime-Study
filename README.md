# CIC-IDS2017 — RF vs SVM Intrusion Detection Experiment

Brief summary of the experiments, key metrics, and how to reproduce them.

## Overview
- Goal: Compare Random Forest (RF) and Support Vector Machines (SVM) for intrusion
  detection on the CIC-IDS2017 dataset under a strict latency constraint (< 50 ms).
- Hypotheses: RF outperforms SVM at large scale (H1); hybrid RF→SVM feature-selection
  may help (H2); reduced-feature (host-based) settings explored (H3).

## Dataset
- CIC-IDS2017 CSVs (8 files) were used. Total samples used in the experiment: ~800,000.
- See [README_dataset.md](README_dataset.md) for the list of CSV files required.

## Methods
- Preprocessing: numeric features only, binary label (BENIGN vs ATTACK), standard scaling.
- Models evaluated:
  - `RF (H1)` — RandomForestClassifier on full feature set
  - `LinearSVM (H1)` — LinearSVC on full feature set
  - `Hybrid RF→SVM (H2)` — RF-based feature selection (top-K) then SVM (RBF)
  - `SVM-RBF top-10 (H3)` and `RF top-10 (H3)` — models trained on reduced feature sets

## Key Results (from results/results_summary.csv)

Model | PR-AUC | Recall | Precision | F1 | Latency (ms)
-----:|:------:|:------:|:---------:|:--:|:------------:
RF (H1) | 0.99966 | 0.99763 | 0.99621 | 0.99692 | 23.436
LinearSVM (H1) | 0.91775 | 0.96048 | 0.68683 | 0.80093 | 0.650
Hybrid RF→SVM (H2) | 0.84899 | 0.96691 | 0.61431 | 0.75129 | 0.816
SVM-RBF top-10 (H3) | 0.66438 | 0.93953 | 0.55255 | 0.69586 | 0.889
RF top-10 (H3) | 0.99650 | 0.98948 | 0.97289 | 0.98111 | 27.562

**Takeaways**
- All evaluated models meet the latency constraint (< 50 ms).
- RF (full and top-10) achieves the highest PR-AUC and F1, showing strong detection
  performance even with reduced features.
- LinearSVM is fast at inference but shows lower precision and PR-AUC on this task.

## Artifacts
- Models, plots, and CSV summary are saved under the `results/` folder:
  - [results/results_summary.csv](results/results_summary.csv)
  - [results/pr_curves.png](results/pr_curves.png)
  - [results/summary_comparison.png](results/summary_comparison.png)
  - [results/feature_importance.png](results/feature_importance.png)
  - Serialized models: `results/rf_h1.pkl`, `results/svm_h1.pkl`, etc.

## Reproduce
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the CIC-IDS2017 CSV files in the project root (see `README_dataset.md`).

3. Run the experiment script:

```bash
python intrusion_detection_experiment.py
```

Outputs will be written to the `results/` directory.

## Notes
- Large datasets are not tracked in the repo; download the CSVs separately as noted.
- For faster iteration, `N_SAMPLES` in `intrusion_detection_experiment.py` can be reduced.

---
Generated summary based on the experiment outputs in `results/results_summary.csv`.
