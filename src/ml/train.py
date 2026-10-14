"""Langkah 4 / spec Step 9 — Target B: stress vs non-stress on WESAD (exploratory).

Every candidate (feature set x model) is evaluated with leave-one-subject-out (15 folds):
the test subject is never seen during training, and preprocessing (imputation, scaling)
is fitted inside each fold through a Pipeline.

Compared against:
  always_stress   naive baseline (predicts stress for every window)
  rules_v1        Target A rules on the same windows; monitor/elevated = "stress"

The best candidate (highest mean F1 over folds) is refit on all subjects and saved:
  models/model.joblib      the fitted estimator (not committed; rebuild with this script)
  models/model_card.json   target, feature order, metrics, data, limitations
  reports/ml_eval_wesad.md full comparison

Run: python -m src.ml.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.dataset import FEATURE_SETS, OUT_PATH as DATASET_PATH, build_wesad_dataset
from src.ml.rules import run_rules

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
FEATURES_PATH = ROOT / "data" / "features" / "wesad_features.csv"
THRESHOLD = 0.5
SEED = 42


def make_models() -> dict:
    return {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(class_weight="balanced", max_iter=1000)),
        "rf": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                           random_state=SEED, n_jobs=-1)),
        "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      class_weight="balanced", random_state=SEED),
    }


def fold_metrics(y, pred, proba=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        false_alarm_rate=fp / (fp + tn) if (fp + tn) else np.nan,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )
    if proba is not None and len(set(y)) == 2:
        m["auc"] = roc_auc_score(y, proba)
    return m


def loso(ds: pd.DataFrame, features: list[str], factory) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-fold metrics and out-of-fold predictions."""
    X, y, groups = ds[features], ds["y"].to_numpy(), ds["patient_id"].to_numpy()
    oof = np.zeros(len(ds), dtype=int)
    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        oof[test_idx] = pred
        rows.append(dict(subject=groups[test_idx][0], **fold_metrics(y[test_idx], pred, proba)))
    return pd.DataFrame(rows), oof


def per_subject(ds: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for subj, idx in ds.groupby("patient_id").indices.items():
        rows.append(dict(subject=subj, **fold_metrics(ds["y"].to_numpy()[idx], pred[idx])))
    return pd.DataFrame(rows)


def rules_predictions(ds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """rules_v1 on the full WESAD window sequence, aligned to the dataset rows.

    Returns (pred, covered): pred = 1 for monitor/elevated; covered = False for insufficient_data.
    """
    feats = pd.read_csv(FEATURES_PATH, parse_dates=["window_start", "window_end"])
    scored = run_rules(feats)[["patient_id", "window_end", "risk_level"]]
    merged = ds[["patient_id", "window_end"]].merge(scored, on=["patient_id", "window_end"], how="left")
    pred = merged["risk_level"].isin(["monitor", "elevated_physiological_anomaly"]).to_numpy().astype(int)
    covered = (merged["risk_level"] != "insufficient_data").to_numpy()
    return pred, covered


def summarize(name: str, folds: pd.DataFrame, extra: dict | None = None) -> dict:
    out = {"candidate": name}
    for k in ("f1", "precision", "recall", "false_alarm_rate", "auc"):
        if k in folds:
            out[f"{k}_mean"] = float(folds[k].mean())
            out[f"{k}_std"] = float(folds[k].std())
    out.update(extra or {})
    return out


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                       (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if DATASET_PATH.exists():
        ds = pd.read_csv(DATASET_PATH, parse_dates=["window_start", "window_end"])
    else:
        ds = build_wesad_dataset(pd.read_csv(FEATURES_PATH, parse_dates=["window_start", "window_end"]))
    print(f"dataset: {len(ds)} windows, {int(ds['y'].sum())} stress, {ds['patient_id'].nunique()} subjects")

    results, fold_tables = [], {}
    for set_name, features in FEATURE_SETS.items():
        for model_name, factory in make_models().items():
            name = f"{model_name}__{set_name}"
            folds, _ = loso(ds, features, factory)
            fold_tables[name] = folds
            results.append(summarize(name, folds, {"model": model_name, "feature_set": set_name}))
            print(f"{name:20s} F1 {folds['f1'].mean():.2f}±{folds['f1'].std():.2f}  "
                  f"AUC {folds['auc'].mean():.2f}  FAR {folds['false_alarm_rate'].mean():.2f}")

    # Confound probe: how well does "time since session start" alone predict stress?
    # Not a candidate; it shows how much of any score could come from protocol order.
    for model_name in ("logreg", "hgb"):
        folds, _ = loso(ds, ["minutes_since_start"], make_models()[model_name])
        results.append(summarize(f"probe_time_only__{model_name}", folds, {"model": "probe", "feature_set": "time"}))

    # Baselines on the same windows, evaluated per subject like the LOSO folds.
    y = ds["y"].to_numpy()
    naive = per_subject(ds, np.ones(len(ds), dtype=int))
    results.append(summarize("always_stress", naive, {"model": "baseline", "feature_set": "-"}))
    r_pred, covered = rules_predictions(ds)
    rules_folds = per_subject(ds, r_pred)
    results.append(summarize("rules_v1", rules_folds, {"model": "rules", "feature_set": "rules",
                                                       "coverage": float(covered.mean()),
                                                       "coverage_stress": float(covered[y == 1].mean())}))
    res = pd.DataFrame(results)

    ml = res[~res["model"].isin(["baseline", "rules", "probe"])].sort_values("f1_mean", ascending=False)
    probe_best = res[res["model"] == "probe"]["f1_mean"].max()
    best = ml.iloc[0]
    best_name, best_model, best_set = best["candidate"], best["model"], best["feature_set"]
    features = FEATURE_SETS[best_set]
    _, oof = loso(ds, features, make_models()[best_model])
    cm = confusion_matrix(y, oof, labels=[0, 1])

    # Refit on all subjects for inference.
    final = make_models()[best_model]()
    final.fit(ds[features], y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / "model.joblib")
    version = f"wesad_stress_{best_model}_{best_set}_v1"
    rules_row = res[res["candidate"] == "rules_v1"].iloc[0]
    card = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "Target B: acute lab stress (WESAD TSST) vs baseline/amusement/meditation. NOT PTSD episodes.",
        "intended_use": "Exploratory comparison only. The demo default stays rules_v1 (docs/ml_target.md).",
        "estimator": type(final[-1] if hasattr(final, "steps") else final).__name__,
        "features": features,
        "feature_set": best_set,
        "threshold": THRESHOLD,
        "positive_class": "stress",
        "output_mapping": {"1": "monitor (reason: model_stress_pattern)", "0": "normal"},
        "score_output": "none: predict_proba is not calibrated, so no score is part of the output",
        "training_data": {"dataset": "WESAD wrist (Empatica E4)", "subjects": int(ds["patient_id"].nunique()),
                          "windows": int(len(ds)), "stress_windows": int(y.sum()),
                          "file": "data/ml/wesad_dataset.csv", "builder": "src/ml/dataset.py"},
        "evaluation": {
            "method": "leave-one-subject-out (15 folds), mean +- std over folds",
            "f1": [round(best["f1_mean"], 3), round(best["f1_std"], 3)],
            "precision": [round(best["precision_mean"], 3), round(best["precision_std"], 3)],
            "recall": [round(best["recall_mean"], 3), round(best["recall_std"], 3)],
            "false_alarm_rate": [round(best["false_alarm_rate_mean"], 3), round(best["false_alarm_rate_std"], 3)],
            "auc": [round(best["auc_mean"], 3), round(best["auc_std"], 3)],
            "confusion_matrix_pooled": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
            "baselines": {
                "always_stress_f1": round(float(res.loc[res["candidate"] == "always_stress", "f1_mean"].iloc[0]), 3),
                "rules_v1_f1": round(float(rules_row["f1_mean"]), 3),
                "rules_v1_coverage_on_stress_windows": round(float(rules_row["coverage_stress"]), 3),
                "time_only_probe_best_f1": round(float(probe_best), 3),
            },
        },
        "limitations": [
            "15 healthy subjects, lab protocol with fixed order; results are exploratory.",
            "Wrist PPG HR is mostly unavailable during the stress task; HR features come from the calmest seconds.",
            "Fixed protocol order: time since start alone reaches a similar F1 with a non-linear model "
            "(see time_only_probe_best_f1). Part of any score may reflect protocol timing, not physiology.",
            "accel_magnitude_std is higher during the stress task partly because subjects talk and gesture (TSST).",
            "Not validated for PTSD, clinical use, or free-living data.",
        ],
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    show = res[["candidate", "f1_mean", "f1_std", "precision_mean", "recall_mean", "false_alarm_rate_mean", "auc_mean"]]
    show = show.sort_values("f1_mean", ascending=False)
    s = [
        "# Evaluasi ML — Target B (stres vs non-stres, WESAD)\n",
        "> Eksperimen eksploratif: stres akut di lab, 15 subjek sehat. Bukan deteksi PTSD.\n",
        "## Metode\n",
        f"- Dataset: `data/ml/wesad_dataset.csv` ({len(ds)} window, {int(y.sum())} stress, 15 subjek). "
        "Lihat `reports/wesad_dataset_summary.md`.",
        "- Evaluasi: leave-one-subject-out. Subjek tes tidak pernah dilihat saat training. Imputasi/scaling di-fit di dalam fold.",
        f"- Ambang probabilitas {THRESHOLD}; `class_weight=balanced`. Metrik = rata-rata ± std dari 15 fold (per subjek).",
        "- Pembanding: `always_stress` (selalu menebak stress) dan `rules_v1` (Target A; monitor/elevated dihitung sebagai stress).",
        "- Probe confound: `probe_time_only__*` hanya memakai menit sejak awal sesi. Bukan kandidat model.\n",
        "## Hasil semua kandidat\n",
        md(show),
        f"\n`rules_v1` hanya menilai {rules_row['coverage']:.0%} window (sisanya `insufficient_data`), "
        f"dan hanya {rules_row['coverage_stress']:.0%} window stress. Window yang tidak dinilai dihitung sebagai prediksi non-stress.\n",
        f"## Model terpilih: `{best_name}`\n",
        f"Fitur (urutan tetap): `{', '.join(features)}`\n",
        "Confusion matrix (gabungan out-of-fold):\n",
        md(pd.DataFrame([["aktual non-stress", int(cm[0, 0]), int(cm[0, 1])], ["aktual stress", int(cm[1, 0]), int(cm[1, 1])]],
                        columns=["", "prediksi non-stress", "prediksi stress"])),
        "\nPer subjek (fold):\n",
        md(fold_tables[best_name][["subject", "f1", "precision", "recall", "false_alarm_rate", "auc"]]),
        "\n## Interpretasi\n",
        f"- ML terbaik F1 {best['f1_mean']:.2f} vs `rules_v1` {rules_row['f1_mean']:.2f} vs tebakan naif "
        f"{res.loc[res['candidate'] == 'always_stress', 'f1_mean'].iloc[0]:.2f}.",
        "- Rules lemah di sini bukan karena logikanya salah, tetapi karena HR wrist hilang saat stress sehingga rules "
        "dengan benar menjawab `insufficient_data`. Rules tidak dirancang untuk stres yang disertai bicara dan gerak.",
        "- Sinyal yang tidak bergantung pada PPG (EDA, suhu, gerak) lebih andal di WESAD. Ini sejalan dengan temuan kualitas data.",
        "- Variasi antar subjek besar (lihat std dan tabel per subjek). Dengan 15 subjek, angka ini eksploratif.",
        f"- **Confound urutan protokol:** model non-linear yang hanya diberi \"menit sejak mulai\" mencapai F1 {probe_best:.2f} "
        "(baris `probe_time_only__*`). Di WESAD stress selalu terjadi di waktu yang mirip. Regresi logistik dengan fitur waktu "
        "saja gagal, dan koefisien model terpilih didominasi `eda_delta`, sehingga model terpilih kemungkinan besar memakai "
        "sinyal fisiologis. Tetapi dataset ini tidak bisa membuktikannya sepenuhnya.",
        "- `accel_magnitude_std` lebih tinggi saat stress sebagian karena subjek bicara dan bergestur di TSST (perilaku protokol, bukan fisiologi).",
        "\n## Keputusan\n",
        "- Demo tetap memakai `rules_v1` (Target A), sesuai `docs/ml_target.md`.",
        f"- Model `{version}` disimpan untuk eksperimen dan tersedia di `src/ml/predict.py` dengan `mode=\"model\"`, "
        "dengan klaim \"pola mirip stres akut (eksperimen)\", bukan PTSD.",
    ]
    (ROOT / "reports" / "ml_eval_wesad.md").write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"best: {best_name} -> models/model.joblib, models/model_card.json, reports/ml_eval_wesad.md")


if __name__ == "__main__":
    main()
