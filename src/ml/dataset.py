"""Langkah 3 / spec Step 8 — WESAD training dataset for Target B (stress vs non-stress).

Decisions (details and numbers in reports/wesad_dataset_summary.md):
  - Windows: majority label in {baseline, stress, amusement, meditation} with purity >= 0.9,
    calibration windows (is_baseline_period) removed. All data_quality values are kept (V2):
    the "good only" variant (V1) leaves 4 stress windows from 3 subjects, too few to learn from.
  - Target: y = 1 for stress, 0 for the other three conditions.
  - Excluded features, because their *missingness* encodes the label (PPG fails while
    subjects talk and gesture during the stress task) rather than physiology:
      valid_sample_ratio, hrv_rmssd_mean, hrv_delta_pct (and the other *_ratio columns).
  - eda_delta and skin_temp_delta come from the feature table (src/features/engineer.py,
    apply_baseline), so training and online inference compute them with the same code.
  - Split must be by subject (groups = patient_id); windows overlap by 4 minutes.

Run: python -m src.ml.dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "features" / "wesad_features.csv"
OUT_PATH = ROOT / "data" / "ml" / "wesad_dataset.csv"

CLASSES = ["baseline", "stress", "amusement", "meditation"]
POSITIVE = "stress"
MIN_PURITY = 0.9

FEATURE_SETS = {
    # Wrist HR + movement. HR uses the seconds that passed the PPG quality gate.
    "hr": ["hr_mean", "hr_std", "hr_delta_pct", "accel_magnitude_std"],
    # HR plus electrodermal activity (also on the E4 wrist).
    "hr_eda": ["hr_mean", "hr_std", "hr_delta_pct", "accel_magnitude_std", "eda_mean", "eda_delta"],
    # No PPG at all: immune to the PPG-missingness leak.
    "eda": ["eda_mean", "eda_delta", "accel_magnitude_std"],
}

EXCLUDED = {
    "valid_sample_ratio": "PPG coverage collapses during stress (see mean valid_sample_ratio below): label leak",
    "hrv_rmssd_mean": "missing far more often in stress (see missingness table); wrist RMSSD not validated vs ECG",
    "hrv_delta_pct": "same as hrv_rmssd_mean",
    "hrv_valid_ratio": "missingness = label leak",
    "accel_valid_ratio": "not informative (always ~1)",
    "skin_temp_mean": "absolute level is person-specific",
    "skin_temp_delta": "drifts with time since session start (protocol-order confound); "
                       "LOSO ablation: F1 0.79 with vs 0.77 without (within 1 std), so no real gain",
    "staleness_s": "PPG-quality indicator, same leak as valid_sample_ratio",
    "gt_*": "ground truth / ECG reference, never a feature",
}

META = ["patient_id", "window_start", "window_end", "gt_label", "gt_label_purity", "data_quality"]


def build_wesad_dataset(feats: pd.DataFrame) -> pd.DataFrame:
    """Feature table -> training rows with y (1 = stress) and every candidate feature."""
    keep =(feats["gt_label"].isin(CLASSES) & (feats["gt_label_purity"] >= MIN_PURITY)
            & ~feats["is_baseline_period"].astype(bool))
    ds = feats.loc[keep].copy()
    ds["y"] = (ds["gt_label"] == POSITIVE).astype(int)
    all_features = sorted({f for fs in FEATURE_SETS.values() for f in fs})
    # minutes_since_start is NOT a model feature: train.py uses it to probe the protocol-order confound.
    start = feats.groupby("patient_id")["window_start"].transform("min")
    ds["minutes_since_start"] = (ds["window_start"] - start.loc[ds.index]).dt.total_seconds() / 60
    return ds[META + all_features + ["minutes_since_start", "y"]].reset_index(drop=True)


def load(path: Path = OUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["window_start", "window_end"])


# --- report -----------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows) + "\n"


def write_summary(feats: pd.DataFrame, ds: pd.DataFrame, path: Path) -> None:
    base = feats[feats["gt_label"].isin(CLASSES) & (feats["gt_label_purity"] >= MIN_PURITY)
                 & ~feats["is_baseline_period"].astype(bool)]
    v1 = base[base["data_quality"] == "good"]
    per_subject = pd.crosstab(ds["patient_id"], ds["gt_label"]).reindex(columns=CLASSES, fill_value=0)
    per_subject["stress_share"] = (per_subject["stress"] / per_subject.sum(axis=1)).round(2)
    per_subject = per_subject.reset_index()
    per_subject["_n"] = per_subject["patient_id"].str.extract(r"(\d+)$").astype(int)
    per_subject = per_subject.sort_values("_n").drop(columns="_n")

    cand = sorted({f for fs in FEATURE_SETS.values() for f in fs} | {"valid_sample_ratio", "hrv_rmssd_mean", "skin_temp_delta"})
    miss = base.groupby("gt_label")[cand].apply(lambda d: d.isna().mean()).T
    miss = miss.reindex(columns=CLASSES).round(2).reset_index().rename(columns={"index": "feature"})
    vsr = base.groupby("gt_label")["valid_sample_ratio"].mean().reindex(CLASSES).round(2)

    s = [
        "# Dataset WESAD untuk Target B (stres vs non-stres)\n",
        "> Eksperimen eksploratif: stres akut di lab pada 15 subjek sehat. Bukan episode PTSD.\n",
        "## Filter\n",
        f"1. Label window ∈ {CLASSES}; `transient` dan `ignore` dibuang.",
        f"2. `gt_label_purity` ≥ {MIN_PURITY} (label lain maksimal 30 detik dalam window 5 menit).",
        "3. Window kalibrasi baseline (`is_baseline_period=True`) dibuang, karena delta-nya ≈ 0 secara konstruksi.",
        "4. Target: `y = 1` untuk `stress`, `0` untuk baseline/amusement/meditation.\n",
        "## Keputusan varian\n",
        _md(pd.DataFrame([
            ("V1: hanya window `good`", len(v1), int((v1["gt_label"] == "stress").sum()),
             v1.loc[v1["gt_label"] == "stress", "patient_id"].nunique(), "**ditolak**: stress terlalu sedikit"),
            ("V2: semua window", len(ds), int(ds["y"].sum()), ds.loc[ds["y"] == 1, "patient_id"].nunique(),
             "**dipakai**, dengan fitur bebas leakage missingness"),
        ], columns=["varian", "window", "window stress", "subjek dgn stress", "keputusan"])),
        "\nKenapa V1 sangat kecil: HR wrist gagal terbaca saat subjek bicara dan bergerak di tugas stress (TSST), "
        "sehingga hampir semua window stress berstatus `insufficient_data`.\n",
        "## Distribusi kelas per subjek (V2)\n",
        _md(per_subject),
        f"\nTotal: {len(ds)} window, {int(ds['y'].sum())} stress ({ds['y'].mean():.0%}). "
        "Kelas positif tersedia di semua subjek, jadi leave-one-subject-out bisa dipakai.\n",
        "## Missingness per kelas (alasan pemilihan fitur)\n",
        "Proporsi window dengan nilai kosong. Perbedaan besar antar kelas berarti missingness ikut membawa label.\n",
        _md(miss),
        f"\nRata-rata `valid_sample_ratio`: " + ", ".join(f"{k} {v}" for k, v in vsr.items()) + ".\n",
        "## Feature set\n",
        _md(pd.DataFrame([(k, ", ".join(v)) for k, v in FEATURE_SETS.items()], columns=["set", "fitur"])),
        "\nFitur yang dikeluarkan:\n",
        _md(pd.DataFrame(list(EXCLUDED.items()), columns=["fitur", "alasan"])),
        "\n## Checklist audit (spec Step 8)\n",
        "| Cek | Status |\n|---|---|",
        "| Data dari sensor wearable | ya, wrist Empatica E4 saja |",
        "| Label ground truth jelas | ya, kondisi protokol (stres akut TSST), bukan PTSD |",
        "| Target sesuai klaim | ya, klaim = \"klasifikasi stres akut di lab\" |",
        "| Timestamp dan label selaras | ya, label 700 Hz sinkron dengan wrist di `SX.pkl` |",
        f"| Jumlah subjek/kelas memadai | 15 subjek, {int(ds['y'].sum())} window stress — kecil, evaluasi eksploratif |",
        "| Lisensi | sitasi wajib; anggap non-komersial |",
        "| Tidak ada fitur yang membocorkan label | missingness-leak dibuang; `gt_*` tidak dipakai |",
        "",
        "## Risiko yang tetap ada\n",
        "- Urutan protokol tetap (baseline selalu pertama, stress di waktu yang mirip). Waktu sejak mulai saja sudah "
        "memprediksi stress dengan baik (lihat probe di `reports/ml_eval_wesad.md`). `skin_temp_delta` dikeluarkan karena alasan ini.",
        "- HR saat stress dihitung hanya dari detik yang lolos gate SQI (momen paling tenang), sehingga cenderung under-estimate.",
        "- Window bertumpuk 4 menit: split wajib per subjek (`groups = patient_id`).",
        "- Baseline HR S10 dan S15 tidak valid, sehingga `hr_delta_pct` mereka kosong (kosong karena subjek, bukan karena label).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def main() -> None:
    feats = pd.read_csv(FEATURES_PATH, parse_dates=["window_start", "window_end"])
    ds = build_wesad_dataset(feats)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT_PATH, index=False)
    write_summary(feats, ds, ROOT / "reports" / "wesad_dataset_summary.md")
    print(f"{len(ds)} windows, {int(ds['y'].sum())} stress, {ds['patient_id'].nunique()} subjects -> {OUT_PATH}")


if __name__ == "__main__":
    main()
