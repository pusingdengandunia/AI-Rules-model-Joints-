"""Step 6 — run the whole data-engineering pipeline for one source.

    raw  ->  validated  ->  clean (+ gaps)  ->  window features (+ baselines)  ->  quality report

Usage:
  python -m src.pipeline --source synthetic --regenerate
  python -m src.pipeline --source wesad --wesad-root D:/jointsHackathon/WESAD/WESAD
  python -m src.pipeline --source all

Outputs (per source):
  data/validated/<source>_validated.csv   every record + quality_flag / quality_issues
  data/clean/<source>_clean.csv           usable records, invalid values nulled, sessions
  data/clean/<source>_gaps.csv            periods without records
  data/features/<source>_features.csv     one row per 5-min window (step 1 min)
  data/baselines/<source>_baselines.json  personal baselines used for *_delta_pct
  reports/data_quality_<source>.md        data quality and missing-data documentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.engineer import build_feature_table, save_baselines
from src.ingestion.validate import quality_summary, validate_sensor_data
from src.preprocessing.clean import clean_sensor_data, detect_gaps, interval_profile
from src.schema import MIN_VALID_RATIO, SCHEMA_VERSION, WINDOW_S, WINDOW_STEP_S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASELINE_MODE = {"synthetic": "calibration", "wesad": "label"}
TEXT_COLUMNS = {"patient_id": str, "timestamp": str, "received_at": str, "session_id": str, "source": str}


def load_raw(source: str) -> pd.DataFrame:
    folder = DATA / "raw" / source
    files = sorted(p for p in folder.glob("*.csv") if p.name != "events.csv")
    if not files:
        raise FileNotFoundError(f"no raw CSV files in {folder}")
    return pd.concat([pd.read_csv(p, dtype=TEXT_COLUMNS) for p in files], ignore_index=True)


def run(source: str) -> dict:
    raw = load_raw(source)
    validated = validate_sensor_data(raw)
    clean = clean_sensor_data(validated)
    gaps = detect_gaps(clean)
    profile = interval_profile(clean)
    features, baselines = build_feature_table(clean, BASELINE_MODE[source])

    for sub in ("validated", "clean", "features", "baselines"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    validated.to_csv(DATA / "validated" / f"{source}_validated.csv", index=False)
    clean.to_csv(DATA / "clean" / f"{source}_clean.csv", index=False)
    gaps.to_csv(DATA / "clean" / f"{source}_gaps.csv", index=False)
    features.to_csv(DATA / "features" / f"{source}_features.csv", index=False)
    save_baselines(baselines, DATA / "baselines" / f"{source}_baselines.json")

    report = build_report(source, validated, clean, gaps, profile, features, baselines)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"data_quality_{source}.md").write_text(report, encoding="utf-8")

    summary = dict(source=source, **quality_summary(validated), clean_records=len(clean),
                   gaps=len(gaps), windows=len(features),
                   good_windows=int((features["data_quality"] == "good").sum()))
    print(json.dumps(summary, indent=2, default=str))
    return summary


# --- report -------------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df.empty:
        return "_(kosong)_\n"
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return "" if v is None else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def build_report(source, validated, clean, gaps, profile, features, baselines) -> str:
    q = quality_summary(validated)
    s = []
    s.append(f"# Laporan kualitas data — {source}\n")
    s.append(f"Schema {SCHEMA_VERSION}. Window {WINDOW_S // 60} menit, langkah {WINDOW_STEP_S} s, "
             f"`good` jika `valid_sample_ratio` ≥ {MIN_VALID_RATIO}. Dibuat oleh `src/pipeline.py`.\n")
    if source == "synthetic":
        s.append("> Data sintetis. Hanya untuk menguji pipeline dan demo; bukan data klinis.\n")
    else:
        s.append("> WESAD: stres akut yang dipicu di lab pada subjek sehat. Label bukan episode PTSD.\n")

    s.append("## 1. Ringkasan\n")
    s.append(md_table(pd.DataFrame([
        ("Record masuk", q["records"]),
        ("Record usable (masuk clean series)", q["usable"]),
        ("Record dikeluarkan (duplicate / timestamp invalid / tanpa patient_id)", q["records"] - q["usable"]),
        ("Pasien", validated["patient_id"].nunique()),
        ("Sesi", clean["session_id"].nunique()),
        ("Gap terdeteksi", len(gaps)),
        ("Window total", len(features)),
        ("Window `good`", int((features["data_quality"] == "good").sum())),
        ("Window `insufficient_data`", int((features["data_quality"] == "insufficient_data").sum())),
    ], columns=["Metrik", "Nilai"])))

    s.append("## 2. Quality flag per record\n")
    s.append("`quality_flag` = issue dengan prioritas tertinggi. Satu record bisa punya beberapa issue.\n")
    s.append(md_table(pd.Series(q["flags"]).rename_axis("quality_flag").reset_index(name="records")))
    s.append("\nSemua issue:\n")
    s.append(md_table(pd.Series(q["issues"]).rename_axis("issue").reset_index(name="records")) if q["issues"] else "_(tidak ada)_\n")

    if "gt_injected" in validated.columns:
        inj = validated[validated["gt_injected"].notna() & (validated["gt_injected"] != "")]
        if len(inj):
            s.append("\n### Deteksi gangguan yang disuntikkan (skenario `bad_quality`)\n")
            s.append("Baris = gangguan yang disuntikkan simulator. Kolom = flag yang diberikan validator.\n")
            ct = pd.crosstab(inj["gt_injected"], inj["quality_flag"]).reset_index()
            s.append(md_table(ct))
            hit = sum(exp in issues.split(";") for exp, issues in zip(inj["gt_injected"], inj["quality_issues"]))
            s.append(f"\nTerdeteksi dengan issue yang sesuai: **{hit}/{len(inj)}**.\n")

    s.append("\n## 3. Missing data per sesi\n")
    s.append("`hr_missing_ratio` dihitung dari record yang ada. Periode tanpa record sama sekali terlihat di kolom `gaps`.\n")
    win = features.groupby("session_id").agg(
        windows=("data_quality", "size"),
        insufficient=("data_quality", lambda x: int((x == "insufficient_data").sum())),
        mean_valid_ratio=("valid_sample_ratio", "mean"),
        mean_hrv_ratio=("hrv_valid_ratio", "mean"),
    ).reset_index()
    prof = profile.merge(win, on="session_id", how="left")
    prof["duration_min"] = ((prof["end"] - prof["start"]).dt.total_seconds() / 60).round(1)
    s.append(md_table(prof[["session_id", "records", "duration_min", "nominal_interval_s", "gaps", "too_dense",
                            "hr_missing_ratio", "windows", "insufficient", "mean_valid_ratio", "mean_hrv_ratio"]]))

    s.append("\n## 4. Gap (periode tanpa record)\n")
    if gaps.empty:
        s.append("_(tidak ada gap)_\n")
    else:
        s.append(f"Total {len(gaps)} gap, durasi total {gaps['duration_s'].sum():.0f} s, "
                 f"terpanjang {gaps['duration_s'].max():.0f} s. Top 15:\n\n")
        s.append(md_table(gaps.sort_values("duration_s", ascending=False).head(15)[
            ["session_id", "gap_start", "gap_end", "duration_s"]]))

    s.append("\n## 5. Baseline pribadi\n")
    s.append(f"Metode `{BASELINE_MODE[source]}`. Median `hr_mean` / `hrv_rmssd_mean` dari window `good` dalam periode kalibrasi. "
             "Window periode kalibrasi ditandai `is_baseline_period=True`; keluarkan dari evaluasi model.\n\n")
    s.append(md_table(baselines[["patient_id", "hr_baseline", "hrv_baseline", "n_windows", "valid", "period_start", "period_end"]]))

    if "gt_label" in features.columns:
        s.append("\n## 6. Distribusi label WESAD per window\n")
        s.append("Label window = mayoritas label protokol per detik. Window dengan `gt_label_purity` < 1 berada di transisi antar kondisi.\n\n")
        lab = features.assign(pure=features["gt_label_purity"] == 1.0).groupby("gt_label").agg(
            windows=("pure", "size"), pure_windows=("pure", "sum"),
            good_windows=("data_quality", lambda x: int((x == "good").sum())),
            hr_mean=("hr_mean", "mean"), hrv_rmssd_mean=("hrv_rmssd_mean", "mean"),
            accel_magnitude_std=("accel_magnitude_std", "mean")).reset_index()
        s.append(md_table(lab))
        conv = DATA / "raw" / "wesad" / "conversion_report.json"
        if conv.exists():
            s.append("\n## 7. Akurasi HR/HRV wrist (PPG) vs ECG dada (referensi)\n")
            s.append("HR dan RMSSD dari BVP wrist dibandingkan dengan HR/RMSSD dari ECG dada pada detik yang sama.\n\n")
            c = pd.DataFrame(json.loads(conv.read_text()))
            s.append(md_table(c[["patient_id", "timestamp_source", "hr_ppg_coverage", "hr_ppg_vs_ecg_mae_bpm",
                                 "hr_ppg_within_5bpm", "hrv_ppg_coverage", "rmssd_ppg_median_ms",
                                 "rmssd_ecg_median_ms", "rmssd_ppg_vs_ecg_spearman"]]))
            s.append(f"\nRata-rata: MAE HR {c['hr_ppg_vs_ecg_mae_bpm'].mean():.1f} bpm, "
                     f"{c['hr_ppg_within_5bpm'].mean():.0%} detik dalam ±5 bpm, "
                     f"Spearman RMSSD PPG vs ECG {c['rmssd_ppg_vs_ecg_spearman'].mean():.2f}.\n")
            if "gt_hr_ecg" in clean.columns:
                err = (clean["heart_rate"] - pd.to_numeric(clean["gt_hr_ecg"], errors="coerce")).abs()
                by = clean.assign(err=err).groupby("gt_label").agg(
                    hr_coverage=("heart_rate", lambda x: x.notna().mean()),
                    hr_mae_bpm=("err", "mean"),
                    within_5bpm=("err", lambda e: (e.dropna() <= 5).mean()),
                ).reset_index()
                s.append("\nError HR per kondisi (dihitung pada detik yang punya HR wrist dan ECG):\n\n" + md_table(by))

    s.append("\n## Keterbatasan\n")
    s.append("- Batas plausibilitas di `src/schema.py` adalah batas teknis sensor, bukan rentang klinis.\n")
    s.append("- Missing data tidak diisi (tidak ada `fillna(0)` atau interpolasi). Window tanpa sampel cukup menjadi `insufficient_data`.\n")
    s.append("- Window bertumpuk 4 menit. Split train/test harus per pasien, bukan per window acak.\n")
    s.append("- Kolom `gt_*` hanya kunci evaluasi. Jangan dipakai sebagai fitur model.\n")
    if source == "wesad":
        st = features[features["gt_label"] == "stress"]
        s.append(f"- HR wrist (PPG) tidak terbaca saat bicara/gerak tangan (kondisi `stress`/TSST): hanya "
                 f"{int((st['data_quality'] == 'good').sum())} dari {len(st)} window `stress` yang `good`. "
                 "Gate SQI sengaja membuang HR yang tidak bisa dipercaya daripada mengeluarkan angka salah. "
                 "Klasifikasi stres berbasis HR wrist pada WESAD akan sangat bias ke window yang tenang.\n")
        s.append("- RMSSD dari PPG wrist secara sistematis lebih tinggi dari ECG dan korelasinya lemah. "
                 "Perlakukan fitur HRV WESAD sebagai tidak tervalidasi; `gt_hrv_rmssd_ecg` (dada) tersedia hanya untuk pembanding.\n")
        s.append("- Urutan protokol tetap (baseline selalu sebelum stress). Model bisa mempelajari waktu, bukan kondisi.\n")
        s.append("- 15 subjek sehat di lab. Hasil eksperimen bersifat eksploratif.\n")
    # Blank line between blocks so headings, quotes and tables render as markdown.
    return "\n".join(x.rstrip("\n") + "\n" for x in s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the PTSD Guardian data pipeline.")
    ap.add_argument("--source", choices=["synthetic", "wesad", "all"], default="all")
    ap.add_argument("--regenerate", action="store_true", help="re-run the simulator before the pipeline")
    ap.add_argument("--wesad-root", type=Path, help="convert WESAD SX.pkl files first (slow, one-time)")
    args = ap.parse_args()

    sources = ["synthetic", "wesad"] if args.source == "all" else [args.source]
    if "synthetic" in sources and (args.regenerate or not (DATA / "raw" / "synthetic").exists()):
        from src.simulator.generate import main as generate
        import sys
        sys.argv = ["generate", "--plot"]
        generate()
    if "wesad" in sources and args.wesad_root:
        from src.sources.wesad import convert_all
        convert_all(args.wesad_root, DATA / "raw" / "wesad")
    for src in sources:
        run(src)


if __name__ == "__main__":
    main()
