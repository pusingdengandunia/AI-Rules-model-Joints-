"""WESAD adapter (Path B): convert the synchronised wrist (Empatica E4) data into the project schema.

Only the wrist device is used for model inputs, because the product is a smartwatch.
The chest ECG is used for one thing: a reference HR to measure how good the
PPG-derived HR is (reported per subject, never fed to the model).

Per subject output (1 Hz, data/raw/wesad/WESAD_SX.csv):
  heart_rate, hrv_rmssd   derived from wrist BVP (64 Hz) beat detection
  accel_x/y/z             wrist ACC (32 Hz, 1/64 g) -> g, instantaneous sample each second
  eda, skin_temp          wrist EDA/TEMP (4 Hz) -> per-second mean
  gt_label                study-protocol condition (700 Hz) -> per-second majority
  gt_hr_ecg               chest-ECG HR, reference only

Timestamps: the wrist arrays in SX.pkl are an exact slice of the raw E4 ACC.csv, which
carries a Unix start time. The slice offset is found by exact matching, giving real
absolute UTC timestamps. If no match is found, a fixed anchor is used and reported.

SX.pkl files are Python pickles. They are loaded with an allow-list unpickler that only
reconstructs numpy arrays, so a tampered file cannot execute code.

Run: python -m src.sources.wesad --root D:/jointsHackathon/WESAD/WESAD
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import RAW_COLUMNS
from src.signals.beats import beats_to_features, clean_ibis, detect_ecg_peaks, detect_ppg_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "label": 700, "ECG": 700}
FALLBACK_ANCHOR = pd.Timestamp("2017-01-01T00:00:00Z")

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
}


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that only allows the classes needed to rebuild numpy arrays."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global in pickle: {module}.{name}")
        return super().find_class(module, name)


def load_pkl(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy 2 deprecation noise from Python 2 era pickles
        with open(path, "rb") as f:
            return SafeUnpickler(f, encoding="latin1").load()


def find_e4_start(zip_path: Path, wrist_acc: np.ndarray):
    """Absolute start time of the pkl wrist slice, by exact match against raw E4 ACC.csv."""
    if not zip_path.exists():
        return None, "e4_zip_missing"
    with zipfile.ZipFile(zip_path) as z:
        text = z.read("ACC.csv").decode()
    first = text.splitlines()[0]
    start_unix = float(first.split(",")[0])
    raw = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=2)
    probe = wrist_acc[:64]
    # Candidate offsets where the first sample matches, then confirm on 64 samples.
    for off in np.flatnonzero((raw[:, 0] == probe[0, 0]) & (raw[:, 1] == probe[0, 1]) & (raw[:, 2] == probe[0, 2])):
        if off + 64 <= len(raw) and np.array_equal(raw[off:off + 64], probe):
            return pd.Timestamp(start_unix + off / FS["ACC"], unit="s", tz="UTC"), "e4_exact_match"
    return None, "e4_no_match"


def _per_second(x: np.ndarray, fs: int, n: int, how: str = "mean") -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)[: n * fs].reshape(n, fs)
    return x.mean(axis=1) if how == "mean" else x[:, 0]


def _label_per_second(label: np.ndarray, n: int):
    lab = label[: n * FS["label"]].reshape(n, FS["label"]).astype(int)
    counts = np.stack([(lab == k).sum(axis=1) for k in range(8)], axis=1)
    major = counts.argmax(axis=1)
    return np.array([LABELS[k] for k in major], dtype=object), counts.max(axis=1) / FS["label"]


def _hr_hrv(peaks: np.ndarray, n: int):
    """HR from loosely cleaned beats (coverage), RMSSD from strictly cleaned beats (accuracy)."""
    t_loose, ibi_loose = clean_ibis(peaks)
    t_strict, ibi_strict = clean_ibis(peaks, strict=True)
    hr, _ = beats_to_features(n, t_loose, ibi_loose)
    _, rmssd = beats_to_features(n, t_strict, ibi_strict)
    return hr, rmssd, t_loose


def convert_subject(subject_dir: Path) -> tuple[pd.DataFrame, dict]:
    sid = subject_dir.name
    data = load_pkl(subject_dir / f"{sid}.pkl")
    wrist, chest = data["signal"]["wrist"], data["signal"]["chest"]
    n = int(min(len(wrist["ACC"]) // FS["ACC"], len(wrist["BVP"]) // FS["BVP"],
                len(wrist["EDA"]) // FS["EDA"], len(data["label"]) // FS["label"]))

    start, ts_source = find_e4_start(subject_dir / f"{sid}_E4_Data.zip", wrist["ACC"])
    if start is None:
        start = FALLBACK_ANCHOR

    ppg_peaks, sqi = detect_ppg_peaks(wrist["BVP"], FS["BVP"])
    hr, rmssd, beat_t = _hr_hrv(ppg_peaks, n)
    hr_ecg, rmssd_ecg, ecg_t = _hr_hrv(detect_ecg_peaks(chest["ECG"], FS["ECG"]), n)

    acc = np.asarray(wrist["ACC"], dtype=float)[: n * FS["ACC"]].reshape(n, FS["ACC"], 3)[:, 0, :] / 64.0
    eda = _per_second(wrist["EDA"], FS["EDA"], n)
    temp = _per_second(wrist["TEMP"], FS["TEMP"], n)
    label, purity = _label_per_second(np.asarray(data["label"]), n)
    del data, wrist, chest  # free the ~1 GB pickle before building the frame

    ts = start + pd.to_timedelta(np.arange(n), unit="s")
    pid = f"WESAD_{sid}"
    df = pd.DataFrame({
        "patient_id": pid,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "heart_rate": np.round(hr, 1),
        "session_id": f"{pid}_lab",
        "hrv_rmssd": np.round(rmssd, 1),
        "accel_x": np.round(acc[:, 0], 4),
        "accel_y": np.round(acc[:, 1], 4),
        "accel_z": np.round(acc[:, 2], 4),
        "eda": np.round(eda, 4),
        "skin_temp": np.round(temp, 2),
        "source": "empatica_e4",
        "received_at": None,  # offline dataset: arrival time unknown
        "gt_label": label,
        "gt_label_purity": np.round(purity, 3),
        "gt_hr_ecg": np.round(hr_ecg, 1),
        "gt_hrv_rmssd_ecg": np.round(rmssd_ecg, 1),
    })
    both = ~np.isnan(hr) & ~np.isnan(hr_ecg)
    both_hrv = ~np.isnan(rmssd) & ~np.isnan(rmssd_ecg)
    meta = dict(
        patient_id=pid, seconds=n, timestamp_source=ts_source, start=str(start),
        ppg_beats=int(len(beat_t)), ecg_beats=int(len(ecg_t)),
        ppg_sqi_median=round(float(np.median(sqi)), 3),
        hr_ppg_coverage=round(float(np.mean(~np.isnan(hr))), 3),
        hrv_ppg_coverage=round(float(np.mean(~np.isnan(rmssd))), 3),
        hr_ppg_vs_ecg_mae_bpm=round(float(np.mean(np.abs(hr[both] - hr_ecg[both]))), 2) if both.any() else None,
        hr_ppg_within_5bpm=round(float(np.mean(np.abs(hr[both] - hr_ecg[both]) <= 5)), 3) if both.any() else None,
        rmssd_ppg_median_ms=round(float(np.nanmedian(rmssd)), 1) if both_hrv.any() else None,
        rmssd_ecg_median_ms=round(float(np.nanmedian(rmssd_ecg)), 1) if both_hrv.any() else None,
        rmssd_ppg_vs_ecg_spearman=round(float(pd.Series(rmssd[both_hrv]).corr(pd.Series(rmssd_ecg[both_hrv]), method="spearman")), 3)
        if both_hrv.sum() > 10 else None,
    )
    return df, meta


def convert_all(root: Path, out: Path, subjects: list[str] | None = None) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted((d for d in root.iterdir() if d.is_dir() and d.name.startswith("S")), key=lambda d: int(d.name[1:]))
    if subjects:
        dirs = [d for d in dirs if d.name in subjects]
    metas = []
    for d in dirs:
        print(f"[wesad] {d.name} ...", flush=True)
        df, meta = convert_subject(d)
        df[RAW_COLUMNS + [c for c in df.columns if c.startswith("gt_")]].to_csv(out / f"{meta['patient_id']}.csv", index=False)
        metas.append(meta)
        print(f"[wesad] {d.name}: {meta}", flush=True)
    (out / "conversion_report.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    return metas


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert WESAD SX.pkl files into the project raw schema (1 Hz).")
    ap.add_argument("--root", type=Path, required=True, help="folder containing S2, S3, ... subfolders")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "wesad")
    ap.add_argument("--subjects", nargs="*", help="e.g. S2 S3 (default: all)")
    args = ap.parse_args()
    convert_all(args.root, args.out, args.subjects)


if __name__ == "__main__":
    main()
