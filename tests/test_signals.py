"""Beat detection and the WESAD adapter's safety/shape guarantees."""

import io
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.schema import RAW_COLUMNS
from src.signals.beats import beats_to_features, clean_ibis, detect_ppg_peaks
from src.sources.wesad import SafeUnpickler

WESAD_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "wesad"


def synthetic_ppg(bpm: float, seconds: int = 120, fs: int = 64, noise: float = 0.05, seed: int = 0):
    t = np.arange(seconds * fs) / fs
    f = bpm / 60
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t + 0.8)  # systolic + dicrotic harmonic
    return x + np.random.default_rng(seed).normal(0, noise, len(t))


@pytest.mark.parametrize("bpm", [55, 72, 110])
def test_ppg_heart_rate_recovered(bpm):
    peaks, sqi = detect_ppg_peaks(synthetic_ppg(bpm))
    t, ibi = clean_ibis(peaks)
    hr, _ = beats_to_features(120, t, ibi)
    assert np.nanmedian(hr[10:-10]) == pytest.approx(bpm, abs=2)
    assert np.median(sqi) > 0.8


def test_noise_has_low_sqi_and_no_heart_rate():
    noise = np.random.default_rng(1).normal(0, 1, 120 * 64)
    peaks, sqi = detect_ppg_peaks(noise)
    assert np.median(sqi) < 0.5
    t, ibi = clean_ibis(peaks)
    hr, _ = beats_to_features(120, t, ibi)
    assert np.isnan(hr).mean() > 0.8  # missing, not a made-up number


def test_safe_unpickler_blocks_code_execution():
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    payload = pickle.dumps(Evil())
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        SafeUnpickler(io.BytesIO(payload)).load()


def test_safe_unpickler_loads_numpy_arrays():
    data = {"signal": {"wrist": {"BVP": np.arange(5.0)}}, "label": np.array([1, 2], dtype=np.int32)}
    out = SafeUnpickler(io.BytesIO(pickle.dumps(data, protocol=2))).load()
    assert np.array_equal(out["signal"]["wrist"]["BVP"], np.arange(5.0))


@pytest.mark.skipif(not any(WESAD_RAW.glob("WESAD_S*.csv")), reason="WESAD not converted")
def test_wesad_raw_files_follow_schema():
    for path in sorted(WESAD_RAW.glob("WESAD_S*.csv")):
        df = pd.read_csv(path, nrows=2000)
        assert list(df.columns[: len(RAW_COLUMNS)]) == RAW_COLUMNS, path.name
        ts = pd.to_datetime(df["timestamp"], utc=True)
        assert ts.is_monotonic_increasing and ts.diff().dropna().dt.total_seconds().eq(1).all()
        assert set(df["gt_label"]) <= {"transient", "baseline", "stress", "amusement", "meditation", "ignore"}
