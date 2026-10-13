"""Beat-level signal processing shared by the simulator and the WESAD adapter.

HR and RMSSD are always derived from beat times the same way, whether the beats
come from a simulated IBI series, a wrist PPG (BVP) or a chest ECG.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

HR_WINDOW_S = 5
HRV_WINDOW_S = 60
HRV_MIN_DIFFS = 30        # fewer valid successive differences than this -> RMSSD is NaN
MAX_BEAT_GAP_MS = 2500    # beats further apart than this are not a valid successive pair
IBI_MIN_MS, IBI_MAX_MS = 333, 1500   # 40-180 bpm
IBI_MAX_REL_DEV = 0.25    # reject IBI deviating >25% from the local median (artifact)
SUCCESSIVE_MAX_REL_DIFF = 0.20  # reject IBI changing >20% from the previous IBI
# Calibrated on WESAD against chest-ECG HR (15 subjects): SQI>=0.5 keeps 50% of seconds with
# MAE 3.8 bpm and 3% errors >15 bpm, versus 95% / 10.0 bpm / 19% without the gate.
PPG_MIN_SQI = 0.5


def beats_to_features(n_seconds: int, times: np.ndarray, ibis: np.ndarray):
    """Per-second HR (bpm, 5 s window) and RMSSD (ms, 60 s window).

    times: beat times in seconds from session start; ibis: interval ending at each beat (ms).
    Value at index i summarises beats in the window ending at second i + 1.
    """
    sec = np.arange(1, n_seconds + 1, dtype=float)
    csum_ibi = np.concatenate([[0.0], np.cumsum(ibis)])
    lo = np.searchsorted(times, sec - HR_WINDOW_S, side="right")
    hi = np.searchsorted(times, sec, side="right")
    cnt = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = np.where(cnt >= 2, 60000.0 / ((csum_ibi[hi] - csum_ibi[lo]) / np.maximum(cnt, 1)), np.nan)

    if len(times) < 2:
        return hr, np.full(n_seconds, np.nan)
    d_times = times[1:]
    valid = (np.diff(times) * 1000.0) <= MAX_BEAT_GAP_MS
    d2 = np.where(valid, np.diff(ibis) ** 2, 0.0)
    csum_d2 = np.concatenate([[0.0], np.cumsum(d2)])
    csum_v = np.concatenate([[0], np.cumsum(valid.astype(int))])
    lo = np.searchsorted(d_times, sec - HRV_WINDOW_S, side="right")
    hi = np.searchsorted(d_times, sec, side="right")
    nv = csum_v[hi] - csum_v[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        rmssd = np.where(nv >= HRV_MIN_DIFFS, np.sqrt((csum_d2[hi] - csum_d2[lo]) / np.maximum(nv, 1)), np.nan)
    return hr, rmssd


def clean_ibis(peak_times: np.ndarray, strict: bool = False):
    """Turn detected peak times into (beat_times, ibis) with artifact beats removed.

    strict=True also applies the successive-difference rule. Use it for HRV (one bad peak
    corrupts two IBIs and inflates RMSSD); leave it off for HR, where it costs too much coverage.
    """
    if len(peak_times) < 3:
        return np.array([]), np.array([])
    ibis = np.diff(peak_times) * 1000.0
    times = peak_times[1:]
    ok = (ibis >= IBI_MIN_MS) & (ibis <= IBI_MAX_MS)
    # Local median over 9 beats, computed on plausible beats only.
    med = np.full(len(ibis), np.nan)
    idx = np.flatnonzero(ok)
    if len(idx):
        vals = ibis[idx]
        k = 4
        pad = np.pad(vals, k, mode="edge")
        local = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), axis=1)
        med[idx] = local
    ok &= np.abs(ibis - med) <= IBI_MAX_REL_DEV * med
    if strict:
        prev = np.concatenate([[np.nan], ibis[:-1]])
        ok &= ~(np.abs(ibis - prev) > SUCCESSIVE_MAX_REL_DIFF * prev)
    return times[ok], ibis[ok]


def _refine_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions by parabolic interpolation (reduces 1/fs timing jitter)."""
    p = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    a, b, c = x[p - 1], x[p], x[p + 1]
    denom = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(denom != 0, 0.5 * (a - c) / denom, 0.0)
    return p + np.clip(shift, -0.5, 0.5)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def local_periodicity(x: np.ndarray, fs: float, win_s: float = 8.0, hop_s: float = 2.0):
    """Per-sample (beat period in s, signal quality index 0..1) from sliding-window autocorrelation.

    SQI = height of the strongest autocorrelation peak within 40-180 bpm lags. A clean pulse
    is strongly periodic (SQI near 1); motion artifacts are not (SQI low).
    """
    win, hop = int(win_s * fs), int(hop_s * fs)
    lag_lo, lag_hi = int(IBI_MIN_MS / 1000 * fs), int(IBI_MAX_MS / 1000 * fs)
    centers, periods, sqis = [], [], []
    for i in range(0, max(len(x) - win, 1), hop):
        seg = x[i:i + win] - x[i:i + win].mean()
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        centers.append(i + win / 2)
        if ac[0] <= 0:
            periods.append(np.nan)
            sqis.append(0.0)
            continue
        ac = ac / ac[0]
        cand, _ = find_peaks(ac[lag_lo:lag_hi])
        if len(cand):
            best = cand[np.argmax(ac[lag_lo:lag_hi][cand])]
            periods.append((best + lag_lo) / fs)
            sqis.append(float(max(ac[lag_lo:lag_hi][best], 0.0)))
        else:
            periods.append(np.nan)
            sqis.append(0.0)
    if not centers:
        return np.full(len(x), 0.8), np.zeros(len(x))
    p = pd_ffill(np.asarray(periods), default=0.8)
    # Median-smooth over 5 windows so one bad window cannot dictate the period.
    p = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(p, 2, mode="edge"), 5), axis=1)
    idx = np.arange(len(x))
    return np.interp(idx, centers, p), np.interp(idx, centers, sqis)


def pd_ffill(a: np.ndarray, default: float) -> np.ndarray:
    """Forward-fill NaNs (then back-fill, then default) without pulling in pandas."""
    a = a.copy()
    ok = ~np.isnan(a)
    if not ok.any():
        return np.full_like(a, default)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = np.argmax(ok)
    a[:first] = a[first]
    return a


def detect_ppg_peaks(bvp: np.ndarray, fs: float = 64.0, min_sqi: float = PPG_MIN_SQI):
    """Systolic peak times (s) of a wrist PPG/BVP signal, before IBI cleaning, plus per-sample SQI.

    Candidate peaks closer than 65% of the local beat period to the previous accepted
    peak are treated as dicrotic notches / noise and dropped (the higher one is kept).
    Peaks where SQI < min_sqi are discarded: there the pulse is not visible (motion
    artifact) and any "HR" would be a guess, so the output must be missing instead.
    """
    x = _bandpass(np.asarray(bvp, dtype=float).ravel(), fs, 0.7, 3.0)
    period, sqi = local_periodicity(x, fs)
    cand, _ = find_peaks(x, distance=int(0.25 * fs))
    accepted: list[int] = []
    for p in cand:
        if accepted and (p - accepted[-1]) < 0.65 * period[p] * fs:
            if x[p] > x[accepted[-1]]:
                accepted[-1] = p
            continue
        accepted.append(p)
    peaks = np.asarray(accepted, dtype=int)
    peaks = peaks[sqi[peaks] >= min_sqi]
    return _refine_peaks(x, peaks) / fs, sqi


def detect_ecg_peaks(ecg: np.ndarray, fs: float = 700.0) -> np.ndarray:
    """R-peak times (s) of a chest ECG, used only as a reference to check PPG-derived HR/HRV quality."""
    x = _bandpass(np.asarray(ecg, dtype=float).ravel(), fs, 5.0, 20.0)
    energy = x ** 2
    win = int(0.08 * fs)
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(energy, distance=int(0.33 * fs), height=0.25 * np.percentile(energy, 99))
    # Energy peaks are smeared by the 80 ms smoother; snap to the max |ECG| within +-50 ms.
    half = int(0.05 * fs)
    ax = np.abs(x)
    snapped = np.array([p - half + np.argmax(ax[max(p - half, 0):p + half + 1]) if p >= half else p for p in peaks])
    return _refine_peaks(ax, snapped) / fs
