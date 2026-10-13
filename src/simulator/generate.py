"""Synthetic wearable time-series generator for the PTSD Guardian MVP (Path A).

Produces reproducible 1 Hz sensor records in the project schema (src/schema.py)
plus ground-truth columns (gt_*) for seven scenarios: normal, hr_elevation,
hrv_drop, high_activity, missing_data, bad_quality and a longer mixed_demo.

HR and HRV are not drawn independently: a beat-to-beat IBI series is simulated
from target HR/RMSSD curves, then HR (5 s window) and RMSSD (60 s window) are
derived from the beats with the same code used for real PPG data.

IMPORTANT: this data is synthetic. It is for testing the data pipeline, alert
logic and caregiver app. It is NOT clinical data and must not be used to train
or validate a PTSD episode detector.

Run: python -m src.simulator.generate [--seed 42] [--patients 3] [--plot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import RAW_COLUMNS, SCHEMA_VERSION
from src.signals.beats import beats_to_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Effect of each event type at full intensity (envelope == 1).
# hr: added bpm, rmssd: multiplicative factor, eda: added uS, temp: added degC,
# move: movement intensity in g, scr_rate: skin-conductance responses per second.
EFFECTS = {
    "light_activity": dict(hr=(8, 12), rmssd=0.85, eda=0.3, temp=0.0, move=(0.04, 0.08), scr_rate=0.0),
    "arousal": dict(hr=(25, 35), rmssd=0.55, eda=(2.0, 4.0), temp=-0.3, move=(0.005, 0.015), scr_rate=0.06),
    "hrv_drop": dict(hr=(5, 10), rmssd=0.45, eda=(0.8, 1.5), temp=-0.1, move=(0.0, 0.005), scr_rate=0.02),
    "activity": dict(hr=(45, 60), rmssd=0.35, eda=(1.0, 2.0), temp=0.4, move=(0.30, 0.50), scr_rate=0.0),
}

# What the alert system is expected to do while an event is at its peak.
EXPECTED = {
    "light_activity": ("none", 0),
    "arousal": ("alert", 1),
    "hrv_drop": ("warning", 1),
    "activity": ("none", 0),  # physiological arousal explained by movement: must NOT alert
}


@dataclass
class Event:
    type: str
    start_s: int
    ramp_s: int
    hold_s: int
    recover_s: int

    @property
    def end_s(self) -> int:
        return self.start_s + self.ramp_s + self.hold_s + self.recover_s


@dataclass
class Gap:
    type: str  # "disconnected" (no records at all) | "off_wrist" (records with null physiology)
    start_s: int
    duration_s: int


@dataclass
class Scenario:
    name: str
    description: str
    duration_s: int
    events: list[Event] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    dropout_rate: float = 0.0  # fraction of isolated 1 s HR losses
    artifacts: list[tuple[int, int]] = field(default_factory=list)  # (start_s, duration_s) motion spikes
    corrupt: bool = False  # inject transport/format problems (bad_quality)


def build_scenarios() -> list[Scenario]:
    E, G = Event, Gap
    return [
        Scenario(
            "normal",
            "Resting baseline with two short light-activity bouts. No alert expected.",
            3600,
            [E("light_activity", 600, 30, 90, 60), E("light_activity", 2400, 30, 120, 60)],
        ),
        Scenario(
            "hr_elevation",
            "HR rises ~30 bpm at rest (no movement) with HRV drop and EDA rise. Alert expected.",
            3600,
            [E("light_activity", 600, 30, 90, 60), E("arousal", 1800, 120, 300, 300)],
        ),
        Scenario(
            "hrv_drop",
            "RMSSD falls to ~45% of baseline with only a small HR rise, at rest. Warning expected.",
            3600,
            [E("hrv_drop", 1500, 180, 600, 300)],
        ),
        Scenario(
            "high_activity",
            "15 min exercise bout: HR +45-60 bpm, low HRV, high movement. No alert expected (confounder).",
            3600,
            [E("light_activity", 300, 30, 60, 60), E("activity", 1200, 60, 900, 300)],
        ),
        Scenario(
            "missing_data",
            "Normal physiology with BLE disconnect (no records), off-wrist period (null physiology), "
            "random HR dropouts and a motion-artifact HR spike. No alert expected.",
            3600,
            [E("light_activity", 600, 30, 90, 60)],
            gaps=[G("disconnected", 900, 120), G("off_wrist", 2100, 300)],
            dropout_rate=0.02,
            artifacts=[(3000, 8)],
        ),
        Scenario(
            "bad_quality",
            "Normal physiology delivered badly: late batch, re-sent duplicates, invalid and naive "
            "timestamps, missing patient_id, out-of-range and non-numeric values. No alert expected.",
            3600,
            [E("light_activity", 600, 30, 90, 60)],
            corrupt=True,
        ),
        Scenario(
            "mixed_demo",
            "2 h demo session chaining rest, exercise, arousal at rest, HRV drop, disconnect and off-wrist.",
            7200,
            [
                E("light_activity", 300, 30, 90, 60),
                E("activity", 900, 60, 600, 240),
                E("arousal", 2700, 120, 300, 300),
                E("hrv_drop", 4500, 180, 480, 300),
                E("light_activity", 6300, 30, 120, 60),
            ],
            gaps=[G("disconnected", 3900, 90), G("off_wrist", 5700, 240)],
            dropout_rate=0.01,
            artifacts=[(6800, 6)],
        ),
    ]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)


def envelope(n: int, ev: Event) -> np.ndarray:
    """Trapezoid 0 -> 1 -> 0 with smooth ramps, one value per second."""
    t = np.arange(n, dtype=float)
    env = np.zeros(n)
    a, b, c, d = ev.start_s, ev.start_s + ev.ramp_s, ev.start_s + ev.ramp_s + ev.hold_s, ev.end_s
    up = (t >= a) & (t < b)
    env[up] = _smoothstep((t[up] - a) / max(ev.ramp_s, 1))
    env[(t >= b) & (t < c)] = 1.0
    down = (t >= c) & (t < d)
    env[down] = 1 - _smoothstep((t[down] - c) / max(ev.recover_s, 1))
    return env


def _draw(rng: np.random.Generator, v) -> float:
    return float(rng.uniform(*v)) if isinstance(v, tuple) else float(v)


def _slow_wander(rng: np.random.Generator, n: int, sd: float, tau_s: float) -> np.ndarray:
    """AR(1) process with time constant tau_s and stationary std sd."""
    phi = np.exp(-1 / tau_s)
    noise = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _lag(x: np.ndarray, tau_s: float) -> np.ndarray:
    """First-order lag, used for slow responses such as skin temperature."""
    alpha = 1 - np.exp(-1 / tau_s)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def simulate_beats(rng: np.random.Generator, hr_target: np.ndarray, rmssd_target: np.ndarray):
    """Beat times (s) and IBIs (ms). IBI = local mean + white noise, so RMSSD ~= sigma * sqrt(2)."""
    n = len(hr_target)
    times, ibis = [], []
    t = 0.0
    while t < n:
        i = int(t)
        ibi = 60000.0 / hr_target[i] + rng.normal(0, rmssd_target[i] / np.sqrt(2))
        ibi = float(np.clip(ibi, 300, 2000))
        t += ibi / 1000.0
        times.append(t)
        ibis.append(ibi)
    return np.array(times), np.array(ibis)


def scr_train(rng: np.random.Generator, rate: np.ndarray) -> np.ndarray:
    """Skin-conductance responses: Poisson onsets, fast rise, ~8 s exponential decay."""
    n = len(rate)
    onsets = rng.random(n) < rate
    k = np.arange(40, dtype=float)
    kernel = (1 - np.exp(-k / 1.5)) * np.exp(-k / 8.0)
    kernel /= kernel.max()
    amps = np.where(onsets, rng.uniform(0.3, 1.0, n), 0.0)
    return np.convolve(amps, kernel)[:n]


def simulate_accel(rng: np.random.Generator, move: np.ndarray) -> np.ndarray:
    """Instantaneous 1 Hz accelerometer samples (g): wrist orientation x gravity + movement + noise."""
    n = len(move)
    base = rng.normal(0, 1, 3)
    orient = base / np.linalg.norm(base) + np.column_stack([_slow_wander(rng, n, 0.15, 600) for _ in range(3)])
    orient /= np.linalg.norm(orient, axis=1, keepdims=True)
    dyn = rng.normal(0, 1, (n, 3)) * (1.2 * move)[:, None]
    return orient + dyn + rng.normal(0, 0.01, (n, 3))


def generate_session(sc: Scenario, patient_id: str, rng: np.random.Generator, start: datetime):
    n = sc.duration_s
    session_id = f"{patient_id}_{sc.name}"

    # Patient baseline: varies between patients, fixed within a session.
    hr0 = rng.uniform(62, 74)
    rmssd0 = rng.uniform(35, 55)
    eda0 = rng.uniform(1.0, 3.0)
    temp0 = rng.uniform(33.2, 34.0)

    hr_t = hr0 + _slow_wander(rng, n, 2.0, 120)
    rmssd_f = np.ones(n)
    eda = eda0 + _slow_wander(rng, n, 0.15, 300)
    temp_drive = np.zeros(n)
    move = np.abs(rng.normal(0.005, 0.003, n))
    scr_rate = np.full(n, 0.003)

    state = np.array(["baseline"] * n, dtype=object)
    alert_level = np.array(["none"] * n, dtype=object)
    expected = np.zeros(n, dtype=int)
    events_out = []

    for ev in sc.events:
        env = envelope(n, ev)
        eff = EFFECTS[ev.type]
        d_hr, d_eda, d_move = _draw(rng, eff["hr"]), _draw(rng, eff["eda"]), _draw(rng, eff["move"])
        hr_t += d_hr * env
        rmssd_f *= 1 - (1 - eff["rmssd"]) * env
        eda += d_eda * np.roll(env, 3) * (np.arange(n) >= 3)  # EDA reacts with a few seconds of delay
        temp_drive += eff["temp"] * env
        move += np.abs(rng.normal(d_move, 0.12 * d_move + 1e-4, n)) * env
        scr_rate += eff["scr_rate"] * env

        peak_end = ev.start_s + ev.ramp_s + ev.hold_s
        level, exp = EXPECTED[ev.type]
        state[ev.start_s:peak_end] = ev.type
        state[peak_end:ev.end_s] = "recovery"
        alert_level[ev.start_s:peak_end] = level
        expected[ev.start_s:peak_end] = exp
        events_out.append(dict(
            patient_id=patient_id, session_id=session_id, scenario=sc.name, kind="event", type=ev.type,
            start_s=ev.start_s, peak_start_s=ev.start_s + ev.ramp_s, peak_end_s=peak_end, end_s=ev.end_s,
            expected_alert=exp, alert_level=level, hr_delta_bpm=round(d_hr, 1), rmssd_factor=eff["rmssd"],
        ))

    rmssd_t = np.clip(rmssd0 * rmssd_f + _slow_wander(rng, n, 3.0, 180), 8, None)
    times, ibis = simulate_beats(rng, hr_t, rmssd_t)
    accel = simulate_accel(rng, move)

    sensor_status = np.array(["ok"] * n, dtype=object)
    gap_mask = np.zeros(n, dtype=bool)
    for g in sc.gaps:
        sl = slice(g.start_s, g.start_s + g.duration_s)
        sensor_status[sl] = g.type
        gap_mask[sl] = True
        state[sl] = "missing"
        alert_level[sl] = "none"
        expected[sl] = 0
        events_out.append(dict(
            patient_id=patient_id, session_id=session_id, scenario=sc.name, kind="gap", type=g.type,
            start_s=g.start_s, end_s=g.start_s + g.duration_s, expected_alert=0, alert_level="none",
        ))

    # Beats during a gap never reach the pipeline (no skin contact / no link).
    keep = ~gap_mask[np.minimum(times.astype(int), n - 1)]
    hr, rmssd = beats_to_features(n, times[keep], ibis[keep])

    eda = eda + scr_train(rng, scr_rate) + rng.normal(0, 0.02, n)
    temp = temp0 + _lag(temp_drive, 240) + _slow_wander(rng, n, 0.05, 600) + rng.normal(0, 0.02, n)

    for g in sc.gaps:
        if g.type != "off_wrist":
            continue
        sl = slice(g.start_s, g.start_s + g.duration_s)
        k = np.arange(g.duration_s)
        hr[sl] = rmssd[sl] = eda[sl] = np.nan
        temp[sl] = 27.0 + (temp[g.start_s - 1] - 27.0) * np.exp(-k / 90.0)
        accel[sl] = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.003, (g.duration_s, 3))  # lying on a table
        end = g.start_s + g.duration_s
        k = np.arange(min(300, n - end))  # skin temperature re-equilibrates after re-attach
        temp[end:end + len(k)] -= (temp[end:end + len(k)] - 27.0) * np.exp(-k / 90.0)

    if sc.dropout_rate > 0:
        drop = (rng.random(n) < sc.dropout_rate) & ~gap_mask
        hr[drop] = rmssd[drop] = np.nan
        sensor_status[drop] = "dropout"

    for a_start, a_dur in sc.artifacts:
        sl = slice(a_start, a_start + a_dur)
        hr[sl] = rng.uniform(175, 200, a_dur)
        accel[sl] += rng.normal(0, 0.8, (a_dur, 3))
        state[sl] = "artifact"
        events_out.append(dict(
            patient_id=patient_id, session_id=session_id, scenario=sc.name, kind="artifact",
            type="motion_artifact", start_s=a_start, end_s=a_start + a_dur, expected_alert=0, alert_level="none",
        ))

    ts = np.array([start + timedelta(seconds=i) for i in range(n)])
    # BLE delivers in order: arrival = measurement + jittered latency, never before the previous arrival.
    arrival_s = np.maximum.accumulate(np.arange(n) + 1.0 + rng.exponential(1.0, n))
    latency = arrival_s - np.arange(n)
    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": [t.strftime(TS_FORMAT) for t in ts],
        "heart_rate": np.round(hr),
        "session_id": session_id,
        "hrv_rmssd": np.round(rmssd, 1),
        "accel_x": np.round(accel[:, 0], 4),
        "accel_y": np.round(accel[:, 1], 4),
        "accel_z": np.round(accel[:, 2], 4),
        "eda": np.round(eda, 3),
        "skin_temp": np.round(temp, 2),
        "source": "simulator",
        "received_at": [(t + timedelta(seconds=float(l))).strftime(TS_FORMAT) for t, l in zip(ts, latency)],
        "gt_scenario": sc.name,
        "gt_t_s": np.arange(n),
        "gt_state": state,
        "gt_alert_level": alert_level,
        "gt_expected_alert": expected,
        "gt_sensor_status": sensor_status,
        "gt_injected": "",
    })
    # A disconnected device sends nothing: drop those rows entirely.
    df = df[~np.isin(sensor_status, ["disconnected"])].reset_index(drop=True)
    if sc.corrupt:
        df = inject_quality_problems(df, rng)

    for e in events_out:
        e["start_time"] = (start + timedelta(seconds=e["start_s"])).strftime(TS_FORMAT)
        e["end_time"] = (start + timedelta(seconds=e["end_s"])).strftime(TS_FORMAT)
    return df, events_out


def inject_quality_problems(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Transport/format faults for the bad_quality scenario. gt_injected names the expected quality_flag."""
    df = df.copy()
    df["heart_rate"] = df["heart_rate"].astype(object)
    recv = pd.to_datetime(df["received_at"], utc=True)
    by_t = {int(t): i for i, t in enumerate(df["gt_t_s"])}

    # Late batch: 60 s buffered on the phone and delivered 10 minutes later.
    late = [by_t[t] for t in range(1200, 1260)]
    recv.iloc[late] += pd.Timedelta(seconds=600)
    df.loc[late, "gt_injected"] = "late"

    # Re-sent batch: identical copies arrive 30 s after the originals (originals become duplicates).
    resend = [by_t[t] for t in range(2000, 2030)]
    copies = df.loc[resend].copy()
    copies["gt_injected"] = ""
    df.loc[resend, "gt_injected"] = "duplicate"
    # Conflicting re-send: same timestamp, different value; the later record wins.
    conflict = [by_t[t] for t in range(2500, 2503)]
    c2 = df.loc[conflict].copy()
    c2["heart_rate"] = [float(v) + 5 for v in c2["heart_rate"]]
    c2["gt_injected"] = ""
    df.loc[conflict, "gt_injected"] = "duplicate"

    picks = iter(rng.choice([by_t[t] for t in range(2700, 3500)], size=20, replace=False))

    for bad in ["not-a-timestamp", "", "1970-01-01T00:00:00Z", "2099-01-01T00:00:00Z", "2026-13-45T25:61:00Z"]:
        i = next(picks)
        df.loc[i, "timestamp"] = bad
        df.loc[i, "gt_injected"] = "invalid_timestamp"
    for _ in range(3):
        i = next(picks)
        df.loc[i, "patient_id"] = ""
        df.loc[i, "gt_injected"] = "missing_patient_id"
    for col, val in [("heart_rate", 0), ("heart_rate", 300), ("heart_rate", -5), ("hrv_rmssd", 900),
                     ("accel_x", 40.0), ("heart_rate", "abc")]:
        i = next(picks)
        df.loc[i, col] = val
        df.loc[i, "gt_injected"] = "invalid_value"
    for _ in range(3):  # no timezone designator: accepted as UTC but reported
        i = next(picks)
        df.loc[i, "timestamp"] = df.loc[i, "timestamp"].rstrip("Z")
        df.loc[i, "gt_injected"] = "naive_timezone"

    df["received_at"] = recv.dt.strftime(TS_FORMAT)
    copies["received_at"] = (pd.to_datetime(copies["received_at"], utc=True) + pd.Timedelta(seconds=30)).dt.strftime(TS_FORMAT)
    c2["received_at"] = (pd.to_datetime(c2["received_at"], utc=True) + pd.Timedelta(seconds=30)).dt.strftime(TS_FORMAT)
    out = pd.concat([df, copies, c2], ignore_index=True)
    # File order = arrival order, as a backend would log it.
    return out.sort_values("received_at", kind="stable").reset_index(drop=True)


def generate_all(seed: int, n_patients: int, scenarios: list[Scenario] | None = None):
    scenarios = scenarios or build_scenarios()
    base = datetime(2026, 1, 5, 2, 0, 0, tzinfo=timezone.utc)
    frames, events = [], []
    for p_idx in range(n_patients):
        patient_id = f"SIM{p_idx + 1:02d}"
        for sc_idx, sc in enumerate(scenarios):
            # Independent stream per (patient, scenario): adding a scenario does not change the others.
            rng = np.random.default_rng([seed, p_idx, sc_idx])
            start = base + timedelta(days=sc_idx, hours=p_idx)  # scenario 0 (normal) is each patient's first session
            df, ev = generate_session(sc, patient_id, rng, start)
            frames.append(df)
            events.extend(ev)
    return frames, pd.DataFrame(events), scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "synthetic")
    ap.add_argument("--plot", action="store_true", help="save a preview PNG per scenario (first patient)")
    args = ap.parse_args()

    frames, events, scenarios = generate_all(args.seed, args.patients)
    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for df in frames:
        name = f"{df.session_id.iloc[0]}.csv"
        path = args.out / name
        df[RAW_COLUMNS + [c for c in df.columns if c.startswith("gt_")]].to_csv(path, index=False)
        files.append(dict(file=name, rows=len(df), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    events.to_csv(args.out / "events.csv", index=False)

    manifest = dict(
        schema_version=SCHEMA_VERSION, seed=args.seed, patients=args.patients, sample_rate_hz=1,
        generated_by="src/simulator/generate.py", synthetic=True,
        warning="Synthetic data for pipeline/demo testing only. Not clinical data. Not for training PTSD detectors.",
        scenarios=[dict(name=s.name, description=s.description, duration_s=s.duration_s,
                        events=[asdict(e) for e in s.events], gaps=[asdict(g) for g in s.gaps],
                        dropout_rate=s.dropout_rate, artifacts=s.artifacts, corrupt=s.corrupt) for s in scenarios],
        files=files,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.plot:
        from src.simulator.plot import plot_session
        plot_dir = PROJECT_ROOT / "reports" / "plots" / "synthetic"
        plot_dir.mkdir(parents=True, exist_ok=True)
        first = frames[0].patient_id.iloc[0]
        for df in frames:
            if df.patient_id.iloc[0] == first:
                plot_session(df, plot_dir / f"{df.session_id.iloc[0]}.png")

    print(f"Wrote {len(files)} sessions ({sum(f['rows'] for f in files)} rows) to {args.out}")


if __name__ == "__main__":
    main()
