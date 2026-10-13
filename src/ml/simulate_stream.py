"""Langkah 6 — offline dress rehearsal of the live flow, before any API exists.

For each raw session file, records "arrive" in received_at order. Every 60 s of simulated
clock the same chain the backend will run is executed on the records received so far:

    validate_sensor_data -> clean_sensor_data -> compute_window_features(window_end=now)
    -> Predictor.predict -> AlertDeduplicator

and the result is checked automatically against the simulator's ground truth
(data/raw/synthetic/events.csv), covering the three demo scenarios of the spec (Bagian 7):
  A stable           no alert in calm periods / sessions
  B abnormal         arousal -> elevated (one alert per episode), hrv_drop -> >= monitor,
                     exercise -> no alert
  C sensor offline   gap / off-wrist -> insufficient_data and a device-check notification

Run (after `python -m src.pipeline --source synthetic --regenerate`):
  python -m src.ml.simulate_stream                      # all SIM01 sessions, rules
  python -m src.ml.simulate_stream --mode model         # same with the WESAD model (experimental)
  python -m src.ml.simulate_stream --mode mix           # rules + model hints
  python -m src.ml.simulate_stream --sessions SIM01_mixed_demo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineer import compute_window_features, load_baselines
from src.ingestion.validate import validate_sensor_data
from src.ml.predict import Predictor
from src.ml.rules import SEVERITY
from src.preprocessing.clean import clean_sensor_data
from src.schema import GROUND_TRUTH_PREFIX, MAX_STALENESS_S, WINDOW_S

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "synthetic"
BASELINES = ROOT / "data" / "baselines" / "synthetic_baselines.json"
TEXT = {"patient_id": str, "timestamp": str, "received_at": str, "session_id": str, "source": str}
TICK_S = 60
BUFFER_S = 900          # records kept in memory (arrival time), enough for one window plus late data
ALERT_COOLDOWN_S = 1800  # same notification type is not repeated within 30 min
DEVICE_CHECK_TICKS = 2   # insufficient_data must last 2 ticks before telling the caregiver


@dataclass
class AlertDeduplicator:
    """Turns a stream of statuses into caregiver notifications without repeating them.

    alert         once when entering elevated; a new episode within ALERT_COOLDOWN_S of the
                  last alert is not re-notified (prevents flapping elevated <-> monitor)
    resolved      when leaving elevated
    device_check  once per outage: insufficient_data for DEVICE_CHECK_TICKS ticks in a row,
                  only after data has been good once (not while the first window fills up)
    """

    last_alert: pd.Timestamp | None = None
    prev_level: str | None = None
    insufficient_run: int = 0
    seen_good: bool = False

    def update(self, now: pd.Timestamp, level: str) -> list[dict]:
        out = []
        if level != "insufficient_data":
            self.seen_good = True
        self.insufficient_run = self.insufficient_run + 1 if level == "insufficient_data" else 0

        if level == "elevated_physiological_anomaly" and self.prev_level != level:
            if self.last_alert is None or (now - self.last_alert).total_seconds() >= ALERT_COOLDOWN_S:
                out.append({"time": now, "type": "alert", "message": "Pola fisiologis abnormal — perlu pemeriksaan caregiver"})
                self.last_alert = now
        if self.prev_level == "elevated_physiological_anomaly" and level in ("normal", "monitor"):
            out.append({"time": now, "type": "resolved", "message": "Pola fisiologis kembali stabil"})
        if self.seen_good and self.insufficient_run == DEVICE_CHECK_TICKS:
            out.append({"time": now, "type": "device_check", "message": "Data sensor tidak cukup — periksa perangkat"})
        self.prev_level = level
        return out


def simulate(raw: pd.DataFrame, baselines: dict, predictor: Predictor) -> tuple[pd.DataFrame, list[dict]]:
    """Stream one session through the full chain. Returns (timeline per tick, notifications)."""
    raw = raw.copy()
    gt_cols = [c for c in raw.columns if c.startswith(GROUND_TRUTH_PREFIX)]
    sent = raw.drop(columns=gt_cols)  # the system under test never sees ground truth
    arrival = pd.to_datetime(sent["received_at"], utc=True, errors="coerce")
    arrival = arrival.fillna(pd.to_datetime(sent["timestamp"], utc=True, errors="coerce", format="ISO8601"))
    order = np.argsort(arrival.to_numpy(), kind="stable")
    sent, arrival = sent.iloc[order].reset_index(drop=True), arrival.iloc[order].reset_index(drop=True)

    first = arrival.min().ceil(f"{TICK_S}s")
    last = arrival.max().ceil(f"{TICK_S}s")  # stop with the recording, not after it
    pid = sent["patient_id"].dropna().mode().iloc[0]
    baseline = baselines.get(pid)
    dedup = AlertDeduplicator()
    predictor.reset(pid)
    timeline, notes = [], []

    for now in pd.date_range(first, last, freq=f"{TICK_S}s"):
        received = sent[(arrival <= now) & (arrival > now - pd.Timedelta(seconds=BUFFER_S))]
        clean = clean_sensor_data(validate_sensor_data(received, now=now)) if len(received) else None
        if clean is not None and len(clean):
            clean = clean[clean["patient_id"] == pid]
            rows = clean[(clean["timestamp_utc"] >= now - pd.Timedelta(seconds=WINDOW_S)) & (clean["timestamp_utc"] < now)]
        else:
            rows = pd.DataFrame(columns=["timestamp_utc", "heart_rate"])
        feats = compute_window_features(rows, baseline, window_end=now)
        feats.update(patient_id=pid, window_end=now)
        result = predictor.predict(feats)
        new = dedup.update(now, result["risk_level"])
        notes.extend(new)
        timeline.append({"now": now, "risk_level": result["risk_level"], "reasons": ";".join(result["reasons"]),
                         "data_quality": result["data_quality"], "hr_mean": feats.get("hr_mean"),
                         "hr_delta_pct": feats.get("hr_delta_pct"), "hrv_delta_pct": feats.get("hrv_delta_pct"),
                         "accel_magnitude_std": feats.get("accel_magnitude_std"),
                         "staleness_s": feats.get("staleness_s"), "notifications": ";".join(n["type"] for n in new),
                         "model_version": result["model_version"]})
    return pd.DataFrame(timeline), notes


# --- automatic checks against ground truth ----------------------------------------------

def check_session(timeline: pd.DataFrame, notes: list[dict], events: pd.DataFrame, session_start: pd.Timestamp,
                  scenario: str) -> list[dict]:
    t = timeline.assign(t_s=(timeline["now"] - session_start).dt.total_seconds(),
                        sev=timeline["risk_level"].map(SEVERITY))
    nt = pd.DataFrame(notes)
    if len(nt):
        nt["t_s"] = (nt["time"] - session_start).dt.total_seconds()
    checks = []

    def add(name, ok, detail):
        # ok=None: not applicable (reported, not counted)
        checks.append({"scenario": scenario, "check": name, "pass": None if ok is None else bool(ok), "detail": detail})

    episodes = events[events["type"].isin(["arousal", "hrv_drop", "sleep_disruption"])]
    covered = np.zeros(len(t), dtype=bool)
    for e in episodes.itertuples():
        covered |= ((t["t_s"] > e.start_s) & (t["t_s"] <= e.end_s + WINDOW_S + 120)).to_numpy()

    for e in events.itertuples():
        if e.type in ("arousal", "sleep_disruption"):
            w = t[(t["t_s"] > e.start_s) & (t["t_s"] <= e.peak_end_s + 120)]
            n_alert = int(((nt["type"] == "alert") & (nt["t_s"] > e.start_s) & (nt["t_s"] <= e.end_s + WINDOW_S)).sum()) if len(nt) else 0
            hit = w[w["sev"] >= 2]
            add(f"B: {e.type} -> elevated", len(hit) > 0,
                f"first elevated {hit['t_s'].min() - e.start_s:.0f} s after onset" if len(hit) else f"max {w['risk_level'].max()}")
            add("B: one alert per episode (dedup)", n_alert == 1, f"{n_alert} alert notification(s)")
        elif e.type == "hrv_drop":
            w = t[(t["t_s"] > e.start_s) & (t["t_s"] <= e.peak_end_s + 120)]
            add("B: hrv_drop -> >= monitor", (w["sev"] >= 1).any(), f"levels seen: {sorted(set(w['risk_level']))}")
        elif e.type == "brief_wake":
            w = t[(t["t_s"] > e.start_s) & (t["t_s"] <= e.end_s + WINDOW_S)]
            n_alert = int(((nt["type"] == "alert") & (nt["t_s"] > e.start_s) & (nt["t_s"] <= e.end_s + WINDOW_S)).sum()) if len(nt) else 0
            add("B: brief wake -> at most monitor, no alert", (w["sev"] <= 1).all() and n_alert == 0,
                f"levels seen: {sorted(set(w['risk_level']))}, {n_alert} alert(s)")
        elif e.type == "activity":
            w = t[(t["t_s"] > e.start_s) & (t["t_s"] <= e.end_s + WINDOW_S)]
            add("B: exercise -> no alert", (w["sev"] < 1).all(), f"levels seen: {sorted(set(w['risk_level']))}")
        elif e.kind == "gap":
            dur = e.end_s - e.start_s
            w = t[(t["t_s"] >= e.start_s + MAX_STALENESS_S + 2) & (t["t_s"] <= e.end_s)]
            dev = int(((nt["type"] == "device_check") & (nt["t_s"] >= e.start_s) & (nt["t_s"] <= e.end_s + TICK_S)).sum()) if len(nt) else 0
            # A gap shorter than staleness + one tick cannot show up, by design (short BLE blips
            # should not alarm anyone); the device check needs DEVICE_CHECK_TICKS ticks on top.
            if dur < MAX_STALENESS_S + TICK_S:
                add(f"C: {e.type} -> insufficient_data", None, f"n/a: gap {dur:.0f} s < {MAX_STALENESS_S + TICK_S} s")
            else:
                add(f"C: {e.type} -> insufficient_data", (w["risk_level"] == "insufficient_data").all(),
                    f"{int((w['risk_level'] == 'insufficient_data').sum())}/{len(w)} ticks, gap {dur:.0f} s")
            if dur < MAX_STALENESS_S + DEVICE_CHECK_TICKS * TICK_S:
                add(f"C: {e.type} -> device-check notification", None,
                    f"n/a: gap {dur:.0f} s < {MAX_STALENESS_S + DEVICE_CHECK_TICKS * TICK_S} s ({dev} sent)")
            else:
                add(f"C: {e.type} -> device-check notification", dev == 1, f"{dev} notification(s)")

    monitor_ok = np.zeros(len(t), dtype=bool)  # a brief wake may show "monitor"
    for e in events[events["type"] == "brief_wake"].itertuples():
        monitor_ok |= ((t["t_s"] > e.start_s) & (t["t_s"] <= e.end_s + WINDOW_S)).to_numpy()
    warm = t["t_s"] >= WINDOW_S  # first window still filling
    false = t[warm & ~covered & (((t["sev"] >= 1) & ~monitor_ok) | (t["sev"] >= 2))]
    add("A: no alert outside episodes", len(false) == 0, f"{len(false)} tick(s) with monitor/elevated outside episodes")
    add("A: never 'normal' without good data", not ((t["risk_level"] == "normal") & (t["data_quality"] != "good")).any(),
        "normal only when data_quality == good")
    return checks


def plot_timeline(raw: pd.DataFrame, timeline: pd.DataFrame, notes: list[dict], events: pd.DataFrame,
                  session_start: pd.Timestamp, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"normal": "#2a9d8f", "monitor": "#e9c46a", "elevated_physiological_anomaly": "#d1495b",
              "insufficient_data": "#8d99ae"}
    shade = {"arousal": "#d1495b", "hrv_drop": "#edae49", "activity": "#00798c", "light_activity": "#8ecae6",
             "disconnected": "#555555", "off_wrist": "#555555", "sleep_disruption": "#6a4c93", "brief_wake": "#b8a1d9"}
    r = raw[raw.get("gt_injected", pd.Series("", index=raw.index)).fillna("").eq("")]
    ts = pd.to_datetime(r["timestamp"], utc=True, errors="coerce", format="ISO8601")
    hr = pd.to_numeric(r["heart_rate"], errors="coerce")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot((ts - session_start).dt.total_seconds() / 60, hr, lw=0.6, color="#222222")
    for e in events.itertuples():
        if e.type in shade:
            for a in (ax, ax2):
                a.axvspan(e.start_s / 60, e.end_s / 60, color=shade[e.type], alpha=0.15, lw=0)
    tm = (timeline["now"] - session_start).dt.total_seconds() / 60
    for lvl, c in colors.items():
        m = timeline["risk_level"] == lvl
        ax2.bar(tm[m], 1, width=1.0, color=c, label=lvl, align="edge")
    for n in notes:
        x = (n["time"] - session_start).total_seconds() / 60
        ax.axvline(x, color={"alert": "red", "device_check": "black", "resolved": "green"}[n["type"]], ls="--", lw=1)
        ax.text(x, ax.get_ylim()[1], f" {n['type']}", rotation=90, va="top", fontsize=7)
    ax.set_ylabel("HR (bpm)")
    ax2.set_yticks([])
    ax2.set_xlabel("menit sejak awal sesi")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=4, fontsize=8)
    fig.suptitle(f"{raw['session_id'].dropna().iloc[0]} — status per menit (pita bawah) vs ground truth (arsiran). SYNTHETIC",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["rules", "model", "mix"], default="rules")
    ap.add_argument("--sessions", nargs="*", help="session ids (default: all SIM01 sessions)")
    args = ap.parse_args()

    baselines = load_baselines(BASELINES)
    events = pd.read_csv(RAW_DIR / "events.csv")
    sessions = args.sessions or sorted(p.stem for p in RAW_DIR.glob("SIM01_*.csv"))
    out_dir = ROOT / "reports" / "stream"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_checks, note_rows = [], []
    for sid in sessions:
        raw = pd.read_csv(RAW_DIR / f"{sid}.csv", dtype=TEXT)
        predictor = Predictor(args.mode)
        timeline, notes = simulate(raw, baselines, predictor)
        start = pd.to_datetime(raw.loc[raw["gt_t_s"] == 0, "timestamp"].iloc[0], utc=True)
        ev = events[events["session_id"] == sid]
        checks = check_session(timeline, notes, ev, start, sid.split("_", 1)[1])
        all_checks.extend(checks)
        note_rows.extend({"session": sid, "minute": round((n["time"] - start).total_seconds() / 60), **n} for n in notes)
        timeline.to_csv(out_dir / f"{sid}_{args.mode}_timeline.csv", index=False)
        plot_timeline(raw, timeline, notes, ev, start, out_dir / f"{sid}_{args.mode}.png")
        counted = [c["pass"] for c in checks if c["pass"] is not None]
        print(f"{sid:24s} {sum(counted)}/{len(counted)} checks pass, {len(notes)} notification(s), "
              f"model={predictor.model_version}")

    counted = [c for c in all_checks if c["pass"] is not None]
    checks = pd.DataFrame(counted)
    notes = pd.DataFrame(note_rows)
    n_ok = int(checks["pass"].sum())
    lines = [
        f"# Gladi bersih offline — mode `{args.mode}`\n",
        "> Data sintetis. Membuktikan alur dan logika sistem, bukan akurasi klinis.\n",
        "Alur per tick 60 s: record diterima (urut `received_at`) → `validate_sensor_data` → `clean_sensor_data` → "
        "`compute_window_features(window_end=now)` → `Predictor.predict` → deduplikasi notifikasi. "
        "Kolom `gt_*` dibuang sebelum masuk sistem.\n",
        f"## Hasil: {n_ok}/{len(checks)} cek lulus\n",
        "| sesi | cek | lulus | detail |", "|---|---|---|---|",
        *[f"| {c['scenario']} | {c['check']} | {'n/a' if c['pass'] is None else ('ya' if c['pass'] else '**TIDAK**')} "
          f"| {c['detail']} |" for c in all_checks],
        "\n## Notifikasi ke caregiver\n",
        "| sesi | menit | tipe | pesan |", "|---|---|---|---|",
        *[f"| {r.session} | {r.minute} | {r.type} | {r.message} |" for r in notes.itertuples()],
        "\n## Aturan notifikasi\n",
        "- `alert`: saat status masuk `elevated_physiological_anomaly`. Episode baru dalam 30 menit setelah alert "
        "terakhir tidak dikirim ulang (mencegah notifikasi berulang saat status naik-turun).",
        "- `resolved`: saat status turun dari `elevated` ke `normal`/`monitor`.",
        f"- `device_check`: sekali per gangguan, saat `insufficient_data` bertahan {DEVICE_CHECK_TICKS} tick, setelah data "
        "pernah `good` (tidak saat window pertama masih terisi).",
        f"- Gap lebih pendek dari {MAX_STALENESS_S + TICK_S} s tidak terlihat sebagai `insufficient_data` (disengaja: "
        "putus BLE singkat tidak perlu membuat caregiver panik). Cek seperti itu ditandai n/a.",
        "- `monitor` tidak mengirim notifikasi, hanya terlihat di status app.\n",
        f"Plot dan timeline per sesi: `reports/stream/<sesi>_{args.mode}.png` dan `_timeline.csv`.",
    ]
    (ROOT / "reports" / f"stream_rehearsal_{args.mode}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{n_ok}/{len(checks)} checks pass -> reports/stream_rehearsal_{args.mode}.md")


if __name__ == "__main__":
    main()
