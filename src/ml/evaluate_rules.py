"""Event-level evaluation (and optional tuning) of the rules on synthetic data — Target A.

Patients are split so thresholds are never judged on the data they were tuned on:
  tune     seed 42, SIM01-SIM02
  test     seed 42, SIM03
  holdout  seed 7,  SIM01-SIM03 (freshly generated, different random patients)

Criteria (docs/ml_target.md Bagian 5):
  arousal  event -> reaches elevated_physiological_anomaly        recall 100 %
  hrv_drop event -> reaches at least monitor                       recall >= 80 %
  normal / high_activity / missing_data / bad_quality sessions    0 alerts
  windows ending inside a gap / off-wrist period                  100 % insufficient_data

Run:
  python -m src.ml.evaluate_rules           # evaluate RULES_CONFIG, write the report
  python -m src.ml.evaluate_rules --tune    # grid search on the tune split first
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from src.features.engineer import build_feature_table
from src.ingestion.validate import validate_sensor_data
from src.ml.rules import RULES_CONFIG, SEVERITY, run_rules
from src.preprocessing.clean import clean_sensor_data
from src.schema import MAX_STALENESS_S
from src.simulator.generate import generate_all

ROOT = Path(__file__).resolve().parents[2]
TARGET_EVENTS = {"arousal": 2, "hrv_drop": 1}  # required minimum severity
NO_ALERT_SCENARIOS = ["normal", "high_activity", "missing_data", "bad_quality"]
TOLERANCE_S = 120

TUNE_GRID = {
    "hr_delta_pct_high": [15.0, 20.0, 25.0],
    "hrv_delta_pct_low": [-25.0, -30.0, -35.0, -40.0],
    "activity_accel_std": [0.08, 0.10, 0.15],
    "post_activity_cooldown_s": [300, 600],
    "persist_windows": [2, 3],
}


def build_synthetic(seed: int, n_patients: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulator -> validate -> clean -> features, in memory. Adds session-relative ws_s / we_s."""
    frames, events, _ = generate_all(seed, n_patients)
    clean = clean_sensor_data(validate_sensor_data(pd.concat(frames, ignore_index=True)))
    feats, _ = build_feature_table(clean, "calibration")
    start = clean.groupby("session_id")["timestamp_utc"].min()
    s0 = feats["session_id"].map(start)
    feats["ws_s"] = (feats["window_start"] - s0).dt.total_seconds()
    feats["we_s"] = (feats["window_end"] - s0).dt.total_seconds()
    return feats, events


def evaluate(scored: pd.DataFrame, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Metrics for windows already scored with risk_level."""
    scored = scored.assign(sev=scored["risk_level"].map(SEVERITY))
    sessions = set(scored["session_id"])
    ev = events[events["session_id"].isin(sessions)]
    rows = []
    covered = pd.Series(False, index=scored.index)  # windows that belong to a real episode

    for e in ev[ev["type"].isin(TARGET_EVENTS)].itertuples():
        w = scored[(scored["session_id"] == e.session_id)
                   & (scored["ws_s"] < e.peak_end_s + TOLERANCE_S) & (scored["we_s"] > e.start_s)]
        need = TARGET_EVENTS[e.type]
        hit = w[w["sev"] >= need]
        rows.append(dict(session_id=e.session_id, event=e.type, start_s=e.start_s,
                         max_level=w.loc[w["sev"].idxmax(), "risk_level"] if len(w) else None,
                         detected=bool(len(hit)),
                         latency_s=float(hit["we_s"].min() - e.start_s) if len(hit) else None))
        # Recovery belongs to the episode too, so alerts there are not false alarms.
        covered |= ((scored["session_id"] == e.session_id)
                    & (scored["ws_s"] < e.end_s + TOLERANCE_S) & (scored["we_s"] > e.start_s))
    per_event = pd.DataFrame(rows)

    false = scored[(scored["sev"] >= 1) & ~covered].sort_values(["session_id", "we_s"])
    episodes = int((false.groupby("session_id")["we_s"].diff().fillna(1e9) > 90).sum())
    hours = float((scored.groupby("session_id")["we_s"].max() - scored.groupby("session_id")["ws_s"].min()).sum() / 3600)

    gap_rows = []
    for g in ev[ev["kind"] == "gap"].itertuples():
        w = scored[(scored["session_id"] == g.session_id)
                   & (scored["we_s"] >= g.start_s + MAX_STALENESS_S + 2) & (scored["we_s"] <= g.end_s)]
        gap_rows.append(dict(n=len(w), ok=int((w["risk_level"] == "insufficient_data").sum())))
    gap_n = sum(r["n"] for r in gap_rows)

    scen = scored["session_id"].str.split("_", n=1).str[1]
    no_alert = {s: int(((scen == s) & (scored["sev"] >= 1)).sum()) for s in NO_ALERT_SCENARIOS}

    def recall(t):
        sub = per_event[per_event["event"] == t] if len(per_event) else per_event
        return float(sub["detected"].mean()) if len(sub) else float("nan")

    metrics = dict(
        arousal_recall=recall("arousal"),
        hrv_drop_recall=recall("hrv_drop"),
        mean_latency_s=float(per_event["latency_s"].dropna().mean()) if len(per_event) else float("nan"),
        false_alarm_windows=int(len(false)),
        false_alarm_episodes=episodes,
        false_alarm_episodes_per_hour=round(episodes / hours, 3) if hours else float("nan"),
        alerts_in_no_alert_scenarios=no_alert,
        gap_windows=gap_n,
        gap_insufficient_ratio=(sum(r["ok"] for r in gap_rows) / gap_n) if gap_n else float("nan"),
        hours=round(hours, 2),
    )
    return metrics, per_event


def passes(m: dict) -> bool:
    return (m["arousal_recall"] == 1.0 and m["hrv_drop_recall"] >= 0.8
            and sum(m["alerts_in_no_alert_scenarios"].values()) == 0 and m["gap_insufficient_ratio"] == 1.0)


def tune(feats: pd.DataFrame, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Grid search on the tune split.

    Pick a passing config with the fewest false alarms; break ties toward the conservative side
    (longer persistence, longer cool-down) before latency. Picking the fastest config overfits:
    it sits right at the edge of the tune patients' noise and fails on new patients.
    """
    keys = list(TUNE_GRID)
    results = []
    for values in itertools.product(*TUNE_GRID.values()):
        cfg = dict(zip(keys, values))
        m, _ = evaluate(run_rules(feats, cfg), events)
        results.append({**cfg, **{k: m[k] for k in ("arousal_recall", "hrv_drop_recall", "false_alarm_episodes",
                                                   "mean_latency_s", "gap_insufficient_ratio")},
                        "no_alert_total": sum(m["alerts_in_no_alert_scenarios"].values()), "passes": passes(m)})
    res = pd.DataFrame(results)
    ok = res[res["passes"]].sort_values(
        ["false_alarm_episodes", "persist_windows", "post_activity_cooldown_s", "mean_latency_s"],
        ascending=[True, False, False, True])
    if ok.empty:
        raise RuntimeError("no configuration meets the criteria on the tune split")
    best = {k: ok.iloc[0][k] for k in keys}
    best = {k: (int(v) if k in ("post_activity_cooldown_s", "persist_windows") else float(v)) for k, v in best.items()}
    return best, res


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(kosong)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if v is None or (isinstance(v, float) and pd.isna(v))
                                     else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out) + "\n"


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append(dict(split=name, arousal_recall=m["arousal_recall"], hrv_drop_recall=m["hrv_drop_recall"],
                         mean_latency_s=m["mean_latency_s"], false_alarm_episodes=m["false_alarm_episodes"],
                         fa_per_hour=m["false_alarm_episodes_per_hour"],
                         no_alert_scenario_alerts=sum(m["alerts_in_no_alert_scenarios"].values()),
                         gap_insufficient=m["gap_insufficient_ratio"], hours=m["hours"],
                         passes="ya" if passes(m) else "TIDAK"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    f42, e42 = build_synthetic(42, 3)
    f7, e7 = build_synthetic(7, 3)
    splits = {
        "tune (seed 42, SIM01-02)": (f42[f42["patient_id"].isin(["SIM01", "SIM02"])], e42),
        "test (seed 42, SIM03)": (f42[f42["patient_id"] == "SIM03"], e42),
        "holdout (seed 7, SIM01-03)": (f7, e7),
    }

    config, grid = dict(RULES_CONFIG), None
    if args.tune:
        config, grid = tune(*splits["tune (seed 42, SIM01-02)"])
        print("best config on tune split:", json.dumps(config))

    results, per_event = {}, []
    for name, (feats, events) in splits.items():
        m, pe = evaluate(run_rules(feats, config), events)
        results[name] = m
        per_event.append(pe.assign(split=name))
        print(name, json.dumps(m, default=str))

    s = ["# Evaluasi rules_v1 pada data sintetis (Target A)\n",
         "> Data sintetis: hasil ini membuktikan logika rules dan pipeline bekerja sesuai desain skenario, "
         "bukan akurasi klinis.\n",
         "## Konfigurasi\n", "```json\n" + json.dumps(config, indent=2) + "\n```\n",
         "## Hasil per split\n",
         "Kriteria lulus: recall arousal→elevated 100%, recall hrv_drop→monitor ≥ 80%, 0 alert di skenario "
         "normal/high_activity/missing_data/bad_quality, 100% window gap = insufficient_data.\n",
         md(metrics_table(results)),
         "\n`fa_per_hour` = episode false alarm per jam data. Episode = alert berurutan di luar event "
         f"(event + recovery + {TOLERANCE_S} s toleransi).\n",
         "## Detail per event\n", md(pd.concat(per_event)[["split", "session_id", "event", "start_s",
                                                          "max_level", "detected", "latency_s"]])]
    if grid is not None:
        top = grid.sort_values(["passes", "false_alarm_episodes", "persist_windows", "post_activity_cooldown_s",
                                "mean_latency_s"], ascending=[False, True, False, False, True]).head(10)
        s += ["\n## Tuning (grid search, hanya split tune)\n",
              f"{int(grid['passes'].sum())} dari {len(grid)} konfigurasi lulus kriteria di split tune. Top 10:\n", md(top)]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "rules_eval_synthetic.md").write_text("\n".join(s), encoding="utf-8")
    print("report: reports/rules_eval_synthetic.md")


if __name__ == "__main__":
    main()
