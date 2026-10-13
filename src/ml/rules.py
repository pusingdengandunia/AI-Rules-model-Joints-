"""Rule-based baseline (spec Step 7) — Target A, physiological anomaly.

Transparent prototype rules, NOT validated medical criteria. Thresholds live in
RULES_CONFIG; the daytime ones were tuned on synthetic patients only
(see reports/rules_eval_synthetic.md).

Per window, in this order:
  1. data_quality != good, or no valid baseline        -> insufficient_data (always wins)
  2. window_end inside the patient's sleep window (local time, default 22:00-06:00):
       restless   accel_magnitude_std >= SLEEP_RESTLESS_ACCEL_STD
       arousal    still, and hr_delta_pct >= SLEEP_HR_ELEVATION_PCT
       otherwise  calm sleep -> normal
     Movement at night is never dismissed as "activity".
  3. daytime:
       moving (accel_magnitude_std high), or within the
       cool-down after movement                        -> normal (reason: activity / post_activity)
       at rest and HR above or HRV below baseline      -> abnormal candidate

Persistence (RuleEngine, per patient):
  daytime    1-2 consecutive abnormal windows -> monitor; >= persist_windows -> elevated
  night      anomaly covering < SLEEP_SUSTAINED_WINDOWS x 5 min -> monitor (sleep_disruption_candidate)
             covering >= that (~10 min)                          -> elevated (sleep_disruption)
             Coverage is measured in time, not in window count, because windows overlap
             (5 min long, 1 min step): a 3-minute wake-up shows in ~7 overlapping windows but
             covers < 10 min, so it stays at monitor.
A gap in the window sequence or an insufficient window resets the streaks:
an episode is never extended across data the system could not see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from src.schema import WINDOW_S, WINDOW_STEP_S

RULES_VERSION = "rules_v1"

RULES_CONFIG = {
    # Daytime rules. Tuned by `python -m src.ml.evaluate_rules --tune` on synthetic SIM01-02 (seed 42);
    # 36/144 grid configs pass there and all 36 also pass SIM03 and the seed-7 holdout.
    "hr_delta_pct_high": 15.0,      # hr_mean >= baseline + 15 %
    "hrv_delta_pct_low": -40.0,     # hrv_rmssd_mean <= baseline - 40 %
    "activity_accel_std": 0.08,     # g; above this the window counts as movement
    "post_activity_cooldown_s": 600,  # HR/HRV need time to recover after exercise
    "persist_windows": 3,           # consecutive abnormal windows for "elevated"
    # Sleeptime anomaly (night rules). Prototype values, not tuned on real sleep data.
    "PATIENT_TIMEZONE": "Asia/Jakarta",  # default; a window may carry its own "timezone"
    "SLEEP_WINDOW_START": "22:00",
    "SLEEP_WINDOW_END": "06:00",
    "SLEEP_HR_ELEVATION_PCT": 0.15,      # fraction: +15 % vs baseline (compared with hr_delta_pct / 100)
    "SLEEP_RESTLESS_ACCEL_STD": 0.08,    # g
    "SLEEP_SUSTAINED_WINDOWS": 2,        # x 5-minute windows (~10 min) for "elevated"
}

LEVELS = ["normal", "monitor", "elevated_physiological_anomaly", "insufficient_data"]
SEVERITY = {"insufficient_data": -1, "normal": 0, "monitor": 1, "elevated_physiological_anomaly": 2}
SLEEP_KINDS = {"sleep_restless", "sleep_arousal", "sleep_calm"}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_sleep_window(window_end, config: dict = RULES_CONFIG, timezone: str | None = None) -> bool:
    """True if window_end, in the patient's local time, falls inside [start, end) (may cross midnight)."""
    if window_end is None:
        return False
    ts = pd.Timestamp(window_end)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    local = ts.tz_convert(timezone or config["PATIENT_TIMEZONE"])
    t = local.hour * 60 + local.minute
    start, end = _minutes(config["SLEEP_WINDOW_START"]), _minutes(config["SLEEP_WINDOW_END"])
    return (start <= t < end) if start < end else (t >= start or t < end)


def assess_window(features: dict, config: dict = RULES_CONFIG) -> tuple[str, list[str]]:
    """Stateless check of one window.

    Returns (kind, reasons) with kind in: insufficient, sleep_restless, sleep_arousal, sleep_calm,
    activity, abnormal, normal.
    """
    config = {**RULES_CONFIG, **config}
    if features.get("data_quality") != "good":
        stale = features.get("staleness_s")
        return "insufficient", ["stale_data" if not _missing(stale) and stale > 60 else "insufficient_samples"]
    hr_d, hrv_d = features.get("hr_delta_pct"), features.get("hrv_delta_pct")
    if _missing(hr_d):
        return "insufficient", ["no_baseline"]
    accel = features.get("accel_magnitude_std")

    if in_sleep_window(features.get("window_end"), config, features.get("timezone")):
        if not _missing(accel) and accel >= config["SLEEP_RESTLESS_ACCEL_STD"]:
            return "sleep_restless", ["high_night_restlessness"]
        if hr_d >= config["SLEEP_HR_ELEVATION_PCT"] * 100:
            return "sleep_arousal", ["sustained_night_arousal"]
        return "sleep_calm", []

    if not _missing(accel) and accel > config["activity_accel_std"]:
        return "activity", ["activity"]
    reasons = []
    if hr_d >= config["hr_delta_pct_high"]:
        reasons.append("hr_above_baseline")
    if not _missing(hrv_d) and hrv_d <= config["hrv_delta_pct_low"]:
        reasons.append("hrv_below_baseline")
    if reasons:
        return "abnormal", reasons + ["at_rest"]
    return "normal", []


@dataclass
class _PatientState:
    streak: int = 0
    last_window_end: pd.Timestamp | None = None
    last_activity_end: pd.Timestamp | None = None
    sleep_first_end: pd.Timestamp | None = None   # first window_end of the current night anomaly
    sleep_prev_end: pd.Timestamp | None = None    # previous anomalous night window_end


@dataclass
class RuleEngine:
    """Stateful rules for a stream of windows (one call per window, in time order per patient)."""

    config: dict = field(default_factory=lambda: dict(RULES_CONFIG))
    _state: dict = field(default_factory=dict)

    def __post_init__(self):
        self.config = {**RULES_CONFIG, **self.config}  # partial configs (e.g. tuning grids) keep the rest

    def reset(self, patient_id) -> None:
        """Forget the streaks and activity history of one patient (e.g. new session)."""
        self._state.pop(patient_id, None)

    def predict(self, features: dict) -> dict:
        pid = features.get("patient_id")
        end = pd.Timestamp(features["window_end"]) if features.get("window_end") is not None else None
        st = self._state.setdefault(pid, _PatientState())

        # Non-consecutive windows: we did not see what happened in between.
        if st.last_window_end is not None and end is not None:
            if (end - st.last_window_end).total_seconds() > WINDOW_STEP_S * 1.5:
                st.streak = 0
        st.last_window_end = end

        kind, reasons = assess_window(features, self.config)
        if kind in SLEEP_KINDS:
            st.streak = 0
            level, reasons = self._night(st, kind, reasons, end)
        else:
            st.sleep_first_end = st.sleep_prev_end = None
            level, reasons = self._day(st, kind, reasons, end)

        return {
            "patient_id": pid,
            "window_end": str(end) if end is not None else None,
            "risk_level": level,
            "risk_score": None,
            "reasons": reasons,
            "model_version": RULES_VERSION,
            "data_quality": features.get("data_quality"),
        }

    def _night(self, st: _PatientState, kind: str, reasons: list[str], end) -> tuple[str, list[str]]:
        if kind == "sleep_calm":
            st.sleep_first_end = st.sleep_prev_end = None
            return "normal", []
        # Continue the run if the previous anomalous window is at most one window length back
        # (works for 1-minute sliding windows and for back-to-back 5-minute windows).
        continuing = (st.sleep_prev_end is not None and end is not None
                      and (end - st.sleep_prev_end).total_seconds() <= WINDOW_S * 1.1)
        spacing = (end - st.sleep_prev_end).total_seconds() if continuing else WINDOW_STEP_S
        if not continuing:
            st.sleep_first_end = end
        st.sleep_prev_end = end
        covered = (end - st.sleep_first_end).total_seconds() + spacing if end is not None else 0
        if covered >= self.config["SLEEP_SUSTAINED_WINDOWS"] * WINDOW_S:
            return "elevated_physiological_anomaly", ["sleep_disruption"] + reasons
        return "monitor", ["sleep_disruption_candidate"]

    def _day(self, st: _PatientState, kind: str, reasons: list[str], end) -> tuple[str, list[str]]:
        if kind == "activity":
            st.last_activity_end = end
        elif kind == "abnormal" and st.last_activity_end is not None and end is not None:
            if (end - st.last_activity_end).total_seconds() <= self.config["post_activity_cooldown_s"]:
                kind, reasons = "normal", ["post_activity"]
        if kind == "abnormal":
            st.streak += 1
            level = "elevated_physiological_anomaly" if st.streak >= self.config["persist_windows"] else "monitor"
            return level, reasons
        st.streak = 0
        return ("insufficient_data" if kind == "insufficient" else "normal"), reasons


def run_rules(features: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Batch helper: run a fresh engine over a feature table in time order per session.

    Returns the input rows with risk_level and reasons added.
    """
    engine = RuleEngine(config=dict(config or {}))
    out = features.sort_values(["patient_id", "session_id", "window_end"]).copy()
    levels, reasons = [], []
    last_session = {}
    for rec in out.to_dict("records"):
        # Each session starts with a clean streak (a new recording).
        if last_session.get(rec["patient_id"]) != rec["session_id"]:
            engine.reset(rec["patient_id"])
            last_session[rec["patient_id"]] = rec["session_id"]
        r = engine.predict(rec)
        levels.append(r["risk_level"])
        reasons.append(";".join(r["reasons"]))
    out["risk_level"] = levels
    out["reasons"] = reasons
    return out
