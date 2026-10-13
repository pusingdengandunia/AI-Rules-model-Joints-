"""Rule-based baseline (spec Step 7) — Target A, physiological anomaly.

Transparent prototype rules, NOT validated medical criteria. Thresholds live in
RULES_CONFIG and were tuned on synthetic patients only (see reports/rules_eval_synthetic.md).

Per window:
  1. data_quality != good, or no valid baseline        -> insufficient_data
  2. moving (accel_magnitude_std high), or within the
     cool-down after movement                          -> normal (reason: activity / post_activity)
  3. at rest and HR above or HRV below baseline         -> abnormal candidate
Persistence (RuleEngine, per patient):
  1-2 consecutive abnormal windows -> monitor
  >= persist_windows               -> elevated_physiological_anomaly
A gap in the window sequence or an insufficient window resets the streak:
an episode is never extended across data the system could not see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from src.schema import WINDOW_STEP_S

RULES_VERSION = "rules_v1"

# Tuned by `python -m src.ml.evaluate_rules --tune` on synthetic SIM01-02 (seed 42);
# 36/144 grid configs pass there and all 36 also pass SIM03 and the seed-7 holdout.
RULES_CONFIG = {
    "hr_delta_pct_high": 15.0,      # hr_mean >= baseline + 15 %
    "hrv_delta_pct_low": -40.0,     # hrv_rmssd_mean <= baseline - 40 %
    "activity_accel_std": 0.08,     # g; above this the window counts as movement
    "post_activity_cooldown_s": 600,  # HR/HRV need time to recover after exercise
    "persist_windows": 3,           # consecutive abnormal windows for "elevated"
}

LEVELS = ["normal", "monitor", "elevated_physiological_anomaly", "insufficient_data"]
SEVERITY = {"insufficient_data": -1, "normal": 0, "monitor": 1, "elevated_physiological_anomaly": 2}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def assess_window(features: dict, config: dict = RULES_CONFIG) -> tuple[str, list[str]]:
    """Stateless check of one window -> ('insufficient' | 'activity' | 'abnormal' | 'normal', reasons)."""
    if features.get("data_quality") != "good":
        stale = features.get("staleness_s")
        return "insufficient", ["stale_data" if not _missing(stale) and stale > 60 else "insufficient_samples"]
    hr_d, hrv_d = features.get("hr_delta_pct"), features.get("hrv_delta_pct")
    if _missing(hr_d):
        return "insufficient", ["no_baseline"]

    accel = features.get("accel_magnitude_std")
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


@dataclass
class RuleEngine:
    """Stateful rules for a stream of windows (one call per window, in time order per patient)."""

    config: dict = field(default_factory=lambda: dict(RULES_CONFIG))
    _state: dict = field(default_factory=dict)

    def reset(self, patient_id) -> None:
        """Forget the streak and activity history of one patient (e.g. new session)."""
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
        if kind == "activity":
            st.last_activity_end = end
        elif kind == "abnormal" and st.last_activity_end is not None and end is not None:
            if (end - st.last_activity_end).total_seconds() <= self.config["post_activity_cooldown_s"]:
                kind, reasons = "normal", ["post_activity"]

        if kind == "abnormal":
            st.streak += 1
            level = "elevated_physiological_anomaly" if st.streak >= self.config["persist_windows"] else "monitor"
        else:
            st.streak = 0
            level = "insufficient_data" if kind == "insufficient" else "normal"

        return {
            "patient_id": pid,
            "window_end": str(end) if end is not None else None,
            "risk_level": level,
            "risk_score": None,
            "reasons": reasons,
            "model_version": RULES_VERSION,
            "data_quality": features.get("data_quality"),
        }


def run_rules(features: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Batch helper: run a fresh engine over a feature table in time order per session.

    Returns the input rows with risk_level and reasons added.
    """
    engine = RuleEngine(config=dict(config or RULES_CONFIG))
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
