"""Live status engine: the backend counterpart of `simulate()` in src/ml/simulate_stream.py.

Per patient it keeps the records received in the last 15 minutes and, every 60 s of that
patient's clock, runs the same chain as the offline rehearsal:

    validate_sensor_data -> clean_sensor_data -> compute_window_features(window_end=now)
    -> Predictor("rules").predict -> AlertDeduplicator

so a simulator session replayed into POST /ingest gives exactly reports/stream/<sesi>_rules_timeline.csv
(checked by tests/test_backend_replay.py).

Two clocks, chosen per incoming record:
  wall    real devices (Health Connect). arrival = server time; ticks on the wall-clock minute.
  replay  simulator streams (source "simulator" or header X-Replay-Speed). arrival = the record's
          received_at (else its timestamp), so the clock follows the data. While no data arrives the
          clock keeps running at the replay speed, so a sensor gap turns gray live during the demo.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from backend import texts
from backend.db import Database, iso
from src.features.engineer import build_feature_table, compute_window_features
from src.ingestion.validate import validate_sensor_data
from src.ml.predict import Predictor
from src.ml.simulate_stream import BUFFER_S, TICK_S, AlertDeduplicator
from src.preprocessing.clean import clean_sensor_data, nominal_interval_s
from src.schema import BASELINE_CALIBRATION_S, RAW_COLUMNS, REQUIRED_COLUMNS, SESSION_GAP_S, WINDOW_S

log = logging.getLogger("neuvana.engine")

ELEVATED = texts.ELEVATED
TICK = pd.Timedelta(seconds=TICK_S)
COLLECTING_S = WINDOW_S          # first window after the watch is put on: "Mengumpulkan data"
STALE_STATUS_S = 120             # an older status is never shown as if it still held
REPLAY_IDLE_GRACE_S = 2.0        # wall seconds without simulator data before its clock runs on its own
REPLAY_REWIND_S = 300            # simulator data jumping back this far = a new replay: start over
ACTIVE_TTL_S = 6 * 3600          # stop ticking a patient this long after their last record
HR_EVENT_EVERY_S = 3.0           # live heart-rate events are throttled to one per this many wall seconds

Emit = Callable[[str, dict, str], None]                         # (patient_id, event, audience)
Dispatch = Callable[[str, str, pd.Timestamp], None]             # (patient_id, kind, event_time)


def parse_ts(value) -> pd.Timestamp | None:
    if not isinstance(value, str) or not value.strip():
        return None
    ts = pd.to_datetime(value.strip(), utc=True, errors="coerce", format="ISO8601")
    return None if pd.isna(ts) else ts


@dataclass
class Runtime:
    """In-memory state of one patient (one process holds all of them; PRD Keandalan)."""

    patient_id: str
    clock: str = "wall"
    speed: float = 30.0
    buffer: deque = field(default_factory=deque)          # (arrival, record)
    dedup: AlertDeduplicator = field(default_factory=AlertDeduplicator)
    next_tick: pd.Timestamp | None = None
    last_arrival: pd.Timestamp | None = None
    last_wall: float = 0.0
    stream_started: pd.Timestamp | None = None
    session_id: str | None = None
    status: dict | None = None
    prev_level: str | None = None
    guide_step: int = 0
    episode: dict | None = None
    heart_rate: dict | None = None
    last_hr_event: float = float("-inf")


class StatusEngine:
    def __init__(self, db: Database, emit: Emit | None = None, dispatch: Dispatch | None = None,
                 clock: Callable[[], tuple[pd.Timestamp, float]] | None = None, replay_speed: float = 30.0):
        self.db = db
        # Rules only: statuses and notifications never come from the WESAD model (PRD Keamanan).
        self.predictor = Predictor("rules")
        self.emit = emit or (lambda *a: None)
        self.dispatch = dispatch or (lambda *a: None)
        self.clock = clock or (lambda: (pd.Timestamp.now(tz="UTC"), time.monotonic()))
        self.replay_speed = replay_speed
        self.runtimes: dict[str, Runtime] = {}

    # --- runtimes and clocks -------------------------------------------------------------

    def reset(self, patient_id: str) -> Runtime:
        """New session (or restart): forget streaks, buffer and dedup state."""
        self.predictor.reset(patient_id)
        self.runtimes[patient_id] = Runtime(patient_id, speed=self.replay_speed)
        return self.runtimes[patient_id]

    def runtime(self, patient_id: str) -> Runtime:
        return self.runtimes.get(patient_id) or self.reset(patient_id)

    def clock_now(self, rt: Runtime | None, wall_now: pd.Timestamp, mono: float) -> pd.Timestamp:
        if rt is None or rt.clock == "wall" or rt.last_arrival is None:
            return wall_now
        idle = max(0.0, mono - rt.last_wall) if rt.speed > 0 else 0.0
        return rt.last_arrival + pd.Timedelta(seconds=idle * rt.speed)

    def now_for(self, patient_id: str) -> pd.Timestamp:
        return self.clock_now(self.runtimes.get(patient_id), *self.clock())

    # --- ingest --------------------------------------------------------------------------

    def ingest(self, payloads: list, replay_speed: float | None = None) -> list[dict]:
        """Store every record as received and feed the usable ones to their patient's window buffer.

        Never raises on bad records: they are stored with a quality_flag (PRD F1).
        """
        wall_now, mono = self.clock()
        recs, arrivals = [], []
        for p in payloads:
            if not isinstance(p, dict):
                recs.append(None)
                arrivals.append(None)
                continue
            rec = {k: p[k] for k in RAW_COLUMNS if k in p}
            for k in REQUIRED_COLUMNS:
                rec.setdefault(k, None)
            pid = rec["patient_id"]
            pid = str(pid).strip() if pid is not None and not isinstance(pid, (dict, list)) else ""
            rec["patient_id"] = pid or None
            if replay_speed is not None or rec.get("source") == "simulator":
                arrival = parse_ts(rec.get("received_at")) or parse_ts(rec.get("timestamp"))
            else:
                arrival = wall_now
                rec["received_at"] = iso(wall_now)
            recs.append(rec)
            arrivals.append(arrival)

        flags = self._flags([r for r in recs if r is not None], wall_now)
        flag_iter = iter(flags)
        out, rows = [], []
        for p, rec, arrival in zip(payloads, recs, arrivals):
            if rec is None:
                rows.append((None, None, None, None, "invalid_record", {"raw": p}))
                out.append({"patient_id": None, "quality_flag": "invalid_record"})
                continue
            flag = next(flag_iter)
            stored = {**p, "received_at": rec["received_at"]}
            rows.append((rec["patient_id"], rec["timestamp"], rec["received_at"],
                         iso(arrival) if arrival is not None else None, flag, stored))
            out.append({"patient_id": rec["patient_id"], "quality_flag": flag})
        self.db.insert_records(rows)

        for rec, arrival in zip(recs, arrivals):
            if rec is not None and rec["patient_id"]:
                replay = replay_speed is not None or rec.get("source") == "simulator"
                self._accept(rec, arrival, "replay" if replay else "wall", replay_speed, mono)
        return out

    @staticmethod
    def _flags(recs: list[dict], now: pd.Timestamp) -> list[str]:
        if not recs:
            return []
        try:
            v = validate_sensor_data(pd.DataFrame.from_records(recs), now=now)
            return v.sort_values("ingest_seq")["quality_flag"].tolist()
        except Exception:  # a flag is informational; never lose the record over it
            log.exception("validation of an ingest batch failed")
            return ["unvalidated"] * len(recs)

    def _accept(self, rec: dict, arrival: pd.Timestamp | None, clock: str, replay_speed: float | None,
                mono: float) -> None:
        pid = rec["patient_id"]
        if not self.db.get_patient(pid):
            self.db.ensure_patient(pid)
        rt = self.runtime(pid)
        session = rec.get("session_id") if isinstance(rec.get("session_id"), str) else None
        if rt.next_tick is not None:
            rewind = (clock == "replay" and arrival is not None and rt.last_arrival is not None
                      and arrival < rt.last_arrival - pd.Timedelta(seconds=REPLAY_REWIND_S))
            new_session = clock == "replay" and session and rt.session_id and session != rt.session_id
            if rt.clock != clock or rewind or new_session:
                rt = self.reset(pid)
        rt.clock = clock
        rt.speed = replay_speed if replay_speed is not None else self.replay_speed
        rt.session_id = session or rt.session_id
        if arrival is None:
            return  # stored, but can never enter a window (same as the offline rehearsal)
        if rt.next_tick is None:
            rt.next_tick = arrival.ceil(f"{TICK_S}s")
        self._run_ticks(rt, arrival, inclusive=False)  # a record arriving at t belongs to tick t
        if rt.stream_started is None or (arrival - rt.last_arrival).total_seconds() > SESSION_GAP_S:
            rt.stream_started = arrival
        rt.buffer.append((arrival, rec))
        rt.last_arrival = arrival if rt.last_arrival is None else max(rt.last_arrival, arrival)
        rt.last_wall = mono
        self._heart_rate_event(rt, rec, mono)

    def _heart_rate_event(self, rt: Runtime, rec: dict, mono: float) -> None:
        hr = rec.get("heart_rate")
        if isinstance(hr, bool) or not isinstance(hr, (int, float)) or not 25 <= hr <= 240:
            return
        rt.heart_rate = {"value": round(float(hr)), "timestamp": rec.get("timestamp")}
        if mono - rt.last_hr_event >= HR_EVENT_EVERY_S:
            rt.last_hr_event = mono
            self.emit(rt.patient_id, {"type": "heart_rate", "heart_rate": rt.heart_rate}, "all")

    # --- ticking -------------------------------------------------------------------------

    def poll(self) -> None:
        """Called about twice a second: runs every tick that is due, for every active patient.

        Patients who stopped sending data keep being ticked, which is how they turn gray (PRD F2).
        """
        wall_now, mono = self.clock()
        for rt in list(self.runtimes.values()):
            if rt.next_tick is None or rt.last_arrival is None:
                continue
            if rt.clock == "replay" and mono - rt.last_wall < REPLAY_IDLE_GRACE_S:
                continue  # data is flowing: the records themselves drive the clock
            limit = self.clock_now(rt, wall_now, mono)
            if (limit - rt.last_arrival).total_seconds() > ACTIVE_TTL_S:
                continue
            self._run_ticks(rt, limit, inclusive=True)

    def flush(self, patient_id: str, until: pd.Timestamp) -> None:
        """Run all ticks up to `until` (end of a replay at --speed 0, tests)."""
        rt = self.runtimes.get(patient_id)
        if rt is not None and rt.next_tick is not None:
            self._run_ticks(rt, until, inclusive=True)

    def _run_ticks(self, rt: Runtime, limit: pd.Timestamp, inclusive: bool) -> None:
        while rt.next_tick is not None and (rt.next_tick <= limit if inclusive else rt.next_tick < limit):
            self._tick(rt, rt.next_tick)
            rt.next_tick += TICK

    def _window(self, pid: str, recs: list[dict], now: pd.Timestamp) -> tuple[pd.DataFrame, float]:
        if not recs:
            return pd.DataFrame(columns=["timestamp_utc", "heart_rate"]), WINDOW_S
        clean = clean_sensor_data(validate_sensor_data(pd.DataFrame.from_records(recs), now=now))
        if not len(clean):
            return pd.DataFrame(columns=["timestamp_utc", "heart_rate"]), WINDOW_S
        clean = clean[clean["patient_id"] == pid]
        rows = clean[(clean["timestamp_utc"] >= now - pd.Timedelta(seconds=WINDOW_S)) & (clean["timestamp_utc"] < now)]
        # Health Connect may sample every few seconds; judge completeness against the device's own rate.
        step = min(max(nominal_interval_s(clean["timestamp_utc"]), 1.0), 60.0) if len(clean) > 1 else 1.0
        return rows, WINDOW_S / step

    def _tick(self, rt: Runtime, now: pd.Timestamp) -> None:
        pid = rt.patient_id
        patient = self._check_calibration(self.db.get_patient(pid), now)
        rt.buffer = deque(x for x in rt.buffer if x[0] > now - pd.Timedelta(seconds=BUFFER_S))
        recs = [r for a, r in rt.buffer if a <= now]
        feats: dict = {}
        try:
            rows, expected = self._window(pid, recs, now)
            feats = compute_window_features(rows, patient["baseline"], window_end=now, expected_samples=expected)
            feats.update(patient_id=pid, window_end=now, timezone=patient["timezone"])
            status = self.predictor.predict(feats)
        except Exception:  # ML failure: "Data tidak tersedia", never green (PRD)
            log.exception("status computation failed for %s at %s", pid, now)
            status = {"patient_id": pid, "window_end": str(now), "risk_level": None, "reasons": [],
                      "model_version": None, "data_quality": None}
        level, prev = status["risk_level"], rt.prev_level
        notes = rt.dedup.update(now, level) if level else []

        if level == ELEVATED:
            rt.guide_step = 1 if prev != ELEVATED else min(rt.guide_step + 1, texts.GUIDE_STEPS)
        else:
            rt.guide_step = 0
        if level == "monitor" and prev not in ("monitor", ELEVATED):
            self.emit(pid, {"type": "greeting", "time": iso(now)}, "patient")
        self._track_episode(rt, patient, level, status["reasons"], now, resolved=any(n["type"] == "resolved" for n in notes))

        rt.status, rt.prev_level = status, level
        state = self._display_state(rt, patient, status, now)
        self.db.add_status(status, state, feats.get("hr_mean"))
        self.emit(pid, {"type": "status", **self._payload(rt, patient, now)}, "all")
        for n in notes:
            self.dispatch(pid, n["type"], now)

    def _track_episode(self, rt: Runtime, patient: dict, level: str | None, reasons: list[str], now: pd.Timestamp,
                       resolved: bool) -> None:
        """Therapist note (PRD F13): one per episode that reached red, written when it resolves."""
        if level in ("monitor", ELEVATED):
            ep = rt.episode or {"start": now, "highest": level, "reasons": []}
            if level == ELEVATED:
                ep["highest"] = ELEVATED
            ep["reasons"] += [r for r in reasons if r not in ep["reasons"]]
            rt.episode = ep
        if resolved and rt.episode and rt.episode["highest"] == ELEVATED:
            ep, tz = rt.episode, patient["timezone"]
            minutes = max(1, round((now - ep["start"]).total_seconds() / 60))
            summary = texts.therapist_summary(ep["start"].tz_convert(tz).strftime("%H:%M"),
                                              now.tz_convert(tz).strftime("%H:%M"), minutes, ep["highest"], ep["reasons"])
            self.db.add_note(rt.patient_id, iso(ep["start"]), iso(now), ep["highest"], texts.reason_texts(ep["reasons"]),
                             summary)
            rt.episode = None
        elif level in ("normal", None, "insufficient_data") and rt.episode and rt.episode["highest"] != ELEVATED:
            rt.episode = None

    # --- calibration (PRD F11) -------------------------------------------------------------

    def start_calibration(self, patient_id: str) -> pd.Timestamp:
        now = self.now_for(patient_id)
        self.db.update_patient(patient_id, calibration_started_at=iso(now))
        return now

    def finish_calibration(self, patient_id: str, now: pd.Timestamp | None = None) -> dict:
        """Personal baseline from the calm data received since calibration started. Raises ValueError."""
        patient = self.db.get_patient(patient_id)
        if not patient or not patient["calibration_started_at"]:
            raise ValueError("Kalibrasi belum dimulai")
        now = now if now is not None else self.now_for(patient_id)
        records = self.db.records_between(patient_id, patient["calibration_started_at"], iso(now))
        records = [{k: r.get(k) for k in RAW_COLUMNS} for r in records if isinstance(r, dict)]
        baseline = None
        if records:
            clean = clean_sensor_data(validate_sensor_data(pd.DataFrame.from_records(records), now=now))
            clean = clean[clean["patient_id"] == patient_id]
            if len(clean):
                _, baselines = build_feature_table(clean, "calibration")
                baseline = baselines.set_index("patient_id").to_dict("index").get(patient_id) if len(baselines) else None
        self.db.update_patient(patient_id, calibration_started_at=None)
        if not baseline or not baseline.get("valid"):
            self.emit(patient_id, {"type": "calibration", "result": "failed", "message": texts.CALIBRATION_FAILED},
                      "patient")
            raise ValueError(texts.CALIBRATION_FAILED)
        baseline["patient_id"] = patient_id
        self.db.update_patient(patient_id, baseline=baseline)
        self.predictor.reset(patient_id)  # streaks were measured against no baseline
        self.emit(patient_id, {"type": "calibration", "result": "done"}, "patient")
        return self.db.get_patient(patient_id)["baseline"]

    def _check_calibration(self, patient: dict, now: pd.Timestamp) -> dict:
        start = parse_ts(patient["calibration_started_at"])
        if start is not None and (now - start).total_seconds() >= BASELINE_CALIBRATION_S:
            try:
                self.finish_calibration(patient["patient_id"], now)
            except ValueError:
                pass  # the patient was told through the "calibration" event
            return self.db.get_patient(patient["patient_id"])
        return patient

    # --- what the app shows ----------------------------------------------------------------

    def _display_state(self, rt: Runtime | None, patient: dict, status: dict | None, now: pd.Timestamp) -> str:
        if patient["calibration_started_at"]:
            return "calibrating"
        if status is None:
            return "collecting" if rt is not None and rt.stream_started is not None else "waiting"
        level = status["risk_level"]
        if level is None:
            return "unavailable"
        window_end = pd.Timestamp(status["window_end"])
        if (now - window_end).total_seconds() > STALE_STATUS_S:
            return "unavailable"
        if level == "insufficient_data":
            if "no_baseline" in status["reasons"]:
                return "needs_calibration"
            if rt.stream_started is not None and (window_end - rt.stream_started).total_seconds() <= COLLECTING_S:
                return "collecting"
        return level

    def _payload(self, rt: Runtime | None, patient: dict, now: pd.Timestamp) -> dict:
        status = rt.status if rt else None
        state = self._display_state(rt, patient, status, now)
        base = status or {"patient_id": patient["patient_id"], "window_end": None, "risk_level": None, "reasons": [],
                          "model_version": None, "data_quality": None}
        remaining = None
        if state == "calibrating":
            remaining = BASELINE_CALIBRATION_S - (now - parse_ts(patient["calibration_started_at"])).total_seconds()
        guide_on = state == ELEVATED and rt is not None and rt.guide_step > 0
        return {**base,
                "display": texts.display(state, base["reasons"], remaining),
                "guide": {"active": guide_on, "step": rt.guide_step if guide_on else 0, "total": texts.GUIDE_STEPS},
                "heart_rate": rt.heart_rate if rt else None,
                "clock_now": iso(now)}

    def snapshot(self, patient_id: str) -> dict:
        """Current status as GET /patients/{id}/status returns it (before privacy filtering)."""
        patient = self.db.get_patient(patient_id)
        rt = self.runtimes.get(patient_id)
        return self._payload(rt, patient, self.now_for(patient_id))
