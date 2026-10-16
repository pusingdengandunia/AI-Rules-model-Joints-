"""Backend vs the reference timelines: every SIM01 session replayed into the engine must give
the same status per minute and the same notifications as reports/stream/<sesi>_rules_timeline.csv."""

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.db import Database
from backend.engine import StatusEngine
from backend.tools.replay import arrival, load_messages

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "synthetic"
REF = ROOT / "reports" / "stream"
BASELINES = ROOT / "data" / "baselines" / "synthetic_baselines.json"
SESSIONS = ["SIM01_mixed_demo", "SIM01_normal", "SIM01_hr_elevation", "SIM01_hrv_drop", "SIM01_high_activity",
            "SIM01_missing_data", "SIM01_bad_quality", "SIM01_sleep_disruption"]

pytestmark = pytest.mark.skipif(not BASELINES.exists(),
                                reason="run: python -m src.pipeline --source synthetic --regenerate")


@pytest.mark.parametrize("session", SESSIONS)
def test_replay_matches_reference_timeline(tmp_path, session):
    db = Database(tmp_path / "t.db")
    for b in json.loads(BASELINES.read_text(encoding="utf-8")):
        db.ensure_patient(b["patient_id"], baseline=b, is_simulated=True)
    sent = []
    engine = StatusEngine(db, dispatch=lambda pid, kind, t: sent.append((t, kind)))

    msgs = load_messages(RAW / f"{session}.csv")
    for i in range(0, len(msgs), 250):
        engine.ingest(msgs[i:i + 250], replay_speed=30)
    last = max(a for a in map(arrival, msgs) if a is not None)
    engine.flush("SIM01", last.ceil("60s"))

    ref = pd.read_csv(REF / f"{session}_rules_timeline.csv", keep_default_na=False)
    got = db.history("SIM01", limit=100000)
    assert [r["window_end"] for r in got] == [str(pd.Timestamp(t)) for t in ref["now"]]
    assert [r["risk_level"] for r in got] == ref["risk_level"].tolist()
    assert [";".join(r["reasons"]) for r in got] == ref["reasons"].tolist()
    expected = [(pd.Timestamp(t), kind) for t, kinds in zip(ref["now"], ref["notifications"]) if kinds
                for kind in kinds.split(";")]
    assert sent == expected
