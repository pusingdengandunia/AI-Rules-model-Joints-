"""API flows of the Neuvana backend with a fake clock (wall-clock patients, like Health Connect)."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import texts
from backend.config import Settings
from backend.main import create_app

BASELINE = {"patient_id": "SIM01", "hr_baseline": 70.0, "hrv_baseline": 45.0, "eda_baseline": None,
            "skin_temp_baseline": None, "n_windows": 20, "valid": True, "method": "calibration"}
START = pd.Timestamp("2026-01-05T02:00:00Z")  # 09:00 WIB, daytime rules


class Clock:
    def __init__(self):
        self.wall, self.mono = START, 1000.0

    def advance(self, seconds: float):
        self.wall += pd.Timedelta(seconds=seconds)
        self.mono += seconds

    def __call__(self):
        return self.wall, self.mono


@pytest.fixture
def env(tmp_path):
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps([BASELINE]), encoding="utf-8")
    settings = Settings(db_path=tmp_path / "neuvana.db", baselines_path=baselines, notify_mode="dry_run",
                        poll_interval_s=3600, public_web_url="https://neuvana.example")
    with TestClient(create_app(settings)) as client:
        clock = Clock()
        client.app.state.engine.clock = clock
        yield client, clock


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def new_patient(client, nickname="Rani", demo_baseline="SIM01"):
    r = client.post("/accounts", json={"role": "patient", "nickname": nickname, "timezone": "WIB",
                                       "demo_baseline": demo_baseline})
    assert r.status_code == 201, r.text
    body = r.json()
    return body["patient"]["patient_id"], body["token"]


def new_caregiver(client, patient_token, pid, nickname="Ibu"):
    token = client.post("/accounts", json={"role": "caregiver", "nickname": nickname}).json()["token"]
    code = client.post(f"/circles/{pid}/invites", headers=auth(patient_token)).json()["code"]
    r = client.post("/circles/join", json={"code": code}, headers=auth(token))
    assert r.status_code == 201, r.text
    return token, r.json()["member_id"]


def new_web_member(client, patient_token, pid, phone="+6281200000000"):
    code = client.post(f"/circles/{pid}/invites", headers=auth(patient_token)).json()["code"]
    r = client.post("/circles/join", json={"code": code, "name": "Kakak", "phone": phone})
    assert r.status_code == 201, r.text
    return r.json()


def poll(client):
    client.portal.call(client.app.state.engine.poll)


def feed(client, clock, pid, seconds, hr, step=10, seed=0):
    """Phone forwarding Health Connect data: every `step` s a batch of 1 Hz samples, then the minute ticks."""
    rng = np.random.default_rng(seed)
    for _ in range(int(seconds // step)):
        clock.advance(step)
        recs = [{"patient_id": pid, "timestamp": (clock.wall - pd.Timedelta(seconds=step - 1 - i)).isoformat(),
                 "heart_rate": round(float(hr + rng.normal(0, 1)), 1), "source": "health_connect"}
                for i in range(step)]
        assert client.post("/ingest", json=recs).status_code == 202
        poll(client)


def status(client, pid, token):
    r = client.get(f"/patients/{pid}/status", headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def kinds(client, pid, token):
    return [n["kind"] for n in client.get(f"/patients/{pid}/notifications", headers=auth(token)).json()]


def test_health(env):
    client, _ = env
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["predictor_mode"] == "rules"


def test_full_episode_on_wall_clock(env):
    client, clock = env
    pid, ptoken = new_patient(client)
    ctoken, _ = new_caregiver(client, ptoken, pid)
    client.post("/devices/push-token", json={"token": "ExponentPushToken[ibu]"}, headers=auth(ctoken))
    assert status(client, pid, ptoken)["display"]["state"] == "waiting"

    feed(client, clock, pid, 60, hr=70)
    assert status(client, pid, ptoken)["display"]["state"] == "collecting"  # never the scary gray at start
    feed(client, clock, pid, 300, hr=70)
    s = status(client, pid, ptoken)
    assert s["risk_level"] == "normal" and s["display"]["color"] == "green"
    assert s["heart_rate"]["value"] in range(65, 76)

    feed(client, clock, pid, 600, hr=95)
    s = status(client, pid, ctoken)
    assert s["risk_level"] == "elevated_physiological_anomaly" and s["display"]["color"] == "red"
    assert s["guide"]["active"] and 1 <= s["guide"]["step"] <= 6
    assert "Detak jantung di atas pola biasanya" in s["display"]["reasons_text"]
    assert kinds(client, pid, ctoken) == ["alert"]

    feed(client, clock, pid, 600, hr=70)
    assert status(client, pid, ptoken)["risk_level"] == "normal"
    assert kinds(client, pid, ctoken) == ["alert", "resolved"]
    notes = client.get(f"/patients/{pid}/notes", headers=auth(ptoken)).json()
    assert len(notes) == 1 and notes[0]["highest_level"] == "elevated_physiological_anomaly"
    assert all(texts.is_safe_text(t) for t in [notes[0]["summary"], *notes[0]["reasons"]])

    for _ in range(5):  # watch taken off: nothing arrives, the minutes keep ticking
        clock.advance(60)
        poll(client)
    s = status(client, pid, ptoken)
    assert s["risk_level"] == "insufficient_data" and s["display"]["color"] == "gray"
    assert kinds(client, pid, ctoken) == ["alert", "resolved", "device_check"]
    assert kinds(client, pid, ptoken) == ["device_check"]  # the patient never gets the red alert push

    pushes = [n for n in client.app.state.db.notifications(pid, "caregiver", channel="push")]
    assert [(n["kind"], n["status"]) for n in pushes] == [("alert", "dry_run"), ("resolved", "dry_run"),
                                                         ("device_check", "dry_run")]

    clock.advance(600)  # no ticks at all (e.g. the scheduler stalled): old status must not look current
    assert status(client, pid, ptoken)["display"]["state"] == "unavailable"


def test_no_baseline_asks_for_calibration(env):
    client, clock = env
    pid, ptoken = new_patient(client, demo_baseline=None)
    feed(client, clock, pid, 420, hr=70)
    s = status(client, pid, ptoken)
    assert s["risk_level"] == "insufficient_data" and s["display"]["state"] == "needs_calibration"


def test_calibration_builds_baseline(env):
    client, clock = env
    pid, ptoken = new_patient(client, demo_baseline=None)
    r = client.post(f"/patients/{pid}/calibration", json={"action": "start"}, headers=auth(ptoken))
    assert r.json()["status"] == "calibrating"
    feed(client, clock, pid, 600, hr=72, step=30)
    s = status(client, pid, ptoken)
    assert s["display"]["state"] == "calibrating" and 0 < s["display"]["calibration_remaining_s"] < 1800
    feed(client, clock, pid, 1260, hr=72, step=30)  # the tick at 30 min finishes it automatically
    p = client.get(f"/patients/{pid}", headers=auth(ptoken)).json()
    assert p["has_baseline"] and p["calibration_started_at"] is None
    feed(client, clock, pid, 120, hr=72, step=30)
    assert status(client, pid, ptoken)["risk_level"] == "normal"


def test_calibration_without_data_is_refused(env):
    client, _ = env
    pid, ptoken = new_patient(client, demo_baseline=None)
    client.post(f"/patients/{pid}/calibration", json={"action": "start"}, headers=auth(ptoken))
    r = client.post(f"/patients/{pid}/calibration", json={"action": "finish"}, headers=auth(ptoken))
    assert r.status_code == 409


def test_ml_failure_is_never_green(env, monkeypatch):
    client, clock = env
    pid, ptoken = new_patient(client)
    feed(client, clock, pid, 400, hr=70)

    def boom(_):
        raise RuntimeError("ml down")

    monkeypatch.setattr(client.app.state.engine.predictor, "predict", boom)
    feed(client, clock, pid, 60, hr=70)
    s = status(client, pid, ptoken)
    assert s["risk_level"] is None and s["display"]["state"] == "unavailable" and s["display"]["color"] != "green"


def test_bad_records_never_crash(env):
    client, _ = env
    pid, _ = new_patient(client)
    junk = [42, None, "x", {}, {"patient_id": pid}, {"patient_id": pid, "timestamp": "not a time", "heart_rate": "abc"},
            {"patient_id": pid, "timestamp": "2026-01-05T02:00:00Z", "heart_rate": 70},
            {"patient_id": pid, "timestamp": "2026-01-05T02:00:00Z", "heart_rate": 71},  # duplicate
            {"patient_id": {"nested": 1}, "timestamp": 5, "heart_rate": [1]}]
    r = client.post("/ingest", json=junk)
    assert r.status_code == 202 and r.json()["accepted"] == len(junk)
    flags = [x["quality_flag"] for x in r.json()["results"]]
    assert flags[0] == "invalid_record" and "duplicate" in flags
    assert client.post("/ingest", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_simulator_stream_uses_replay_clock(env):
    client, _ = env
    t0 = pd.Timestamp("2026-01-11T02:00:00Z")
    recs = [{"patient_id": "SIM01", "timestamp": (t0 + pd.Timedelta(seconds=i)).isoformat(), "heart_rate": 70.0,
             "received_at": (t0 + pd.Timedelta(seconds=i + 1)).isoformat(), "source": "simulator"} for i in range(400)]
    client.post("/ingest", json=recs, headers={"X-Replay-Speed": "30"})
    s = client.get("/patients/SIM01/status").json()  # simulated patient: open for the demo
    assert s["risk_level"] == "normal" and s["window_end"].startswith("2026-01-11 02:06")


def test_circle_privacy_and_revoke(env):
    client, clock = env
    pid, ptoken = new_patient(client)
    ctoken, member_id = new_caregiver(client, ptoken, pid)
    web = new_web_member(client, ptoken, pid)
    assert web["web_url"].startswith("https://neuvana.example/w/")

    # one account, one role
    code = client.post(f"/circles/{pid}/invites", headers=auth(ptoken)).json()["code"]
    assert client.post("/circles/join", json={"code": code}, headers=auth(ptoken)).status_code == 403
    # used invite cannot be reused; caregiver cannot read another patient
    other, other_token = new_patient(client, "Budi")
    assert client.get(f"/patients/{other}/status", headers=auth(ctoken)).status_code == 403
    assert client.get(f"/patients/{pid}/notes", headers=auth(ctoken)).status_code == 403

    client.patch(f"/patients/{pid}/settings", json={"share_reasons": False, "share_heart_rate": False},
                 headers=auth(ptoken))
    feed(client, clock, pid, 400, hr=70)
    feed(client, clock, pid, 240, hr=95)
    mine, theirs = status(client, pid, ptoken), status(client, pid, ctoken)
    assert mine["reasons"] and mine["heart_rate"]
    assert theirs["reasons"] == [] and theirs["display"]["reasons_text"] == [] and "heart_rate" not in theirs
    page = client.get(f"/web/{web['web_token']}").json()
    assert page["patient"]["nickname"] == "Rani" and page["status"]["reasons"] == []

    members = client.get(f"/circles/{pid}/members", headers=auth(ptoken)).json()
    assert {m["kind"] for m in members} == {"app", "web"}
    client.delete(f"/circles/{pid}/members/{member_id}", headers=auth(ptoken))
    client.delete(f"/circles/{pid}/members/{web['member_id']}", headers=auth(ptoken))
    assert client.get(f"/patients/{pid}/status", headers=auth(ctoken)).status_code == 403
    assert client.get(f"/web/{web['web_token']}").status_code == 404


def test_unsafe_notifies_every_press(env):
    client, _ = env
    pid, ptoken = new_patient(client)
    ctoken, _ = new_caregiver(client, ptoken, pid)
    new_web_member(client, ptoken, pid)
    client.patch(f"/patients/{pid}/settings", json={"emergency_contacts": [{"name": "Ibu", "phone": "0812"}]},
                 headers=auth(ptoken))
    for _ in range(2):
        r = client.post(f"/patients/{pid}/unsafe", headers=auth(ptoken))
        assert r.json()["emergency_contacts"] == [{"name": "Ibu", "phone": "0812"}]
    assert kinds(client, pid, ctoken) == ["unsafe", "unsafe"]
    wa = client.app.state.db.notifications(pid, "caregiver", channel="whatsapp")
    assert [(n["kind"], n["recipient"], n["status"]) for n in wa] == [("unsafe", "+6281200000000", "dry_run")] * 2
    assert client.post(f"/patients/{pid}/unsafe", headers=auth(ctoken)).status_code == 403


def test_websocket_live_updates(env):
    client, clock = env
    pid, ptoken = new_patient(client)
    ctoken, member_id = new_caregiver(client, ptoken, pid)
    with client.websocket_connect(f"/ws/patients/{pid}?token={ctoken}") as ws:
        first = ws.receive_json()
        assert first["type"] == "status" and first["display"]["state"] == "waiting"
        feed(client, clock, pid, 60, hr=70)
        seen = set()
        while "status" not in seen:
            seen.add(ws.receive_json()["type"])
        assert "heart_rate" in seen
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/patients/{pid}?token=wrong") as ws:
            ws.receive_json()


def test_texts_avoid_banned_words():
    shown = [t for _, t in texts.DISPLAY.values()] + [t for t in texts.REASONS.values() if t]
    shown += [texts.notification_text(k, "Rani") for k in ("alert", "resolved", "device_check", "unsafe")]
    shown += [texts.CALIBRATION_FAILED]
    assert all(texts.is_safe_text(t) for t in shown)
    assert not texts.is_safe_text("Risiko PTSD 80%")
