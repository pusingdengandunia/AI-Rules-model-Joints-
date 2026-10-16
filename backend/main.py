"""Neuvana MVP API (PRD "API dan data utama").

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend import texts
from backend.config import TIMEZONES, Settings
from backend.db import Database, iso
from backend.engine import StatusEngine
from backend.hub import Connection, Hub, view_for
from backend.notify import Notifier

log = logging.getLogger("neuvana")
INVITE_TTL = pd.Timedelta(days=7)


# --- request bodies ----------------------------------------------------------------------

def _timezone(value: str) -> str:
    tz = TIMEZONES.get(value.strip().upper(), value.strip())
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("timezone must be WIB, WITA, WIT or an IANA name")
    return tz


def _nickname(value: str) -> str:
    if not texts.valid_nickname(value):
        raise ValueError("nickname: 1-40 letters, digits, spaces, . ' -")
    return value.strip()


class AccountIn(BaseModel):
    role: Literal["patient", "caregiver"]
    nickname: str
    timezone: str = "WIB"
    demo_baseline: str | None = Field(None, description="Borrow the baseline of SIM01/SIM02/SIM03 (demo only)")

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v):
        return _timezone(v)

    @field_validator("nickname")
    @classmethod
    def _nick(cls, v):
        return _nickname(v)


class Contact(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    phone: str = Field(min_length=3, max_length=30)


class SettingsIn(BaseModel):
    nickname: str | None = None
    timezone: str | None = None
    share_reasons: bool | None = None
    share_heart_rate: bool | None = None
    notify_me_first: bool | None = None
    emergency_contacts: list[Contact] | None = Field(None, max_length=10)

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v):
        return None if v is None else _timezone(v)

    @field_validator("nickname")
    @classmethod
    def _nick(cls, v):
        return None if v is None else _nickname(v)


class CalibrationIn(BaseModel):
    action: Literal["start", "finish", "cancel"]


class JoinIn(BaseModel):
    code: str = Field(min_length=4, max_length=12)
    name: str | None = Field(None, max_length=60)
    phone: str | None = Field(None, max_length=30, description="WhatsApp number, for caregivers without the app")


class PushTokenIn(BaseModel):
    token: str = Field(min_length=10, max_length=300)


# --- app ---------------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.db_path)
        _seed_simulated_patients(db, settings)
        hub = Hub(db)
        notifier = Notifier(db, settings, hub.publish)
        engine = StatusEngine(db, emit=hub.publish, dispatch=notifier.dispatch, replay_speed=settings.replay_speed)
        app.state.db, app.state.hub, app.state.notifier, app.state.engine = db, hub, notifier, engine
        await notifier.start()
        poller = asyncio.create_task(_poll_forever(engine, settings.poll_interval_s))
        try:
            yield
        finally:
            poller.cancel()
            await notifier.stop()
            db.close()

    app = FastAPI(title="Neuvana API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.state.settings = settings
    _routes(app, settings)
    return app


async def _poll_forever(engine: StatusEngine, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            engine.poll()
        except Exception:
            log.exception("poll failed")


def _seed_simulated_patients(db: Database, settings: Settings) -> None:
    if not settings.baselines_path.exists():
        log.warning("no demo baselines at %s (run: python -m src.pipeline --source synthetic --regenerate)",
                    settings.baselines_path)
        return
    for b in json.loads(settings.baselines_path.read_text(encoding="utf-8")):
        db.ensure_patient(b["patient_id"], nickname=b["patient_id"], baseline=b, is_simulated=True)


def _demo_baseline(settings: Settings, source: str) -> dict:
    if settings.baselines_path.exists():
        for b in json.loads(settings.baselines_path.read_text(encoding="utf-8")):
            if b["patient_id"] == source:
                return b
    raise HTTPException(422, f"no demo baseline {source!r}")


def _routes(app: FastAPI, settings: Settings) -> None:
    def db(request: Request) -> Database:
        return request.app.state.db

    def engine(request: Request) -> StatusEngine:
        return request.app.state.engine

    def optional_account(request: Request, authorization: str | None = Header(None)) -> dict | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        acc = request.app.state.db.account_by_token(authorization[7:].strip())
        if acc is None:
            raise HTTPException(401, "invalid token")
        return acc

    def account(acc: dict | None = Depends(optional_account)) -> dict:
        if acc is None:
            raise HTTPException(401, "missing bearer token")
        return acc

    def patient_or_404(request: Request, patient_id: str) -> dict:
        p = request.app.state.db.get_patient(patient_id)
        if p is None:
            raise HTTPException(404, "unknown patient")
        return p

    def viewer_role(request: Request, patient_id: str, acc: dict | None) -> str:
        """patient | caregiver; 403 otherwise. Simulator patients (synthetic data) are open for the demo."""
        p = patient_or_404(request, patient_id)
        if acc and acc["role"] == "patient" and acc["patient_id"] == patient_id:
            return "patient"
        if acc and acc["role"] == "caregiver":
            m = request.app.state.db.active_membership(acc["account_id"])
            if m and m["patient_id"] == patient_id:
                return "caregiver"
        if p["is_simulated"] and request.app.state.db.patient_account(patient_id) is None:
            return "patient"
        raise HTTPException(403, "no access to this patient")

    def owner(request: Request, patient_id: str, acc: dict) -> dict:
        if acc["role"] != "patient" or acc["patient_id"] != patient_id:
            raise HTTPException(403, "only the patient can do this")
        return patient_or_404(request, patient_id)

    def public_patient(p: dict) -> dict:
        return {k: p[k] for k in ("patient_id", "nickname", "timezone", "settings", "emergency_contacts")} | {
            "has_baseline": bool(p["baseline"] and p["baseline"].get("valid")),
            "calibration_started_at": p["calibration_started_at"]}

    # --- service ---------------------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request):
        e = engine(request)
        return {"status": "ok", "predictor_mode": e.predictor.mode, "model_version": e.predictor.model_version,
                "active_patients": len(e.runtimes), "time": iso(pd.Timestamp.now(tz="UTC"))}

    @app.get("/meta")
    async def meta():
        """Texts and codes the app needs (status table, reasons, guide length)."""
        return {"display": {k: {"color": c, "text": t} for k, (c, t) in texts.DISPLAY.items()},
                "reasons": {k: v for k, v in texts.REASONS.items() if v}, "guide_steps": texts.GUIDE_STEPS,
                "timezones": TIMEZONES}

    # --- accounts --------------------------------------------------------------------------

    @app.post("/accounts", status_code=201)
    async def create_account(body: AccountIn, request: Request):
        d = db(request)
        patient = None
        if body.role == "patient":
            baseline = None
            if body.demo_baseline:
                baseline = {**_demo_baseline(settings, body.demo_baseline), "borrowed_from": body.demo_baseline}
            patient = d.ensure_patient(d.new_patient_id(), nickname=body.nickname, timezone=body.timezone,
                                       baseline=baseline)
        acc, token = d.create_account(body.role, body.nickname, patient["patient_id"] if patient else None)
        return {"account": acc, "token": token, "patient": public_patient(patient) if patient else None}

    @app.get("/me")
    async def me(request: Request, acc: dict = Depends(account)):
        d = db(request)
        if acc["role"] == "patient":
            return {"account": acc, "patient": public_patient(d.get_patient(acc["patient_id"]))}
        m = d.active_membership(acc["account_id"])
        linked = d.get_patient(m["patient_id"]) if m else None
        return {"account": acc, "membership": m,
                "patient": {"patient_id": linked["patient_id"], "nickname": linked["nickname"]} if linked else None}

    @app.post("/devices/push-token", status_code=204)
    async def push_token(body: PushTokenIn, request: Request, acc: dict = Depends(account)):
        db(request).set_push_token(acc["account_id"], body.token)

    # --- sensor data -----------------------------------------------------------------------

    @app.post("/ingest", status_code=202)
    async def ingest(request: Request, payload=Body(...), x_replay_speed: float | None = Header(None),
                     x_ingest_key: str | None = Header(None), acc: dict | None = Depends(optional_account)):
        """One sensor record, or a list of them (Health Connect syncs in batches)."""
        records = payload if isinstance(payload, list) else [payload]
        if settings.ingest_key and x_ingest_key != settings.ingest_key:
            pids = {r.get("patient_id") for r in records if isinstance(r, dict)}
            if not (acc and acc["role"] == "patient" and pids <= {acc["patient_id"]}):
                raise HTTPException(401, "X-Ingest-Key or the patient's own token required")
        results = engine(request).ingest(records, replay_speed=x_replay_speed)
        return {"accepted": len(results), "results": results if isinstance(payload, list) else results[0]}

    # --- status ----------------------------------------------------------------------------

    @app.get("/patients/{patient_id}/status")
    async def status(patient_id: str, request: Request, acc: dict | None = Depends(optional_account)):
        role = viewer_role(request, patient_id, acc)
        p = db(request).get_patient(patient_id)
        return view_for(engine(request).snapshot(patient_id), role, p["settings"])

    @app.get("/patients/{patient_id}/history")
    async def history(patient_id: str, request: Request, limit: int = 180,
                      acc: dict | None = Depends(optional_account)):
        role = viewer_role(request, patient_id, acc)
        p = db(request).get_patient(patient_id)
        rows = db(request).history(patient_id, min(max(limit, 1), 2000))
        return [view_for(r, role, p["settings"]) for r in rows]

    @app.get("/patients/{patient_id}/notifications")
    async def notifications(patient_id: str, request: Request, acc: dict | None = Depends(optional_account)):
        role = viewer_role(request, patient_id, acc)
        return db(request).notifications(patient_id, role)

    @app.post("/patients/{patient_id}/session/reset")
    async def reset_session(patient_id: str, request: Request, acc: dict | None = Depends(optional_account)):
        """New recording session: forget streaks and buffered data ("Mengumpulkan data" again)."""
        viewer = viewer_role(request, patient_id, acc)
        if viewer != "patient":
            raise HTTPException(403, "only the patient can do this")
        engine(request).reset(patient_id)
        return {"reset": True}

    # --- patient profile, privacy, calibration, unsafe -------------------------------------------

    @app.get("/patients/{patient_id}")
    async def get_patient(patient_id: str, request: Request, acc: dict = Depends(account)):
        return public_patient(owner(request, patient_id, acc))

    @app.patch("/patients/{patient_id}/settings")
    async def update_settings(patient_id: str, body: SettingsIn, request: Request, acc: dict = Depends(account)):
        p = owner(request, patient_id, acc)
        fields = {}
        if body.nickname is not None:
            fields["nickname"] = body.nickname
        if body.timezone is not None:
            fields["timezone"] = body.timezone
        if body.emergency_contacts is not None:
            fields["emergency_contacts"] = [c.model_dump() for c in body.emergency_contacts]
        flags = {k: getattr(body, k) for k in ("share_reasons", "share_heart_rate", "notify_me_first")
                 if getattr(body, k) is not None}
        if flags:
            fields["settings"] = {**p["settings"], **flags}
        return public_patient(db(request).update_patient(patient_id, **fields))

    @app.post("/patients/{patient_id}/calibration")
    async def calibration(patient_id: str, body: CalibrationIn, request: Request, acc: dict = Depends(account)):
        owner(request, patient_id, acc)
        e = engine(request)
        if body.action == "start":
            started = e.start_calibration(patient_id)
            return {"status": "calibrating", "started_at": iso(started), "duration_s": 1800}
        if body.action == "cancel":
            db(request).update_patient(patient_id, calibration_started_at=None)
            return {"status": "cancelled"}
        try:
            baseline = e.finish_calibration(patient_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"status": "done", "baseline": baseline}

    @app.post("/patients/{patient_id}/unsafe")
    async def unsafe(patient_id: str, request: Request, acc: dict = Depends(account)):
        """'Aku tidak aman': tell the circle every time it is pressed. The app shows the
        emergency contacts from its own copy without waiting for this call (PRD F9)."""
        p = owner(request, patient_id, acc)
        request.app.state.notifier.dispatch(patient_id, "unsafe", pd.Timestamp.now(tz="UTC"))
        return {"notified": True, "emergency_contacts": p["emergency_contacts"]}

    @app.get("/patients/{patient_id}/notes")
    async def notes(patient_id: str, request: Request, acc: dict = Depends(account)):
        owner(request, patient_id, acc)
        return db(request).notes(patient_id)

    # --- support circle ----------------------------------------------------------------------

    @app.post("/circles/{patient_id}/invites", status_code=201)
    async def invite(patient_id: str, request: Request, acc: dict = Depends(account)):
        owner(request, patient_id, acc)
        inv = db(request).create_invite(patient_id, INVITE_TTL)
        return {"code": inv["code"], "expires_at": inv["expires_at"],
                "link": f"{settings.public_web_url}/join/{inv['code']}"}

    @app.post("/circles/join", status_code=201)
    async def join(body: JoinIn, request: Request, acc: dict | None = Depends(optional_account)):
        """With a caregiver token: link the app account. Without: create a web-link membership."""
        d = db(request)
        if acc and acc["role"] != "caregiver":
            raise HTTPException(403, "one account holds one role: patients cannot join a circle")
        if acc and d.active_membership(acc["account_id"]):
            raise HTTPException(409, "this account already accompanies someone")
        patient_id = d.use_invite(body.code)
        if patient_id is None:
            raise HTTPException(404, "invite code invalid, used or expired")
        patient = d.get_patient(patient_id)
        if acc:
            m = d.add_member(patient_id, body.name or acc["nickname"], acc["account_id"], body.phone, None)
            return {"member_id": m["member_id"], "patient": {"patient_id": patient_id, "nickname": patient["nickname"]}}
        token = secrets.token_urlsafe(24)
        m = d.add_member(patient_id, body.name, None, body.phone, token)
        return {"member_id": m["member_id"], "web_token": token, "web_url": f"{settings.public_web_url}/w/{token}",
                "patient": {"nickname": patient["nickname"]}}

    @app.get("/circles/{patient_id}/members")
    async def members(patient_id: str, request: Request, acc: dict = Depends(account)):
        owner(request, patient_id, acc)
        return [{k: m[k] for k in ("member_id", "name", "phone", "joined_at", "revoked_at")}
                | {"kind": "app" if m["account_id"] else "web"} for m in db(request).members(patient_id)]

    @app.delete("/circles/{patient_id}/members/{member_id}")
    async def revoke(patient_id: str, member_id: str, request: Request, acc: dict = Depends(account)):
        owner(request, patient_id, acc)
        m = db(request).revoke_member(patient_id, member_id)
        if m is None or m["patient_id"] != patient_id:
            raise HTTPException(404, "unknown member")
        await request.app.state.hub.drop_member(patient_id, member_id)
        return {"member_id": member_id, "revoked_at": m["revoked_at"]}

    # --- caregiver web page (no app, no login) -------------------------------------------------

    @app.get("/web/{token}")
    async def web(token: str, request: Request):
        d = db(request)
        m = d.active_member_by_web_token(token)
        if m is None:
            raise HTTPException(404, "link invalid or revoked")
        p = d.get_patient(m["patient_id"])
        return {"patient": {"patient_id": p["patient_id"], "nickname": p["nickname"]},
                "status": view_for(engine(request).snapshot(p["patient_id"]), "caregiver", p["settings"]),
                "notifications": d.notifications(p["patient_id"], "caregiver", limit=20),
                "ws_path": f"/ws/patients/{p['patient_id']}?token={token}"}

    # --- live updates --------------------------------------------------------------------------

    @app.websocket("/ws/patients/{patient_id}")
    async def live(websocket: WebSocket, patient_id: str, token: str = ""):
        app_ = websocket.app
        d: Database = app_.state.db
        p = d.get_patient(patient_id)
        role, member_id = None, None
        acc = d.account_by_token(token) if token else None
        if p is None:
            pass
        elif acc and acc["role"] == "patient" and acc["patient_id"] == patient_id:
            role = "patient"
        elif acc and acc["role"] == "caregiver":
            m = d.active_membership(acc["account_id"])
            if m and m["patient_id"] == patient_id:
                role, member_id = "caregiver", m["member_id"]
        elif token and (m := d.active_member_by_web_token(token)) and m["patient_id"] == patient_id:
            role, member_id = "caregiver", m["member_id"]
        elif p["is_simulated"] and d.patient_account(patient_id) is None:
            role = "patient"
        if role is None:
            await websocket.close(code=4403)
            return
        await websocket.accept()
        conn = Connection(websocket, role, member_id)
        hub: Hub = app_.state.hub
        hub.add(patient_id, conn)
        try:
            snap = app_.state.engine.snapshot(patient_id)
            await websocket.send_json(view_for({"type": "status", **snap}, role, d.get_patient(patient_id)["settings"]))
            while True:
                await websocket.receive_text()  # client pings; nothing to read
        except WebSocketDisconnect:
            pass
        finally:
            hub.remove(patient_id, conn)


app = create_app()
