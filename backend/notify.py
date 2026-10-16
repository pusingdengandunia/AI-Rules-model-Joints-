"""Notifications: in-app feed, Expo push (app users) and WhatsApp templates (caregivers without the app).

Which notifications exist and when is decided upstream (AlertDeduplicator in the engine, or the
"Aku tidak aman" button). This module only fans one notification out to its recipients and
records every attempt in the notifications table.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from backend import texts
from backend.config import Settings
from backend.db import Database, iso

log = logging.getLogger("neuvana.notify")

# Yellow is never a push notification (PRD); "unsafe" has no rate limit.
AUDIENCE = {"alert": "caregivers", "resolved": "caregivers", "device_check": "all", "unsafe": "caregivers"}


class Notifier:
    def __init__(self, db: Database, settings: Settings, emit):
        self.db = db
        self.settings = settings
        self.emit = emit
        self.http: httpx.AsyncClient | None = None
        self.tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self.http = httpx.AsyncClient(timeout=10)

    async def stop(self) -> None:
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.http:
            await self.http.aclose()

    def dispatch(self, patient_id: str, kind: str, event_time: pd.Timestamp) -> None:
        patient = self.db.get_patient(patient_id)
        audience = AUDIENCE[kind]
        message = texts.notification_text(kind, patient["nickname"])
        when = iso(event_time)
        self.db.add_notification(patient_id, kind, audience, "in_app", None, message, when, "sent")
        event = {"type": "notification", "kind": kind, "message": message, "time": when}
        if kind == "alert":
            event["guide_step"] = 1
        self.emit(patient_id, event, audience)

        pushes = [(t, self.db.add_notification(patient_id, kind, audience, "push", t, message, when, "queued"))
                  for t in self.db.push_tokens(patient_id, audience)]
        whatsapp = []
        if audience in ("caregivers", "all"):
            whatsapp = [(m, self.db.add_notification(patient_id, kind, audience, "whatsapp", m["phone"], message, when,
                                                     "queued"))
                        for m in self.db.whatsapp_recipients(patient_id)]
        if not pushes and not whatsapp:
            return
        if self.settings.notify_mode == "dry_run":
            self.db.set_notification_status([i for _, i in pushes + whatsapp], "dry_run")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.db.set_notification_status([i for _, i in pushes + whatsapp], "skipped", "no event loop")
            return
        task = loop.create_task(self._send(patient, kind, message, pushes, whatsapp))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _send(self, patient: dict, kind: str, message: str, pushes: list, whatsapp: list) -> None:
        if pushes:
            await self._send_push(patient, kind, message, pushes)
        for member, nid in whatsapp:
            await self._send_whatsapp(patient, kind, member, nid)

    async def _send_push(self, patient: dict, kind: str, message: str, pushes: list) -> None:
        body = [{"to": token, "title": "Neuvana", "body": message, "sound": "default", "priority": "high",
                 "data": {"kind": kind, "patient_id": patient["patient_id"]}} for token, _ in pushes]
        headers = {"Accept": "application/json"}
        if self.settings.expo_access_token:
            headers["Authorization"] = f"Bearer {self.settings.expo_access_token}"
        try:
            resp = await self.http.post(self.settings.expo_push_url, json=body, headers=headers)
            resp.raise_for_status()
            tickets = resp.json().get("data", [])
        except Exception as exc:
            log.warning("expo push failed: %s", exc)
            self.db.set_notification_status([i for _, i in pushes], "failed", str(exc)[:300])
            return
        for (_, nid), ticket in zip(pushes, tickets):
            ok = ticket.get("status") == "ok"
            self.db.set_notification_status([nid], "sent" if ok else "failed",
                                            None if ok else str(ticket.get("message"))[:300])

    async def _send_whatsapp(self, patient: dict, kind: str, member: dict, nid: int) -> None:
        s = self.settings
        if not (s.wa_token and s.wa_phone_number_id):
            self.db.set_notification_status([nid], "skipped", "WhatsApp is not configured")
            return
        link = f"{s.public_web_url}/w/{member['web_token']}"
        body = {"messaging_product": "whatsapp", "to": member["phone"], "type": "template",
                "template": {"name": s.wa_templates[kind], "language": {"code": s.wa_language},
                             "components": [{"type": "body", "parameters": [
                                 {"type": "text", "text": patient["nickname"]}, {"type": "text", "text": link}]}]}}
        url = f"https://graph.facebook.com/{s.wa_api_version}/{s.wa_phone_number_id}/messages"
        try:
            resp = await self.http.post(url, json=body, headers={"Authorization": f"Bearer {s.wa_token}"})
            resp.raise_for_status()
            self.db.set_notification_status([nid], "sent")
        except Exception as exc:
            log.warning("whatsapp send failed: %s", exc)
            self.db.set_notification_status([nid], "failed", str(exc)[:300])
