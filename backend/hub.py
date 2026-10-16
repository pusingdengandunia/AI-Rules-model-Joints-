"""WebSocket fan-out of live events, with each viewer's privacy filter applied."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections import defaultdict
from dataclasses import dataclass

from fastapi import WebSocket

from backend.db import Database

log = logging.getLogger("neuvana.hub")


def view_for(event: dict, role: str, settings: dict) -> dict | None:
    """What one viewer may see. The patient sees everything; caregivers only what was shared.

    Returns None when the event must not reach this viewer at all.
    """
    if role == "patient":
        return event
    if event.get("type") == "heart_rate" and not settings["share_heart_rate"]:
        return None
    out = copy.deepcopy(event)
    if not settings["share_heart_rate"]:
        out.pop("heart_rate", None)
    if not settings["share_reasons"]:
        if "reasons" in out:
            out["reasons"] = []
        if isinstance(out.get("display"), dict):
            out["display"]["reasons_text"] = []
    return out


@dataclass(eq=False)
class Connection:
    ws: WebSocket
    role: str                 # "patient" | "caregiver"
    member_id: str | None = None


class Hub:
    def __init__(self, db: Database):
        self.db = db
        self.conns: dict[str, set[Connection]] = defaultdict(set)
        self.tasks: set[asyncio.Task] = set()

    def add(self, patient_id: str, conn: Connection) -> None:
        self.conns[patient_id].add(conn)

    def remove(self, patient_id: str, conn: Connection) -> None:
        self.conns[patient_id].discard(conn)

    def publish(self, patient_id: str, event: dict, audience: str = "all") -> None:
        """audience: all | patient | caregivers. Safe to call from sync code on the event loop."""
        targets = [c for c in self.conns.get(patient_id, ())
                   if audience == "all" or (audience == "patient") == (c.role == "patient")]
        if not targets:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        settings = self.db.get_patient(patient_id)["settings"]
        for c in targets:
            msg = view_for(event, c.role, settings)
            if msg is not None:
                task = loop.create_task(self._send(patient_id, c, msg))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

    async def _send(self, patient_id: str, conn: Connection, msg: dict) -> None:
        try:
            await conn.ws.send_json(msg)
        except Exception:
            self.remove(patient_id, conn)

    async def drop_member(self, patient_id: str, member_id: str) -> None:
        """Access revoked: close that caregiver's live connections now."""
        for c in [c for c in self.conns.get(patient_id, ()) if c.member_id == member_id]:
            self.remove(patient_id, c)
            try:
                await c.ws.close(code=4403)
            except Exception:
                pass
