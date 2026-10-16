"""Backend settings, read from environment variables (see backend/README.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIMEZONES = {"WIB": "Asia/Jakarta", "WITA": "Asia/Makassar", "WIT": "Asia/Jayapura"}


def _templates() -> dict[str, str]:
    return {
        "alert": os.environ.get("WA_TEMPLATE_ALERT", "neuvana_alert"),
        "resolved": os.environ.get("WA_TEMPLATE_RESOLVED", "neuvana_resolved"),
        "device_check": os.environ.get("WA_TEMPLATE_DEVICE_CHECK", "neuvana_device_check"),
        "unsafe": os.environ.get("WA_TEMPLATE_UNSAFE", "neuvana_unsafe"),
    }


@dataclass
class Settings:
    db_path: Path = ROOT / "backend_data" / "neuvana.db"
    # Seeds the simulator patients SIM01-SIM03 and the demo baselines a new account may borrow.
    baselines_path: Path = ROOT / "data" / "baselines" / "synthetic_baselines.json"
    public_web_url: str = "http://localhost:8081"   # Expo web app; links are <url>/w/<token>
    replay_speed: float = 30.0                      # default speed of simulator streams
    notify_mode: str = "live"                       # live | dry_run (record, never send)
    ingest_key: str | None = None                   # if set, POST /ingest needs X-Ingest-Key or the patient's token
    expo_push_url: str = "https://exp.host/--/api/v2/push/send"
    expo_access_token: str | None = None
    wa_token: str | None = None
    wa_phone_number_id: str | None = None
    wa_api_version: str = "v21.0"
    wa_language: str = "id"
    wa_templates: dict[str, str] = field(default_factory=_templates)
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    poll_interval_s: float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        s = cls()
        if env("NEUVANA_DB"):
            s.db_path = Path(env("NEUVANA_DB"))
        if env("NEUVANA_BASELINES"):
            s.baselines_path = Path(env("NEUVANA_BASELINES"))
        s.public_web_url = env("PUBLIC_WEB_URL", s.public_web_url).rstrip("/")
        s.replay_speed = float(env("REPLAY_SPEED", s.replay_speed))
        s.notify_mode = env("NOTIFY_MODE", s.notify_mode)
        s.ingest_key = env("INGEST_KEY") or None
        s.expo_access_token = env("EXPO_ACCESS_TOKEN") or None
        s.wa_token = env("WA_TOKEN") or None
        s.wa_phone_number_id = env("WA_PHONE_NUMBER_ID") or None
        s.wa_api_version = env("WA_API_VERSION", s.wa_api_version)
        s.wa_language = env("WA_LANGUAGE", s.wa_language)
        if env("CORS_ORIGINS"):
            s.cors_origins = [o.strip() for o in env("CORS_ORIGINS").split(",") if o.strip()]
        return s
