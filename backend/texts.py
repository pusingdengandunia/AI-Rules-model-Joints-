"""Every user-facing text the backend produces (PRD "Status, teks, dan pemberitahuan").

Soft wording agreed by the team. Words in BANNED_WORDS must never appear in the app;
tests/test_backend_texts.py checks every text here against it, and any AI-written text
must pass `is_safe_text` before it is sent.
"""

from __future__ import annotations

import re

ELEVATED = "elevated_physiological_anomaly"

# state -> (color, text). States are the four risk levels plus the special display states.
DISPLAY = {
    "normal": ("green", "Pola tubuh stabil"),
    "monitor": ("yellow", "Pola tubuh berubah, pantau"),
    ELEVATED: ("red", "Pola tubuh menyimpang, perlu perhatian"),
    "insufficient_data": ("gray", "Data sensor tidak cukup, periksa perangkat"),
    "waiting": ("neutral", "Menunggu data dari jam tangan"),
    "collecting": ("neutral", "Mengumpulkan data"),
    "calibrating": ("neutral", "Mengenali pola tubuhmu"),
    "needs_calibration": ("gray", "Belum ada pola pribadi — perlu kalibrasi"),
    "unavailable": ("gray", "Data tidak tersedia"),
}

# Handoff Bagian 5. None = not shown.
REASONS = {
    "hr_above_baseline": "Detak jantung di atas pola biasanya",
    "hrv_below_baseline": "Variabilitas detak jantung di bawah pola biasanya",
    "at_rest": "Terjadi saat sedang diam",
    "activity": "Sedang beraktivitas fisik",
    "post_activity": "Masa pemulihan setelah aktivitas",
    "sleep_disruption_candidate": "Kemungkinan gangguan tidur — pantau",
    "sleep_disruption": "Gangguan pola tidur",
    "high_night_restlessness": "Gelisah saat jam istirahat",
    "sustained_night_arousal": "Detak jantung tinggi saat jam istirahat",
    "model_stress_pattern": "Pola mirip stres (eksperimen)",
    "no_baseline": "Belum ada pola pribadi — perlu kalibrasi",
    "insufficient_samples": "Data sensor tidak cukup",
    "stale_data": "Sensor tidak mengirim data",
    "missing_feature": "Data sensor tidak lengkap",
    "model_agrees": None,
    "fallback_rules": None,
}

GUIDE_STEPS = 6  # content of the 6-step caregiver guide lives in the app

CALIBRATION_FAILED = "Belum cukup data tenang untuk mengenali pola tubuhmu. Coba kalibrasi lagi."

BANNED_WORDS = ["ptsd", "serangan", "diagnos", "risiko", "%", "sembuh", "obat", "sertralin", "sertraline",
                "fluoksetin", "fluoxetine", "paroksetin", "paroxetine", "prazosin", "venlafaxine", "venlafaksin"]


def is_safe_text(text: str) -> bool:
    low = text.lower()
    return not any(w in low for w in BANNED_WORDS)


def reason_texts(reasons: list[str]) -> list[str]:
    out = []
    for r in reasons:
        text = REASONS.get(r.split(":", 1)[0], None)
        if text and text not in out:
            out.append(text)
    return out


def display(state: str, reasons: list[str], calibration_remaining_s: float | None = None) -> dict:
    color, text = DISPLAY[state]
    out = {"state": state, "color": color, "text": text, "reasons_text": reason_texts(reasons)}
    if state == "calibrating" and calibration_remaining_s is not None:
        out["calibration_remaining_s"] = max(0, int(calibration_remaining_s))
    return out


def notification_text(kind: str, name: str) -> str:
    return {
        "alert": f"Pola tubuh {name} menyimpang, perlu perhatian. Buka panduan langkah 1.",
        "resolved": f"Pola tubuh {name} kembali stabil",
        "device_check": "Data sensor tidak cukup — periksa perangkat",
        "unsafe": f"{name} sedang merasa tidak aman. Hubungi {name} sekarang.",
    }[kind]


def therapist_summary(start_local: str, end_local: str, minutes: int, highest: str, reasons: list[str]) -> str:
    text = (f"Pola tubuh berubah dari pukul {start_local} sampai {end_local} (sekitar {minutes} menit). "
            f"Status tertinggi: {DISPLAY[highest][1]}.")
    shown = reason_texts(reasons)
    return text + (" Yang terlihat: " + "; ".join(shown) + "." if shown else "")


_NAME_OK = re.compile(r"^[\w .'-]{1,40}$", re.UNICODE)


def valid_nickname(name: str) -> bool:
    return bool(_NAME_OK.match(name.strip()))
