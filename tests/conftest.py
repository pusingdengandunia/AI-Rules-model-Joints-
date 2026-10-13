import numpy as np
import pandas as pd
import pytest


def make_records(n: int = 600, patient: str = "P1", start: str = "2026-01-05T08:00:00Z", hr: float = 70.0,
                 session: str | None = "P1_s") -> pd.DataFrame:
    """n seconds of clean 1 Hz records as a device would send them (timestamps as ISO strings)."""
    ts = pd.date_range(start, periods=n, freq="1s")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "patient_id": patient,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heart_rate": np.round(hr + rng.normal(0, 1, n), 1),
        "hrv_rmssd": np.round(45 + rng.normal(0, 2, n), 1),
        "accel_x": rng.normal(0, 0.01, n),
        "accel_y": rng.normal(0, 0.01, n),
        "accel_z": 1 + rng.normal(0, 0.01, n),
        "source": "test",
        "received_at": (ts + pd.Timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    if session:
        df["session_id"] = session
    return df


@pytest.fixture
def records():
    return make_records()
