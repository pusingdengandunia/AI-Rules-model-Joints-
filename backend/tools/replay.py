"""Replay a simulator session into the backend, paced by the data's own clock.

Unlike src/simulator/replay.py (one row per tick of wall time), records are sent in arrival
order (received_at) and spaced by their arrival times / speed, so a sensor gap in the file
is also a pause on the wire and the app turns gray live. The backend gets the speed in the
X-Replay-Speed header and keeps its clock for that patient running during such pauses.

  python -m backend.tools.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30
  python -m backend.tools.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --patient P-1A2B3C4D
  python -m backend.tools.replay data/raw/synthetic/SIM01_normal.csv --speed 0     # as fast as possible
"""

from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from src.simulator.replay import to_message


def load_messages(csv: str, patient: str | None = None) -> list[dict]:
    """Rows of a simulator file as JSON messages (gt_* dropped), sorted by arrival time."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False).replace({"": None})
    msgs = [to_message(r, keep_gt=False) for r in df.to_dict("records")]
    if patient:
        for m in msgs:
            m["patient_id"] = patient
    return sorted(msgs, key=lambda m: arrival(m) or pd.Timestamp.max.tz_localize("UTC"))


def arrival(msg: dict) -> pd.Timestamp | None:
    for key in ("received_at", "timestamp"):
        v = msg.get(key)
        if isinstance(v, str):
            ts = pd.to_datetime(v, utc=True, errors="coerce", format="ISO8601")
            if not pd.isna(ts):
                return ts
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--speed", type=float, default=30.0, help="data seconds per wall second; 0 = as fast as possible")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--patient", help="send as this patient_id (e.g. the demo account) instead of SIM01")
    ap.add_argument("--ingest-key", help="X-Ingest-Key, if the backend requires one")
    args = ap.parse_args()

    msgs = load_messages(args.csv, args.patient)
    headers = {"X-Replay-Speed": str(args.speed)}
    if args.ingest_key:
        headers["X-Ingest-Key"] = args.ingest_key
    url = args.url.rstrip("/") + "/ingest"
    with httpx.Client(timeout=30) as client:
        if args.speed <= 0:
            for i in range(0, len(msgs), 500):
                client.post(url, json=msgs[i:i + 500], headers=headers).raise_for_status()
        else:
            t0_data = arrival(msgs[0])
            t0_wall = time.monotonic()
            batch: list[dict] = []
            for m in msgs:
                a = arrival(m)
                due = t0_wall + ((a - t0_data).total_seconds() / args.speed if a is not None else 0)
                if due > time.monotonic() and batch:
                    client.post(url, json=batch, headers=headers).raise_for_status()
                    batch = []
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                batch.append(m)
            if batch:
                client.post(url, json=batch, headers=headers).raise_for_status()
    print(f"sent {len(msgs)} records from {args.csv}")


if __name__ == "__main__":
    main()
