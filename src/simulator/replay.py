"""Replay a raw session file as a live sensor stream, for end-to-end tests up to the caregiver app.

Each row becomes one JSON message in file (arrival) order. Ground-truth columns
(gt_*) are stripped by default so the system under test cannot see the answers.

Examples:
  python -m src.simulator.replay data/raw/synthetic/SIM01_hr_elevation.csv --speed 60
  python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --post http://localhost:8000/ingest
  python -m src.simulator.replay data/raw/synthetic/SIM01_bad_quality.csv --speed 0 > stream.jsonl
"""

import argparse
import json
import math
import sys
import time
import urllib.request

import pandas as pd

from src.schema import GROUND_TRUTH_PREFIX, NUMERIC_COLUMNS


def to_message(row: dict, keep_gt: bool) -> dict:
    msg = {}
    for k, v in row.items():
        if k.startswith(GROUND_TRUTH_PREFIX) and not keep_gt:
            continue
        if isinstance(v, float) and math.isnan(v):
            v = None  # missing sample -> JSON null, as a real device payload would send
        elif k in NUMERIC_COLUMNS and v is not None:
            try:
                v = float(v)
            except ValueError:
                pass  # malformed value (bad_quality scenario) is sent as-is for the backend to reject
        msg[k] = v
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--speed", type=float, default=1.0, help="playback multiplier; 0 = as fast as possible")
    ap.add_argument("--post", help="POST each message as JSON to this URL instead of printing")
    ap.add_argument("--keep-gt", action="store_true", help="include ground-truth columns in messages")
    args = ap.parse_args()

    # Read everything as text so malformed values reach the receiver unchanged.
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False).replace({"": None})
    delay = 0 if args.speed <= 0 else 1.0 / args.speed
    for row in df.to_dict("records"):
        body = json.dumps(to_message(row, args.keep_gt))
        if args.post:
            req = urllib.request.Request(args.post, data=body.encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        else:
            sys.stdout.write(body + "\n")
            sys.stdout.flush()
        if delay:
            time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass  # consumer closed the stream (e.g. piped into head)
