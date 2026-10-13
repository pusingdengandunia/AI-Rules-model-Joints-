"""Notification de-duplication (spec §8: alerts are not sent repeatedly without control)."""

import pandas as pd

from src.ml.simulate_stream import AlertDeduplicator

T0 = pd.Timestamp("2026-01-05T08:00:00Z")
E, M, N, I = "elevated_physiological_anomaly", "monitor", "normal", "insufficient_data"


def feed(levels, start_min=0):
    d = AlertDeduplicator()
    out = []
    for i, lvl in enumerate(levels):
        out += [n["type"] for n in d.update(T0 + pd.Timedelta(minutes=start_min + i), lvl)]
    return out


def test_one_alert_per_episode():
    assert feed([N, M, M, E, E, E, E, N]) == ["alert", "resolved"]


def test_flapping_within_cooldown_is_not_realerted():
    assert feed([N, E, M, E, M, E, N]).count("alert") == 1


def test_new_episode_after_cooldown_is_alerted():
    assert feed([N, E, N] + [N] * 40 + [E]).count("alert") == 2


def test_device_check_once_per_outage():
    assert feed([N, I, I, I, I, N, N, I, I, I]) == ["device_check", "device_check"]


def test_no_device_check_during_warm_up():
    assert feed([I, I, I, I, N]) == []


def test_single_insufficient_tick_is_not_reported():
    assert feed([N, I, N, I, N]) == []
