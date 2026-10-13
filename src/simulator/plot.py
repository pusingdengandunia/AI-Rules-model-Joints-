"""Preview plot for one synthetic session: signals stacked, ground-truth windows shaded."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHADE = {
    "arousal": "#d1495b",
    "hrv_drop": "#edae49",
    "activity": "#00798c",
    "light_activity": "#8ecae6",
    "recovery": "#cccccc",
    "missing": "#555555",
    "artifact": "#6a4c93",
    "sleep_disruption": "#9d4edd",
    "brief_wake": "#c8b6ff",
}
SIGNALS = [
    ("heart_rate", "HR (bpm)"),
    ("hrv_rmssd", "RMSSD (ms)"),
    ("eda", "EDA (µS)"),
    ("skin_temp", "Skin temp (°C)"),
    ("accel_magnitude", "|accel| (g)"),
]


def _spans(t: np.ndarray, states: np.ndarray):
    """Yield (state, start_t, end_t) runs of identical shaded state."""
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            if states[start] in SHADE:
                yield states[start], t[start], t[i - 1] + 1
            start = i


def plot_session(df: pd.DataFrame, path: Path) -> None:
    df = df[df.gt_injected.eq("") | df.gt_injected.isna()].sort_values("gt_t_s")
    df = df.assign(
        heart_rate=pd.to_numeric(df.heart_rate, errors="coerce"),
        accel_magnitude=np.sqrt(df.accel_x**2 + df.accel_y**2 + df.accel_z**2),
    )
    # Reindex to the full second grid so disconnected periods show as breaks.
    df = df.set_index("gt_t_s").reindex(range(int(df.gt_t_s.max()) + 1))
    t = df.index.to_numpy()
    states = df.gt_state.fillna("missing").to_numpy()
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(12, 9), sharex=True)
    for ax, (col, label) in zip(axes, SIGNALS):
        ax.plot(t / 60, df[col], lw=0.8, color="#222222")
        ax.set_ylabel(label, fontsize=8)
        for state, a, b in _spans(t, states):
            ax.axvspan(a / 60, b / 60, color=SHADE[state], alpha=0.25, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in SHADE.values()]
    fig.legend(handles, SHADE.keys(), loc="upper right", ncol=7, fontsize=8)
    axes[-1].set_xlabel("minutes")
    fig.suptitle(f"{df.session_id.dropna().iloc[0]}  (SYNTHETIC)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
