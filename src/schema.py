"""Single source of truth for the sensor data contract (Step 2).

Every stage (simulator, WESAD adapter, validation, cleaning, features, API) imports
column names, ranges and flags from here so they cannot drift apart.
"""

SCHEMA_VERSION = "1.0"

# --- Raw sensor record: one row = one measurement (1 Hz for both sources) -------------

REQUIRED_COLUMNS = ["patient_id", "timestamp", "heart_rate"]

# Optional: a device that does not provide a signal leaves the column out or null.
# Never fill an unavailable signal with a made-up value.
OPTIONAL_COLUMNS = [
    "session_id",   # recording session; derived from time gaps if absent
    "hrv_rmssd",    # ms, RMSSD over the device's own window (60 s here)
    "accel_x",      # g, instantaneous 1 Hz sample incl. gravity
    "accel_y",
    "accel_z",
    "eda",          # uS, extra signal (not in the spec table, kept nullable)
    "skin_temp",    # degC, extra signal (not in the spec table, kept nullable)
    "source",       # "simulator", "empatica_e4", ...
    "received_at",  # when the backend received the record; used for late-data checks
]

RAW_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

NUMERIC_COLUMNS = ["heart_rate", "hrv_rmssd", "accel_x", "accel_y", "accel_z", "eda", "skin_temp"]

# Technical plausibility limits for flagging. These are sensor sanity bounds,
# NOT clinical normal ranges or diagnostic criteria.
VALID_RANGES = {
    "heart_rate": (25, 240),
    "hrv_rmssd": (1, 300),
    "accel_x": (-16, 16),
    "accel_y": (-16, 16),
    "accel_z": (-16, 16),
    "eda": (0, 100),
    "skin_temp": (15, 45),
}

# Columns starting with this prefix are ground truth for evaluation only.
# They must never be used as model input features.
GROUND_TRUTH_PREFIX = "gt_"

# --- Validation (Step 3) ---------------------------------------------------------------

# Primary quality_flag, in priority order (first matching issue wins).
QUALITY_FLAGS = [
    "invalid_timestamp",   # unparseable, before MIN_TIMESTAMP, or in the future
    "missing_patient_id",
    "duplicate",           # same patient_id + timestamp as a later-received record
    "invalid_value",       # a sensor value is non-numeric or outside VALID_RANGES
    "missing_value",       # heart_rate is null
    "late",                # received more than LATE_THRESHOLD_S after measurement
    "out_of_order",        # arrived after a record with a later timestamp
    "good",
]

# Records with these flags are kept in the validated dataset (traceability)
# but excluded from the clean time series.
UNUSABLE_FLAGS = {"invalid_timestamp", "missing_patient_id", "duplicate"}

MIN_TIMESTAMP = "2000-01-01T00:00:00Z"
FUTURE_TOLERANCE_S = 300
LATE_THRESHOLD_S = 120

# --- Preprocessing (Step 4) ------------------------------------------------------------

SESSION_GAP_S = 1800          # a gap longer than this starts a new session
GAP_FACTOR = 3                # interval > GAP_FACTOR * nominal interval counts as a gap
WINDOW_S = 300                # 5-minute windows (spec example)
WINDOW_STEP_S = 60            # sliding step; windows overlap by 4 minutes
MIN_VALID_RATIO = 0.5         # below this the window is insufficient_data (>=150 s of HR per 5 min)
MAX_STALENESS_S = 60          # newest valid HR older than this at window end -> insufficient_data
                              # (sensor disconnected *now* must never look like a normal reading)

DATA_QUALITY_VALUES = ["good", "insufficient_data"]

# --- Features (Step 5) -----------------------------------------------------------------

# Spec features, in the fixed order training and inference must both use.
MODEL_FEATURES = [
    "hr_mean",
    "hr_std",
    "hr_delta_pct",
    "hrv_rmssd_mean",
    "hrv_delta_pct",
    "accel_magnitude_std",
    "valid_sample_ratio",
]

FEATURE_TABLE_COLUMNS = [
    "patient_id",
    "session_id",
    "source",
    "window_start",
    "window_end",
    *MODEL_FEATURES,
    "hrv_valid_ratio",
    "accel_valid_ratio",
    "eda_mean",
    "skin_temp_mean",
    "eda_delta",
    "skin_temp_delta",
    "n_samples",
    "staleness_s",
    "data_quality",
    "is_baseline_period",
]

# Personal baseline: median over good windows of a calibration period.
BASELINE_MIN_WINDOWS = 5
BASELINE_CALIBRATION_S = 1800  # default calibration period: first 30 min of the first session
