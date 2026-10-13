# Evaluasi rules_v1 pada data sintetis (Target A)

> Data sintetis: hasil ini membuktikan logika rules dan pipeline bekerja sesuai desain skenario, bukan akurasi klinis.

## Konfigurasi

```json
{
  "hr_delta_pct_high": 15.0,
  "hrv_delta_pct_low": -40.0,
  "activity_accel_std": 0.08,
  "post_activity_cooldown_s": 600,
  "persist_windows": 3
}
```

## Hasil per split

Kriteria lulus: recall arousal→elevated 100%, recall hrv_drop→monitor ≥ 80%, recall sleep_disruption→elevated 100%, brief_wake tidak pernah elevated, 0 alert di skenario normal/high_activity/missing_data/bad_quality, 100% window gap = insufficient_data.

| split | arousal_recall | hrv_drop_recall | sleep_recall | brief_wake_ok | mean_latency_s | false_alarm_episodes | fa_per_hour | no_alert_scenario_alerts | gap_insufficient | hours | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tune (seed 42, SIM01-02) | 1.00 | 1.00 | 1.00 | 1.00 | 402.00 | 0 | 0.00 | 0 | 1.00 | 26.00 | ya |
| test (seed 42, SIM03) | 1.00 | 1.00 | 1.00 | 1.00 | 396.00 | 0 | 0.00 | 0 | 1.00 | 13.00 | ya |
| holdout (seed 7, SIM01-03) | 1.00 | 1.00 | 1.00 | 1.00 | 412.00 | 0 | 0.00 | 0 | 1.00 | 39.00 | ya |


`fa_per_hour` = episode false alarm per jam data. Episode = alert berurutan di luar event (event + recovery + 120 s toleransi).

## Detail per event

| split | session_id | event | start_s | max_level | detected | latency_s |
|---|---|---|---|---|---|---|
| tune (seed 42, SIM01-02) | SIM01_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 300.00 |
| tune (seed 42, SIM01-02) | SIM01_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 420.00 |
| tune (seed 42, SIM01-02) | SIM01_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 300.00 |
| tune (seed 42, SIM01-02) | SIM01_mixed_demo | hrv_drop | 4500 | elevated_physiological_anomaly | True | 420.00 |
| tune (seed 42, SIM01-02) | SIM01_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 600.00 |
| tune (seed 42, SIM01-02) | SIM02_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 300.00 |
| tune (seed 42, SIM01-02) | SIM02_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 360.00 |
| tune (seed 42, SIM01-02) | SIM02_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 300.00 |
| tune (seed 42, SIM01-02) | SIM02_mixed_demo | hrv_drop | 4500 | elevated_physiological_anomaly | True | 360.00 |
| tune (seed 42, SIM01-02) | SIM02_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 660.00 |
| tune (seed 42, SIM01-02) | SIM01_sleep_disruption | brief_wake | 12600 | monitor | True |  |
| tune (seed 42, SIM01-02) | SIM02_sleep_disruption | brief_wake | 12600 | monitor | True |  |
| test (seed 42, SIM03) | SIM03_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 300.00 |
| test (seed 42, SIM03) | SIM03_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 420.00 |
| test (seed 42, SIM03) | SIM03_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 300.00 |
| test (seed 42, SIM03) | SIM03_mixed_demo | hrv_drop | 4500 | elevated_physiological_anomaly | True | 360.00 |
| test (seed 42, SIM03) | SIM03_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 600.00 |
| test (seed 42, SIM03) | SIM03_sleep_disruption | brief_wake | 12600 | monitor | True |  |
| holdout (seed 7, SIM01-03) | SIM01_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 360.00 |
| holdout (seed 7, SIM01-03) | SIM01_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 420.00 |
| holdout (seed 7, SIM01-03) | SIM01_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 360.00 |
| holdout (seed 7, SIM01-03) | SIM01_mixed_demo | hrv_drop | 4500 | elevated_physiological_anomaly | True | 300.00 |
| holdout (seed 7, SIM01-03) | SIM01_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 600.00 |
| holdout (seed 7, SIM01-03) | SIM02_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 300.00 |
| holdout (seed 7, SIM01-03) | SIM02_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 420.00 |
| holdout (seed 7, SIM01-03) | SIM02_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 300.00 |
| holdout (seed 7, SIM01-03) | SIM02_mixed_demo | hrv_drop | 4500 | monitor | True | 480.00 |
| holdout (seed 7, SIM01-03) | SIM02_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 600.00 |
| holdout (seed 7, SIM01-03) | SIM03_hr_elevation | arousal | 1800 | elevated_physiological_anomaly | True | 300.00 |
| holdout (seed 7, SIM01-03) | SIM03_hrv_drop | hrv_drop | 1500 | elevated_physiological_anomaly | True | 420.00 |
| holdout (seed 7, SIM01-03) | SIM03_mixed_demo | arousal | 2700 | elevated_physiological_anomaly | True | 360.00 |
| holdout (seed 7, SIM01-03) | SIM03_mixed_demo | hrv_drop | 4500 | elevated_physiological_anomaly | True | 360.00 |
| holdout (seed 7, SIM01-03) | SIM03_sleep_disruption | sleep_disruption | 7200 | elevated_physiological_anomaly | True | 600.00 |
| holdout (seed 7, SIM01-03) | SIM01_sleep_disruption | brief_wake | 12600 | monitor | True |  |
| holdout (seed 7, SIM01-03) | SIM02_sleep_disruption | brief_wake | 12600 | monitor | True |  |
| holdout (seed 7, SIM01-03) | SIM03_sleep_disruption | brief_wake | 12600 | monitor | True |  |


## Tuning (grid search, hanya split tune)

36 dari 144 konfigurasi lulus kriteria di split tune. Top 10:

| hr_delta_pct_high | hrv_delta_pct_low | activity_accel_std | post_activity_cooldown_s | persist_windows | arousal_recall | hrv_drop_recall | false_alarm_episodes | mean_latency_s | gap_insufficient_ratio | no_alert_total | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15.00 | -40.00 | 0.08 | 600 | 3 | 1.00 | 1.00 | 0 | 402.00 | 1.00 | 0 | True |
| 15.00 | -40.00 | 0.10 | 600 | 3 | 1.00 | 1.00 | 0 | 402.00 | 1.00 | 0 | True |
| 15.00 | -40.00 | 0.15 | 600 | 3 | 1.00 | 1.00 | 0 | 402.00 | 1.00 | 0 | True |
| 20.00 | -40.00 | 0.08 | 600 | 3 | 1.00 | 1.00 | 0 | 420.00 | 1.00 | 0 | True |
| 20.00 | -40.00 | 0.10 | 600 | 3 | 1.00 | 1.00 | 0 | 420.00 | 1.00 | 0 | True |
| 20.00 | -40.00 | 0.15 | 600 | 3 | 1.00 | 1.00 | 0 | 420.00 | 1.00 | 0 | True |
| 25.00 | -40.00 | 0.08 | 600 | 3 | 1.00 | 1.00 | 0 | 444.00 | 1.00 | 0 | True |
| 25.00 | -40.00 | 0.10 | 600 | 3 | 1.00 | 1.00 | 0 | 444.00 | 1.00 | 0 | True |
| 25.00 | -40.00 | 0.15 | 600 | 3 | 1.00 | 1.00 | 0 | 444.00 | 1.00 | 0 | True |
| 15.00 | -40.00 | 0.08 | 300 | 3 | 1.00 | 1.00 | 0 | 402.00 | 1.00 | 0 | True |
