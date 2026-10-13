# Laporan kualitas data — synthetic

Schema 1.0. Window 5 menit, langkah 60 s, `good` jika `valid_sample_ratio` ≥ 0.5. Dibuat oleh `src/pipeline.py`.

> Data sintetis. Hanya untuk menguji pipeline dan demo; bukan data klinis.

## 1. Ringkasan

| Metrik | Nilai |
|---|---|
| Record masuk | 139869 |
| Record usable (masuk clean series) | 139746 |
| Record dikeluarkan (duplicate / timestamp invalid / tanpa patient_id) | 123 |
| Pasien | 3 |
| Sesi | 24 |
| Gap terdeteksi | 6 |
| Window total | 2244 |
| Window `good` | 2196 |
| Window `insufficient_data` | 48 |

## 2. Quality flag per record

`quality_flag` = issue dengan prioritas tertinggi. Satu record bisa punya beberapa issue.

| quality_flag | records |
|---|---|
| good | 137402 |
| missing_value | 2047 |
| late | 180 |
| duplicate | 99 |
| out_of_order | 99 |
| invalid_value | 18 |
| invalid_timestamp | 15 |
| missing_patient_id | 9 |


Semua issue:

| issue | records |
|---|---|
| missing_value | 2047 |
| out_of_order | 279 |
| late | 180 |
| duplicate | 99 |
| invalid_value | 18 |
| invalid_timestamp | 15 |
| duplicate_conflict | 9 |
| out_of_range:heart_rate | 9 |
| naive_timezone | 9 |
| missing_patient_id | 9 |
| out_of_range:hrv_rmssd | 3 |
| out_of_range:accel_x | 3 |
| non_numeric:heart_rate | 3 |


### Deteksi gangguan yang disuntikkan (skenario `bad_quality`)

Baris = gangguan yang disuntikkan simulator. Kolom = flag yang diberikan validator.

| gt_injected | duplicate | good | invalid_timestamp | invalid_value | late | missing_patient_id |
|---|---|---|---|---|---|---|
| duplicate | 99 | 0 | 0 | 0 | 0 | 0 |
| invalid_timestamp | 0 | 0 | 15 | 0 | 0 | 0 |
| invalid_value | 0 | 0 | 0 | 18 | 0 | 0 |
| late | 0 | 0 | 0 | 0 | 180 | 0 |
| missing_patient_id | 0 | 0 | 0 | 0 | 0 | 9 |
| naive_timezone | 0 | 9 | 0 | 0 | 0 | 0 |


Terdeteksi dengan issue yang sesuai: **330/330**.


## 3. Missing data per sesi

`hr_missing_ratio` dihitung dari record yang ada. Periode tanpa record sama sekali terlihat di kolom `gaps`.

| session_id | records | duration_min | nominal_interval_s | gaps | too_dense | hr_missing_ratio | windows | insufficient | mean_valid_ratio | mean_hrv_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| SIM01_normal | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM01_hr_elevation | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM01_hrv_drop | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM01_high_activity | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM01_missing_data | 3480 | 60.00 | 1.00 | 1 | 0 | 0.11 | 56 | 9 | 0.86 | 0.84 |
| SIM01_bad_quality | 3592 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM01_mixed_demo | 7110 | 120.00 | 1.00 | 1 | 0 | 0.04 | 116 | 7 | 0.94 | 0.94 |
| SIM01_sleep_disruption | 18000 | 300.00 | 1.00 | 0 | 0 | 0.00 | 296 | 0 | 1.00 | 1.00 |
| SIM02_normal | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM02_hr_elevation | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM02_hrv_drop | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM02_high_activity | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM02_missing_data | 3480 | 60.00 | 1.00 | 1 | 0 | 0.10 | 56 | 9 | 0.86 | 0.84 |
| SIM02_bad_quality | 3592 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM02_mixed_demo | 7110 | 120.00 | 1.00 | 1 | 0 | 0.04 | 116 | 7 | 0.94 | 0.93 |
| SIM02_sleep_disruption | 18000 | 300.00 | 1.00 | 0 | 0 | 0.00 | 296 | 0 | 1.00 | 1.00 |
| SIM03_normal | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM03_hr_elevation | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM03_hrv_drop | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM03_high_activity | 3600 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM03_missing_data | 3480 | 60.00 | 1.00 | 1 | 0 | 0.10 | 56 | 9 | 0.86 | 0.84 |
| SIM03_bad_quality | 3592 | 60.00 | 1.00 | 0 | 0 | 0.00 | 56 | 0 | 1.00 | 1.00 |
| SIM03_mixed_demo | 7110 | 120.00 | 1.00 | 1 | 0 | 0.04 | 116 | 7 | 0.94 | 0.93 |
| SIM03_sleep_disruption | 18000 | 300.00 | 1.00 | 0 | 0 | 0.00 | 296 | 0 | 1.00 | 1.00 |


## 4. Gap (periode tanpa record)

Total 6 gap, durasi total 636 s, terpanjang 121 s. Top 15:

| session_id | gap_start | gap_end | duration_s |
|---|---|---|---|
| SIM01_missing_data | 2026-01-09 02:14:59+00:00 | 2026-01-09 02:17:00+00:00 | 121.00 |
| SIM02_missing_data | 2026-01-09 03:14:59+00:00 | 2026-01-09 03:17:00+00:00 | 121.00 |
| SIM03_missing_data | 2026-01-09 04:14:59+00:00 | 2026-01-09 04:17:00+00:00 | 121.00 |
| SIM01_mixed_demo | 2026-01-11 03:04:59+00:00 | 2026-01-11 03:06:30+00:00 | 91.00 |
| SIM02_mixed_demo | 2026-01-11 04:04:59+00:00 | 2026-01-11 04:06:30+00:00 | 91.00 |
| SIM03_mixed_demo | 2026-01-11 05:04:59+00:00 | 2026-01-11 05:06:30+00:00 | 91.00 |


## 5. Baseline pribadi

Metode `calibration`. Median `hr_mean` / `hrv_rmssd_mean` dari window `good` dalam periode kalibrasi. Window periode kalibrasi ditandai `is_baseline_period=True`; keluarkan dari evaluasi model.

| patient_id | hr_baseline | hrv_baseline | n_windows | valid | period_start | period_end |
|---|---|---|---|---|---|---|
| SIM01 | 64.30 | 47.05 | 26 | True | 2026-01-05 02:00:00+00:00 | 2026-01-05 02:30:00+00:00 |
| SIM02 | 69.24 | 45.70 | 26 | True | 2026-01-05 03:00:00+00:00 | 2026-01-05 03:30:00+00:00 |
| SIM03 | 66.33 | 46.33 | 26 | True | 2026-01-05 04:00:00+00:00 | 2026-01-05 04:30:00+00:00 |


## Keterbatasan

- Batas plausibilitas di `src/schema.py` adalah batas teknis sensor, bukan rentang klinis.

- Missing data tidak diisi (tidak ada `fillna(0)` atau interpolasi). Window tanpa sampel cukup menjadi `insufficient_data`.

- Window bertumpuk 4 menit. Split train/test harus per pasien, bukan per window acak.

- Kolom `gt_*` hanya kunci evaluasi. Jangan dipakai sebagai fitur model.
