# Laporan kualitas data — wesad

Schema 1.0. Window 5 menit, langkah 60 s, `good` jika `valid_sample_ratio` ≥ 0.5. Dibuat oleh `src/pipeline.py`.

> WESAD: stres akut yang dipicu di lab pada subjek sehat. Label bukan episode PTSD.

## 1. Ringkasan

| Metrik | Nilai |
|---|---|
| Record masuk | 86868 |
| Record usable (masuk clean series) | 86868 |
| Record dikeluarkan (duplicate / timestamp invalid / tanpa patient_id) | 0 |
| Pasien | 15 |
| Sesi | 15 |
| Gap terdeteksi | 0 |
| Window total | 1375 |
| Window `good` | 707 |
| Window `insufficient_data` | 668 |

## 2. Quality flag per record

`quality_flag` = issue dengan prioritas tertinggi. Satu record bisa punya beberapa issue.

| quality_flag | records |
|---|---|
| missing_value | 43719 |
| good | 43149 |


Semua issue:

| issue | records |
|---|---|
| missing_value | 43719 |


## 3. Missing data per sesi

`hr_missing_ratio` dihitung dari record yang ada. Periode tanpa record sama sekali terlihat di kolom `gaps`.

| session_id | records | duration_min | nominal_interval_s | gaps | too_dense | hr_missing_ratio | windows | insufficient | mean_valid_ratio | mean_hrv_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| WESAD_S10_lab | 5496 | 91.60 | 1.00 | 0 | 0 | 0.38 | 87 | 30 | 0.62 | 0.62 |
| WESAD_S11_lab | 5233 | 87.20 | 1.00 | 0 | 0 | 0.53 | 82 | 39 | 0.48 | 0.36 |
| WESAD_S13_lab | 5537 | 92.30 | 1.00 | 0 | 0 | 0.56 | 87 | 55 | 0.45 | 0.33 |
| WESAD_S14_lab | 5548 | 92.40 | 1.00 | 0 | 0 | 0.32 | 88 | 12 | 0.69 | 0.66 |
| WESAD_S15_lab | 5252 | 87.50 | 1.00 | 0 | 0 | 0.56 | 83 | 56 | 0.45 | 0.34 |
| WESAD_S16_lab | 5631 | 93.80 | 1.00 | 0 | 0 | 0.48 | 89 | 35 | 0.53 | 0.34 |
| WESAD_S17_lab | 5920 | 98.60 | 1.00 | 0 | 0 | 0.60 | 94 | 59 | 0.41 | 0.19 |
| WESAD_S2_lab | 6079 | 101.30 | 1.00 | 0 | 0 | 0.54 | 96 | 59 | 0.46 | 0.30 |
| WESAD_S3_lab | 6493 | 108.20 | 1.00 | 0 | 0 | 0.61 | 104 | 81 | 0.39 | 0.04 |
| WESAD_S4_lab | 6423 | 107.00 | 1.00 | 0 | 0 | 0.38 | 102 | 29 | 0.63 | 0.35 |
| WESAD_S5_lab | 6258 | 104.30 | 1.00 | 0 | 0 | 0.65 | 99 | 63 | 0.36 | 0.15 |
| WESAD_S6_lab | 7071 | 117.80 | 1.00 | 0 | 0 | 0.47 | 113 | 47 | 0.54 | 0.25 |
| WESAD_S7_lab | 5238 | 87.30 | 1.00 | 0 | 0 | 0.50 | 83 | 31 | 0.51 | 0.31 |
| WESAD_S8_lab | 5466 | 91.10 | 1.00 | 0 | 0 | 0.47 | 86 | 33 | 0.54 | 0.43 |
| WESAD_S9_lab | 5223 | 87.00 | 1.00 | 0 | 0 | 0.49 | 82 | 39 | 0.51 | 0.40 |


## 4. Gap (periode tanpa record)

_(tidak ada gap)_


## 5. Baseline pribadi

Metode `label`. Median `hr_mean` / `hrv_rmssd_mean` dari window `good` dalam periode kalibrasi. Window periode kalibrasi ditandai `is_baseline_period=True`; keluarkan dari evaluasi model.

| patient_id | hr_baseline | hrv_baseline | n_windows | valid | period_start | period_end |
|---|---|---|---|---|---|---|
| WESAD_S10 | 95.72 |  | 3 | False | 2017-07-25 07:40:00+00:00 | 2017-07-25 07:47:00+00:00 |
| WESAD_S11 | 75.12 | 51.36 | 8 | True | 2017-07-25 11:36:00+00:00 | 2017-07-25 11:48:00+00:00 |
| WESAD_S13 | 84.99 | 50.70 | 5 | True | 2017-08-08 11:40:00+00:00 | 2017-08-08 11:53:00+00:00 |
| WESAD_S14 | 72.51 | 51.15 | 7 | True | 2017-08-09 07:34:00+00:00 | 2017-08-09 07:45:00+00:00 |
| WESAD_S15 | 80.87 |  | 4 | False | 2017-08-10 07:43:00+00:00 | 2017-08-10 07:51:00+00:00 |
| WESAD_S16 | 63.97 | 87.67 | 7 | True | 2017-08-10 12:25:00+00:00 | 2017-08-10 12:36:00+00:00 |
| WESAD_S17 | 64.67 | 90.48 | 7 | True | 2017-08-11 07:43:00+00:00 | 2017-08-11 07:54:00+00:00 |
| WESAD_S2 | 71.45 | 56.58 | 7 | True | 2017-05-22 07:48:00+00:00 | 2017-05-22 07:59:00+00:00 |
| WESAD_S3 | 54.95 | 142.41 | 6 | True | 2017-05-24 11:33:00+00:00 | 2017-05-24 11:46:00+00:00 |
| WESAD_S4 | 61.06 | 90.44 | 7 | True | 2017-06-13 09:04:00+00:00 | 2017-06-13 09:15:00+00:00 |
| WESAD_S5 | 61.50 | 46.44 | 6 | True | 2017-06-13 12:50:00+00:00 | 2017-06-13 13:00:00+00:00 |
| WESAD_S6 | 66.47 | 72.10 | 8 | True | 2017-06-14 11:52:00+00:00 | 2017-06-14 12:04:00+00:00 |
| WESAD_S7 | 68.66 | 54.61 | 8 | True | 2017-07-06 11:33:00+00:00 | 2017-07-06 11:45:00+00:00 |
| WESAD_S8 | 70.08 | 57.98 | 7 | True | 2017-07-10 11:33:00+00:00 | 2017-07-10 11:44:00+00:00 |
| WESAD_S9 | 75.58 | 38.84 | 5 | True | 2017-07-11 11:28:00+00:00 | 2017-07-11 11:38:00+00:00 |


## 6. Distribusi label WESAD per window

Label window = mayoritas label protokol per detik. Window dengan `gt_label_purity` < 1 berada di transisi antar kondisi.

| gt_label | windows | pure_windows | good_windows | hr_mean | hrv_rmssd_mean | accel_magnitude_std |
|---|---|---|---|---|---|---|
| amusement | 95 | 16 | 67 | 71.59 | 63.78 | 0.03 |
| baseline | 291 | 217 | 215 | 71.24 | 63.79 | 0.04 |
| meditation | 197 | 44 | 174 | 68.73 | 70.72 | 0.03 |
| stress | 169 | 90 | 11 | 78.08 | 61.32 | 0.08 |
| transient | 623 | 209 | 240 | 71.86 | 70.33 | 0.06 |


## 7. Akurasi HR/HRV wrist (PPG) vs ECG dada (referensi)

HR dan RMSSD dari BVP wrist dibandingkan dengan HR/RMSSD dari ECG dada pada detik yang sama.

| patient_id | timestamp_source | hr_ppg_coverage | hr_ppg_vs_ecg_mae_bpm | hr_ppg_within_5bpm | hrv_ppg_coverage | rmssd_ppg_median_ms | rmssd_ecg_median_ms | rmssd_ppg_vs_ecg_spearman |
|---|---|---|---|---|---|---|---|---|
| WESAD_S2 | e4_exact_match | 0.46 | 2.73 | 0.87 | 0.30 | 59.60 | 53.90 | 0.49 |
| WESAD_S3 | e4_exact_match | 0.39 | 3.74 | 0.77 | 0.05 | 113.10 | 95.40 | 0.63 |
| WESAD_S4 | e4_exact_match | 0.62 | 2.97 | 0.84 | 0.34 | 90.00 | 73.60 | 0.51 |
| WESAD_S5 | e4_exact_match | 0.35 | 2.40 | 0.89 | 0.15 | 53.50 | 27.40 | 0.04 |
| WESAD_S6 | e4_exact_match | 0.53 | 3.22 | 0.81 | 0.24 | 82.80 | 39.10 | 0.23 |
| WESAD_S7 | e4_exact_match | 0.51 | 3.59 | 0.78 | 0.30 | 62.60 | 63.20 | 0.75 |
| WESAD_S8 | e4_exact_match | 0.53 | 3.26 | 0.84 | 0.42 | 67.80 | 50.50 | 0.41 |
| WESAD_S9 | e4_exact_match | 0.51 | 2.89 | 0.83 | 0.40 | 40.70 | 34.80 | 0.45 |
| WESAD_S10 | e4_exact_match | 0.62 | 2.88 | 0.90 | 0.61 | 45.30 | 30.50 | 0.33 |
| WESAD_S11 | e4_exact_match | 0.47 | 4.61 | 0.83 | 0.35 | 59.10 | 36.30 | 0.32 |
| WESAD_S13 | e4_exact_match | 0.44 | 3.96 | 0.82 | 0.32 | 47.70 | 31.30 | 0.11 |
| WESAD_S14 | e4_exact_match | 0.68 | 4.91 | 0.87 | 0.65 | 50.10 | 28.20 | 0.16 |
| WESAD_S15 | e4_exact_match | 0.44 | 3.10 | 0.84 | 0.32 | 47.60 | 34.00 | 0.54 |
| WESAD_S16 | e4_exact_match | 0.52 | 7.79 | 0.78 | 0.34 | 63.00 | 39.70 | 0.65 |
| WESAD_S17 | e4_exact_match | 0.40 | 4.87 | 0.74 | 0.19 | 98.80 | 75.50 | 0.30 |


Rata-rata: MAE HR 3.8 bpm, 83% detik dalam ±5 bpm, Spearman RMSSD PPG vs ECG 0.39.


Error HR per kondisi (dihitung pada detik yang punya HR wrist dan ECG):

| gt_label | hr_coverage | hr_mae_bpm | within_5bpm |
|---|---|---|---|
| amusement | 0.71 | 2.45 | 0.88 |
| baseline | 0.65 | 2.42 | 0.89 |
| ignore | 0.52 | 5.45 | 0.77 |
| meditation | 0.79 | 2.74 | 0.85 |
| stress | 0.17 | 19.93 | 0.53 |
| transient | 0.40 | 3.92 | 0.81 |


## Keterbatasan

- Batas plausibilitas di `src/schema.py` adalah batas teknis sensor, bukan rentang klinis.

- Missing data tidak diisi (tidak ada `fillna(0)` atau interpolasi). Window tanpa sampel cukup menjadi `insufficient_data`.

- Window bertumpuk 4 menit. Split train/test harus per pasien, bukan per window acak.

- Kolom `gt_*` hanya kunci evaluasi. Jangan dipakai sebagai fitur model.

- HR wrist (PPG) tidak terbaca saat bicara/gerak tangan (kondisi `stress`/TSST): hanya 11 dari 169 window `stress` yang `good`. Gate SQI sengaja membuang HR yang tidak bisa dipercaya daripada mengeluarkan angka salah. Klasifikasi stres berbasis HR wrist pada WESAD akan sangat bias ke window yang tenang.

- RMSSD dari PPG wrist secara sistematis lebih tinggi dari ECG dan korelasinya lemah. Perlakukan fitur HRV WESAD sebagai tidak tervalidasi; `gt_hrv_rmssd_ecg` (dada) tersedia hanya untuk pembanding.

- Urutan protokol tetap (baseline selalu sebelum stress). Model bisa mempelajari waktu, bukan kondisi.

- 15 subjek sehat di lab. Hasil eksperimen bersifat eksploratif.
