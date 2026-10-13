# Dataset WESAD untuk Target B (stres vs non-stres)

> Eksperimen eksploratif: stres akut di lab pada 15 subjek sehat. Bukan episode PTSD.

## Filter

1. Label window ∈ ['baseline', 'stress', 'amusement', 'meditation']; `transient` dan `ignore` dibuang.
2. `gt_label_purity` ≥ 0.9 (label lain maksimal 30 detik dalam window 5 menit).
3. Window kalibrasi baseline (`is_baseline_period=True`) dibuang, karena delta-nya ≈ 0 secara konstruksi.
4. Target: `y = 1` untuk `stress`, `0` untuk baseline/amusement/meditation.

## Keputusan varian

| varian | window | window stress | subjek dgn stress | keputusan |
|---|---|---|---|---|
| V1: hanya window `good` | 372 | 11 | 3 | **ditolak**: stress terlalu sedikit |
| V2: semua window | 657 | 169 | 15 | **dipakai**, dengan fitur bebas leakage missingness |


Kenapa V1 sangat kecil: HR wrist gagal terbaca saat subjek bicara dan bergerak di tugas stress (TSST), sehingga hampir semua window stress berstatus `insufficient_data`.

## Distribusi kelas per subjek (V2)

| patient_id | baseline | stress | amusement | meditation | stress_share |
|---|---|---|---|---|---|
| WESAD_S2 | 12 | 10 | 6 | 13 | 0.24 |
| WESAD_S3 | 13 | 11 | 7 | 12 | 0.26 |
| WESAD_S4 | 13 | 11 | 6 | 13 | 0.26 |
| WESAD_S5 | 14 | 11 | 6 | 13 | 0.25 |
| WESAD_S6 | 13 | 11 | 6 | 14 | 0.25 |
| WESAD_S7 | 12 | 11 | 6 | 13 | 0.26 |
| WESAD_S8 | 13 | 12 | 7 | 14 | 0.26 |
| WESAD_S9 | 14 | 11 | 7 | 13 | 0.24 |
| WESAD_S10 | 16 | 12 | 7 | 14 | 0.24 |
| WESAD_S11 | 11 | 11 | 7 | 13 | 0.26 |
| WESAD_S13 | 14 | 11 | 6 | 13 | 0.25 |
| WESAD_S14 | 11 | 12 | 6 | 12 | 0.29 |
| WESAD_S15 | 16 | 12 | 6 | 13 | 0.26 |
| WESAD_S16 | 12 | 11 | 6 | 14 | 0.26 |
| WESAD_S17 | 12 | 12 | 6 | 13 | 0.28 |


Total: 657 window, 169 stress (26%). Kelas positif tersedia di semua subjek, jadi leave-one-subject-out bisa dipakai.

## Missingness per kelas (alasan pemilihan fitur)

Proporsi window dengan nilai kosong. Perbedaan besar antar kelas berarti missingness ikut membawa label.

| feature | baseline | stress | amusement | meditation |
|---|---|---|---|---|
| accel_magnitude_std | 0.00 | 0.00 | 0.00 | 0.00 |
| eda_delta | 0.00 | 0.00 | 0.00 | 0.00 |
| eda_mean | 0.00 | 0.00 | 0.00 | 0.00 |
| hr_delta_pct | 0.16 | 0.16 | 0.14 | 0.14 |
| hr_mean | 0.00 | 0.02 | 0.00 | 0.00 |
| hr_std | 0.00 | 0.02 | 0.00 | 0.00 |
| hrv_rmssd_mean | 0.10 | 0.71 | 0.11 | 0.09 |
| skin_temp_delta | 0.00 | 0.00 | 0.00 | 0.00 |
| valid_sample_ratio | 0.00 | 0.00 | 0.00 | 0.00 |


Rata-rata `valid_sample_ratio`: baseline 0.59, stress 0.18, amusement 0.63, meditation 0.71.

## Feature set

| set | fitur |
|---|---|
| hr | hr_mean, hr_std, hr_delta_pct, accel_magnitude_std |
| hr_eda | hr_mean, hr_std, hr_delta_pct, accel_magnitude_std, eda_mean, eda_delta, skin_temp_delta |
| eda | eda_mean, eda_delta, skin_temp_delta, accel_magnitude_std |


Fitur yang dikeluarkan:

| fitur | alasan |
|---|---|
| valid_sample_ratio | PPG coverage collapses during stress (see mean valid_sample_ratio below): label leak |
| hrv_rmssd_mean | missing far more often in stress (see missingness table); wrist RMSSD not validated vs ECG |
| hrv_delta_pct | same as hrv_rmssd_mean |
| hrv_valid_ratio | missingness = label leak |
| accel_valid_ratio | not informative (always ~1) |
| skin_temp_mean | absolute level is person-specific; skin_temp_delta is used instead |
| staleness_s | PPG-quality indicator, same leak as valid_sample_ratio |
| gt_* | ground truth / ECG reference, never a feature |


## Checklist audit (spec Step 8)

| Cek | Status |
|---|---|
| Data dari sensor wearable | ya, wrist Empatica E4 saja |
| Label ground truth jelas | ya, kondisi protokol (stres akut TSST), bukan PTSD |
| Target sesuai klaim | ya, klaim = "klasifikasi stres akut di lab" |
| Timestamp dan label selaras | ya, label 700 Hz sinkron dengan wrist di `SX.pkl` |
| Jumlah subjek/kelas memadai | 15 subjek, 169 window stress — kecil, evaluasi eksploratif |
| Lisensi | sitasi wajib; anggap non-komersial |
| Tidak ada fitur yang membocorkan label | missingness-leak dibuang; `gt_*` tidak dipakai |

## Risiko yang tetap ada

- Urutan protokol tetap (baseline selalu pertama). `skin_temp_delta` bisa ikut menangkap waktu sejak mulai.
- HR saat stress dihitung hanya dari detik yang lolos gate SQI (momen paling tenang), sehingga cenderung under-estimate.
- Window bertumpuk 4 menit: split wajib per subjek (`groups = patient_id`).
- Baseline HR S10 dan S15 tidak valid, sehingga `hr_delta_pct` mereka kosong (kosong karena subjek, bukan karena label).
