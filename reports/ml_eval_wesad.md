# Evaluasi ML — Target B (stres vs non-stres, WESAD)

> Eksperimen eksploratif: stres akut di lab, 15 subjek sehat. Bukan deteksi PTSD.

## Metode

- Dataset: `data/ml/wesad_dataset.csv` (657 window, 169 stress, 15 subjek). Lihat `reports/wesad_dataset_summary.md`.
- Evaluasi: leave-one-subject-out. Subjek tes tidak pernah dilihat saat training. Imputasi/scaling di-fit di dalam fold.
- Ambang probabilitas 0.5; `class_weight=balanced`. Metrik = rata-rata ± std dari 15 fold (per subjek).
- Pembanding: `always_stress` (selalu menebak stress) dan `rules_v1` (Target A; monitor/elevated dihitung sebagai stress).
- Probe confound: `probe_time_only__*` hanya memakai menit sejak awal sesi. Bukan kandidat model.

## Hasil semua kandidat

| candidate | f1_mean | f1_std | precision_mean | recall_mean | false_alarm_rate_mean | auc_mean |
|---|---|---|---|---|---|---|
| logreg__eda | 0.78 | 0.14 | 0.76 | 0.86 | 0.13 | 0.96 |
| logreg__hr_eda | 0.77 | 0.22 | 0.85 | 0.80 | 0.08 | 0.97 |
| rf__hr_eda | 0.75 | 0.20 | 0.79 | 0.82 | 0.13 | 0.95 |
| rf__eda | 0.74 | 0.25 | 0.74 | 0.80 | 0.12 | 0.92 |
| hgb__hr_eda | 0.74 | 0.20 | 0.80 | 0.80 | 0.13 | 0.95 |
| probe_time_only__hgb | 0.73 | 0.10 | 0.62 | 0.94 | 0.23 | 0.90 |
| hgb__eda | 0.71 | 0.27 | 0.73 | 0.75 | 0.11 | 0.91 |
| logreg__hr | 0.64 | 0.23 | 0.65 | 0.70 | 0.14 | 0.87 |
| rf__hr | 0.61 | 0.24 | 0.69 | 0.67 | 0.15 | 0.91 |
| hgb__hr | 0.61 | 0.26 | 0.66 | 0.67 | 0.13 | 0.87 |
| probe_time_only__logreg | 0.42 | 0.38 | 0.34 | 0.55 | 0.40 | 0.61 |
| always_stress | 0.41 | 0.02 | 0.26 | 1.00 | 1.00 |  |
| rules_v1 | 0.04 | 0.11 | 0.08 | 0.03 | 0.04 |  |


`rules_v1` hanya menilai 50% window (sisanya `insufficient_data`), dan hanya 7% window stress. Window yang tidak dinilai dihitung sebagai prediksi non-stress.

## Model terpilih: `logreg__eda`

Fitur (urutan tetap): `eda_mean, eda_delta, accel_magnitude_std`

Confusion matrix (gabungan out-of-fold):

|  | prediksi non-stress | prediksi stress |
|---|---|---|
| aktual non-stress | 423 | 65 |
| aktual stress | 24 | 145 |


Per subjek (fold):

| subject | f1 | precision | recall | false_alarm_rate | auc |
|---|---|---|---|---|---|
| WESAD_S10 | 0.89 | 0.80 | 1.00 | 0.08 | 0.99 |
| WESAD_S11 | 0.91 | 0.91 | 0.91 | 0.03 | 0.99 |
| WESAD_S13 | 0.56 | 0.39 | 1.00 | 0.52 | 1.00 |
| WESAD_S14 | 0.63 | 0.86 | 0.50 | 0.03 | 0.93 |
| WESAD_S15 | 0.83 | 0.83 | 0.83 | 0.06 | 0.97 |
| WESAD_S16 | 0.95 | 1.00 | 0.91 | 0.00 | 1.00 |
| WESAD_S17 | 0.86 | 1.00 | 0.75 | 0.00 | 0.92 |
| WESAD_S2 | 0.62 | 0.50 | 0.80 | 0.26 | 0.77 |
| WESAD_S3 | 0.63 | 0.75 | 0.55 | 0.06 | 0.95 |
| WESAD_S4 | 0.96 | 0.92 | 1.00 | 0.03 | 1.00 |
| WESAD_S5 | 0.85 | 0.73 | 1.00 | 0.12 | 0.99 |
| WESAD_S6 | 0.58 | 0.41 | 1.00 | 0.48 | 1.00 |
| WESAD_S7 | 0.95 | 1.00 | 0.91 | 0.00 | 1.00 |
| WESAD_S8 | 0.73 | 0.61 | 0.92 | 0.21 | 0.96 |
| WESAD_S9 | 0.78 | 0.75 | 0.82 | 0.09 | 0.90 |


## Interpretasi

- ML terbaik F1 0.78 vs `rules_v1` 0.04 vs tebakan naif 0.41.
- Rules lemah di sini bukan karena logikanya salah, tetapi karena HR wrist hilang saat stress sehingga rules dengan benar menjawab `insufficient_data`. Rules tidak dirancang untuk stres yang disertai bicara dan gerak.
- Sinyal yang tidak bergantung pada PPG (EDA, suhu, gerak) lebih andal di WESAD. Ini sejalan dengan temuan kualitas data.
- Variasi antar subjek besar (lihat std dan tabel per subjek). Dengan 15 subjek, angka ini eksploratif.
- **Confound urutan protokol:** model non-linear yang hanya diberi "menit sejak mulai" mencapai F1 0.73 (baris `probe_time_only__*`). Di WESAD stress selalu terjadi di waktu yang mirip. Regresi logistik dengan fitur waktu saja gagal, dan koefisien model terpilih didominasi `eda_delta`, sehingga model terpilih kemungkinan besar memakai sinyal fisiologis. Tetapi dataset ini tidak bisa membuktikannya sepenuhnya.
- `accel_magnitude_std` lebih tinggi saat stress sebagian karena subjek bicara dan bergestur di TSST (perilaku protokol, bukan fisiologi).

## Keputusan

- Demo tetap memakai `rules_v1` (Target A), sesuai `docs/ml_target.md`.
- Model `wesad_stress_logreg_eda_v1` disimpan untuk eksperimen dan tersedia di `src/ml/predict.py` dengan `mode="model"`, dengan klaim "pola mirip stres akut (eksperimen)", bukan PTSD.
