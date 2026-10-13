# Kontrak Data — PTSD Guardian (Data Engineering)

Schema version: `1.0`. Sumber kebenaran di kode: `src/schema.py`. Kalau dokumen ini dan kode berbeda, kode yang berlaku.

> Semua output adalah estimasi pola fisiologis, bukan diagnosis PTSD.

## 1. Record sensor mentah (input pipeline)

Satu baris = satu pengukuran. Simulator dan adapter WESAD menghasilkan 1 Hz.

| Field | Tipe | Wajib | Satuan / format | Catatan |
|---|---|---|---|---|
| `patient_id` | string | ya | pseudonim | contoh `SIM01`, `WESAD_S2` |
| `timestamp` | string ISO-8601 | ya | UTC, `Z` atau offset `+07:00` | tanpa zona waktu: dianggap UTC dan ditandai `naive_timezone` |
| `heart_rate` | float/null | ya (boleh null) | bpm | null = tidak ada pengukuran, bukan 0 |
| `session_id` | string | tidak | | kalau kosong: sesi baru setelah jeda > 30 menit |
| `hrv_rmssd` | float/null | tidak | ms, RMSSD 60 s | null kalau perangkat tidak menyediakan atau detak tidak cukup |
| `accel_x`, `accel_y`, `accel_z` | float/null | tidak | g, termasuk gravitasi | sampel sesaat per detik |
| `eda` | float/null | tidak | µS | sinyal tambahan, nullable |
| `skin_temp` | float/null | tidak | °C | sinyal tambahan, nullable |
| `source` | string | tidak | | `simulator`, `empatica_e4`, ... |
| `received_at` | string ISO-8601 | tidak | UTC | waktu diterima backend; dipakai untuk cek data terlambat |

Kolom berawalan `gt_` adalah ground truth untuk evaluasi. Kolom ini **tidak boleh** dipakai sebagai fitur model dan dibuang oleh `replay.py`.

Batas plausibilitas teknis (`VALID_RANGES`): HR 25–240 bpm, RMSSD 1–300 ms, akselerasi ±16 g, EDA 0–100 µS, suhu 15–45 °C. Ini batas sanity sensor, bukan rentang normal klinis.

## 2. Validasi — `src/ingestion/validate.py`

`validate_sensor_data(df) -> DataFrame` tidak membuang dan tidak memperbaiki apa pun. Fungsi ini menambah kolom:

| Kolom | Arti |
|---|---|
| `ingest_seq` | urutan baris input (urutan kedatangan) |
| `timestamp_utc` | timestamp ter-parse (NaT kalau invalid) |
| `received_utc` | `received_at` ter-parse |
| `quality_issues` | semua issue, dipisah `;` |
| `quality_flag` | issue dengan prioritas tertinggi, atau `good` |
| `is_usable` | False untuk record yang tidak boleh masuk clean series |

Prioritas `quality_flag` (tertinggi dulu):

| Flag | Kondisi | Masuk clean series? |
|---|---|---|
| `invalid_timestamp` | tidak bisa di-parse, sebelum 2000, atau > 5 menit di masa depan | tidak |
| `missing_patient_id` | kosong/spasi | tidak |
| `duplicate` | `patient_id` + `timestamp` sama dengan record yang diterima lebih akhir | tidak (yang terakhir diterima menang) |
| `invalid_value` | nilai non-numerik atau di luar batas | ya, tapi nilai itu jadi NaN di clean |
| `missing_value` | `heart_rate` null | ya |
| `late` | diterima > 120 s setelah diukur | ya |
| `out_of_order` | datang setelah record bertimestamp lebih baru | ya |
| `good` | tidak ada masalah | ya |

Issue tambahan (tidak jadi flag utama): `naive_timezone`, `duplicate_conflict` (duplikat dengan nilai HR berbeda), `out_of_range:<kolom>`, `non_numeric:<kolom>`.

## 3. Preprocessing — `src/preprocessing/clean.py`

- `clean_sensor_data(validated)`: buang record `is_usable=False`, NaN-kan nilai invalid, urutkan, beri `session_id` dan `t_rel_s`.
- `detect_gaps(clean)`: periode tanpa record, yaitu jeda > 3× interval nominal sesi.
- `interval_profile(clean)`: per sesi berisi interval nominal, jumlah jeda terlalu rapat/panjang, dan rasio HR hilang.
- Tidak ada smoothing, `fillna(0)`, atau interpolasi. HR (window 5 s) dan RMSSD (60 s) sudah merupakan nilai berjendela dari perangkat.

Window: 5 menit, geser 1 menit, selaras jam (mulai di detik :00), half-open `[start, end)`, hanya window yang seluruhnya di dalam sesi.

`data_quality` window:
- `good` jika `valid_sample_ratio` ≥ 0,5 (minimal 150 s HR valid) **dan** HR valid terbaru tidak lebih tua dari 60 s sebelum `window_end`;
- selain itu `insufficient_data`.

Aturan kesegaran (`staleness_s`) memastikan sensor yang sedang terputus tidak tampil sebagai kondisi normal (spec Skenario C).

## 4. Tabel fitur — `src/features/engineer.py`

Satu baris per window. Urutan kolom tetap (`FEATURE_TABLE_COLUMNS`):

| Kolom | Arti |
|---|---|
| `patient_id`, `session_id`, `source` | identitas |
| `window_start`, `window_end` | UTC, `[start, end)` |
| `hr_mean`, `hr_std` | dari HR valid; None kalau < 2 sampel |
| `hr_delta_pct` | (hr_mean − hr_baseline) / hr_baseline × 100; None tanpa baseline valid |
| `hrv_rmssd_mean` | rata-rata RMSSD valid |
| `hrv_delta_pct` | perubahan RMSSD terhadap baseline |
| `accel_magnitude_std` | simpangan baku √(x²+y²+z²) |
| `valid_sample_ratio` | HR valid / sampel yang diharapkan |
| `hrv_valid_ratio`, `accel_valid_ratio` | sama, untuk RMSSD dan akselerasi |
| `eda_mean`, `skin_temp_mean` | sinyal tambahan |
| `eda_delta`, `skin_temp_delta` | selisih absolut (µS, °C) terhadap baseline EDA/suhu pasien; tidak butuh baseline HR yang valid |
| `n_samples` | record di dalam window |
| `staleness_s` | detik sejak HR valid terbaru sampai `window_end` |
| `data_quality` | `good` / `insufficient_data` |
| `is_baseline_period` | True kalau window dipakai menghitung baseline; keluarkan dari evaluasi |
| `gt_*` | label agregat untuk evaluasi saja (mayoritas + `_purity`) |

`MODEL_FEATURES` (urutan wajib sama saat training dan inference):
`hr_mean, hr_std, hr_delta_pct, hrv_rmssd_mean, hrv_delta_pct, accel_magnitude_std, valid_sample_ratio`.

### Baseline pribadi

Median `hr_mean`, `hrv_rmssd_mean`, `eda_mean`, dan `skin_temp_mean` dari window `good` dalam periode kalibrasi. Baseline valid jika ada minimal 5 window. Disimpan di `data/baselines/<source>_baselines.json`.

| Mode | Dipakai untuk | Periode kalibrasi |
|---|---|---|
| `calibration` | simulator, perangkat nyata | 30 menit pertama sesi pertama pasien |
| `label` | WESAD | paruh pertama window `good` yang seluruhnya di kondisi protokol `baseline` |

Ini baseline prototipe untuk demo dan eksperimen, bukan referensi klinis.

## 5. Fungsi untuk ML engineer dan backend

```python
from src.ingestion.validate import validate_sensor_data
from src.preprocessing.clean import clean_sensor_data
from src.features.engineer import build_feature_table, compute_window_features, load_baselines
from src.schema import MODEL_FEATURES

# Batch (training / eksperimen)
features = pd.read_csv("data/features/wesad_features.csv", parse_dates=["window_start", "window_end"])
train = features[(features.data_quality == "good") & ~features.is_baseline_period]
X, groups = train[MODEL_FEATURES], train["patient_id"]   # split per patient_id

# Online (API): record 5 menit terakhir satu pasien
clean = clean_sensor_data(validate_sensor_data(raw_last_5_min))
baseline = load_baselines("data/baselines/synthetic_baselines.json").get(patient_id)
feats = compute_window_features(clean, baseline, window_end=now)   # dict, kunci sama dengan tabel batch
if feats["data_quality"] != "good":
    ...  # tampilkan "data tidak cukup", jangan "normal"
```

Batch dan online memakai fungsi yang sama (`summarize_window` + `apply_baseline`). Kesamaan hasilnya diuji di `tests/test_features.py`.

## 6. Kualitas PPG wrist (WESAD)

HR dan RMSSD WESAD dihitung dari BVP wrist 64 Hz:

1. Bandpass 0,7–3 Hz.
2. Periode detak lokal dan **SQI** (signal quality index) dari autokorelasi window 8 s.
3. Puncak ditolak kalau terlalu dekat dengan puncak sebelumnya (puncak dikrotik) atau kalau SQI < 0,5.
4. IBI ditolak kalau di luar 333–1500 ms atau menyimpang > 25% dari median lokal. Untuk RMSSD, IBI juga ditolak kalau berubah > 20% dari IBI sebelumnya.

Ambang SQI 0,5 dikalibrasi terhadap HR dari ECG dada pada 15 subjek:

| SQI min | Coverage HR | MAE vs ECG | Error > 15 bpm |
|---|---|---|---|
| 0 | 95% | 10,0 bpm | 19% |
| 0,5 | 50% | 3,8 bpm | 3% |

HR yang tidak bisa dipercaya dijadikan null, bukan dikeluarkan sebagai angka salah.
