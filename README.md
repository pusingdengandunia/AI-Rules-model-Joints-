# PTSD Guardian MVP — Data Engineering

Pipeline data untuk spec `../PTSD_Guardian_MVP_ML_Data_Engineering.md` Step 1–5: sumber data, skema, validasi, preprocessing, dan feature engineering. Pipeline ini menerima data sensor dari simulator (Jalur A) atau WESAD (Jalur B), lalu menghasilkan tabel fitur per window untuk ML engineer dan backend.

> Hasil sistem adalah estimasi pola fisiologis, bukan diagnosis PTSD. Data simulasi menguji pipeline dan demo; data simulasi tidak membuktikan akurasi klinis. WESAD berisi stres akut di lab, bukan episode PTSD.

## Quick start

```bash
pip install -r requirements.txt

# Jalur A: data simulasi -> pipeline
python -m src.pipeline --source synthetic --regenerate

# Jalur B: WESAD (konversi pkl satu kali, ~1 menit per subjek, butuh ~2 GB RAM bebas)
python -m src.pipeline --source wesad --wesad-root D:/jointsHackathon/WESAD/WESAD

# Setelah konversi: jalankan ulang keduanya
python -m src.pipeline --source all

# Tests
python -m pytest -q
```

Stream data simulasi ke backend (kolom `gt_*` dibuang):

```bash
python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --post http://localhost:8000/ingest
```

## Struktur

```
src/
  schema.py                 kontrak data: kolom, batas, flag, konfigurasi window, urutan fitur
  simulator/                Jalur A: generate.py, replay.py, plot.py
  sources/wesad.py          Jalur B: SX.pkl -> skema proyek (unpickler aman, timestamp absolut)
  signals/beats.py          deteksi detak PPG/ECG, SQI, HR dan RMSSD dari detak
  ingestion/validate.py     Step 3: quality_flag per record
  preprocessing/clean.py    Step 4: clean series, sesi, gap, window
  features/engineer.py      Step 5: fitur per window + baseline pribadi (batch dan online)
  pipeline.py               runner + laporan kualitas data
tests/                      52 test (validasi, clean, fitur, simulator, sinyal)
docs/data_contract.md       kontrak data lengkap + cara pakai untuk ML/backend
docs/jalur_b_datasets.md    survei dataset publik + audit WESAD (lisensi, alignment, leakage)
data/                       output (lihat di bawah)
reports/                    data_quality_<source>.md, plots/synthetic/*.png
```

## Output pipeline (spec "Output Data Engineering")

| Deliverable spec | File |
|---|---|
| Dataset mentah tervalidasi | `data/validated/<source>_validated.csv` |
| Time-series yang dibersihkan | `data/clean/<source>_clean.csv`, `data/clean/<source>_gaps.csv` |
| Tabel fitur per window | `data/features/<source>_features.csv` |
| Baseline pribadi | `data/baselines/<source>_baselines.json` |
| Dokumentasi kualitas dan missing data | `reports/data_quality_<source>.md` |
| Fungsi untuk ML engineer | `validate_sensor_data`, `clean_sensor_data`, `build_feature_table`, `compute_window_features` (lihat `docs/data_contract.md` §5) |

Input mentah: `data/raw/synthetic/*.csv` (+ `events.csv`, `manifest.json`) dan `data/raw/wesad/WESAD_S*.csv` (+ `conversion_report.json`).

## Skenario simulasi (Jalur A)

| Skenario | Isi | Ekspektasi |
|---|---|---|
| `normal` | Istirahat + 2 aktivitas ringan | tidak ada alert |
| `hr_elevation` | HR +25–35 bpm saat diam, RMSSD turun, EDA naik | `alert` |
| `hrv_drop` | RMSSD turun ke ~45%, HR hanya +5–10 bpm | `warning` |
| `high_activity` | Olahraga 15 menit, akselerasi tinggi | tidak ada alert (confounder) |
| `missing_data` | Disconnect 2 menit (tanpa record), off-wrist 5 menit (HR null), dropout, artefak gerak | tidak ada alert; `insufficient_data` |
| `bad_quality` | Batch terlambat, duplikat kirim-ulang, timestamp rusak/tanpa zona, patient_id kosong, nilai di luar batas/non-numerik | tidak ada alert; semua gangguan terdeteksi validator |
| `mixed_demo` | 2 jam gabungan semua kasus | demo end-to-end |

Pemetaan ke tiga skenario demo di spec §7: A = `normal`, B = `hr_elevation` / `hrv_drop` (pembanding: `high_activity`), C = `missing_data`.

## Keputusan desain penting

- **Tidak ada yang dibuang diam-diam.** Validasi hanya memberi flag. Clean series mengeluarkan duplikat, timestamp invalid, dan record tanpa pasien; nilai di luar batas menjadi NaN. Semua jejak tetap ada di file `validated`.
- **Missing ≠ normal.** Tidak ada `fillna(0)` atau interpolasi. Window menjadi `insufficient_data` kalau HR valid < 50% atau HR valid terbaru lebih tua dari 60 s.
- **Satu kode untuk batch dan online.** `compute_window_features` memanggil fungsi yang sama dengan tabel fitur batch, dan kesamaannya diuji.
- **HR wrist yang tidak bisa dipercaya dijadikan null.** Gate SQI PPG dikalibrasi terhadap ECG WESAD (MAE 10,0 → 3,8 bpm).
- **Pickle WESAD dibuka dengan unpickler allow-list** (hanya array numpy), sehingga file yang dimanipulasi tidak bisa menjalankan kode.

## Batasan yang perlu diketahui tim

- WESAD: HR wrist hampir selalu hilang saat kondisi stress (11/169 window `good`). RMSSD wrist lemah korelasinya dengan ECG. Lihat `reports/data_quality_wesad.md` §7 dan `docs/jalur_b_datasets.md`.
- Ambang (batas plausibilitas, 50% valid, 60 s staleness, baseline 30 menit) adalah aturan prototipe, bukan kriteria medis.
- Step 6 ke atas (target, baseline rule `normal/monitor/elevated`, model, API) adalah bagian ML/integrasi dan belum dibuat di sini.
