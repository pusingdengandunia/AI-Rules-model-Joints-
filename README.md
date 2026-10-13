# PTSD Guardian MVP — Data Engineering & ML

Implementasi spec `../PTSD_Guardian_MVP_ML_Data_Engineering.md` Step 1–10 (Data Engineering, rules, model, inference) plus gladi bersih offline. Integrasi API (Step 11+) belum dibuat.

> Hasil sistem adalah estimasi pola fisiologis, bukan diagnosis PTSD. Data simulasi membuktikan alur dan logika, bukan akurasi klinis. WESAD berisi stres akut di lab, bukan episode PTSD.

## Menjalankan dari nol

```bash
pip install -r requirements.txt

# 1. Data Engineering
python -m src.pipeline --source synthetic --regenerate                              # Jalur A (~30 s)
python -m src.pipeline --source wesad --wesad-root D:/jointsHackathon/WESAD/WESAD   # Jalur B, sekali (~15 menit, ~2 GB RAM)
python -m src.pipeline --source all                                                 # jalankan ulang keduanya kapan saja

# 2. Rules (Target A) — evaluasi per event di data sintetis
python -m src.ml.evaluate_rules           # pakai RULES_CONFIG
python -m src.ml.evaluate_rules --tune    # grid search ulang (tune SIM01-02, tes SIM03 + holdout seed 7)

# 3. Model (Target B) — eksperimen WESAD
python -m src.ml.dataset                  # data/ml/wesad_dataset.csv
python -m src.ml.train                    # LOSO, models/model.joblib + model_card.json

# 4. Gladi bersih offline (gerbang sebelum integrasi)
python -m src.ml.simulate_stream          # rules (default demo)
python -m src.ml.simulate_stream --mode model
python -m src.ml.simulate_stream --mode mix

# Tests (88)
python -m pytest -q
```

Stream data simulasi ke backend nanti (kolom `gt_*` dibuang):

```bash
python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --post http://localhost:8000/ingest
```

## Struktur

```
src/
  schema.py                 kontrak data: kolom, batas, flag, window, urutan fitur
  simulator/                Jalur A: generate.py (8 skenario, termasuk sleep_disruption), replay.py, plot.py
  sources/wesad.py          Jalur B: SX.pkl -> skema proyek (unpickler aman, timestamp absolut)
  signals/beats.py          deteksi detak PPG/ECG, SQI, HR dan RMSSD dari detak
  ingestion/validate.py     Step 3: quality_flag per record
  preprocessing/clean.py    Step 4: clean series, sesi, gap, window
  features/engineer.py      Step 5: fitur per window + baseline pribadi (batch = online)
  pipeline.py               runner Data Engineering + laporan kualitas
  ml/rules.py               Step 7: rules_v1 (RuleEngine) + anomali waktu tidur
  ml/evaluate_rules.py      Step 7: evaluasi per event + tuning
  ml/dataset.py             Step 8: dataset WESAD (filter, feature set bebas leakage)
  ml/train.py               Step 9: LOSO, perbandingan, model card
  ml/predict.py             Step 10: Predictor / predict_risk (mode rules, model, mix; kontrak sama)
  ml/simulate_stream.py     gladi bersih: stream -> validate -> fitur -> prediksi -> notifikasi
tests/                      88 test
docs/
  data_contract.md          kontrak data dan cara pakai fungsi
  ml_target.md              target prediksi, kategori output, kontrak output, batas klaim
  jalur_b_datasets.md       survei dataset + audit WESAD
models/model_card.json      model WESAD terpilih (model.joblib dibuat ulang oleh train.py)
reports/                    semua laporan (lihat tabel di bawah)
data/                       output pipeline (tidak di-commit; dibuat ulang)
```

## Laporan

| File | Isi |
|---|---|
| `reports/data_quality_synthetic.md`, `reports/data_quality_wesad.md` | kualitas data, missing data, gap, baseline, akurasi HR wrist vs ECG |
| `reports/rules_eval_synthetic.md` | evaluasi rules per event, hasil tuning |
| `reports/wesad_dataset_summary.md` | filter, distribusi kelas, alasan pemilihan fitur |
| `reports/ml_eval_wesad.md` | perbandingan model vs rules vs baseline naif, probe confound |
| `reports/stream_rehearsal_rules.md` | gladi bersih: cek skenario A/B/C, notifikasi |
| `reports/stream/*.png` | timeline status per sesi |

## Hasil utama

| Komponen | Hasil |
|---|---|
| Validasi | 330/330 gangguan sintetis tertangkap |
| Rules `rules_v1` (sintetis, pasien tes + holdout) | recall arousal 100%, hrv_drop 100%, 0 false alarm, 100% gap = `insufficient_data` |
| Model WESAD terbaik (`logreg__eda`, LOSO) | F1 0,78 ± 0,14, AUC 0,96 (naif 0,41; rules 0,04 karena HR wrist hilang saat stress) |
| Gladi bersih rules | 27/27 cek lulus; 1 alert per episode, `device_check` 1 per gangguan |
| Gladi bersih model | 15/27: model WESAD salah membaca olahraga sebagai stress → demo memakai rules |

## Mode prediksi

```python
from src.ml.predict import Predictor
p = Predictor("rules")   # default demo
p = Predictor("model")   # model WESAD saja (eksperimen)
p = Predictor("mix")     # rules memutuskan; model boleh menambah "monitor" saat rules normal & diam
result = p.predict(features)   # buat satu Predictor, pakai ulang untuk semua window
```

Detail: `docs/ml_target.md` Bagian 8 (anomali waktu tidur) dan Bagian 9 (mode).

## Keputusan desain penting

- **Tidak ada yang dibuang diam-diam.** Validasi hanya memberi flag; jejak lengkap di `data/validated`.
- **Missing ≠ normal.** Window `insufficient_data` jika HR valid < 50% atau HR terbaru > 60 s. App tidak pernah menampilkan `normal` tanpa data `good`.
- **Satu kode untuk batch dan online** (`compute_window_features`, `apply_baseline`), diuji sama persis.
- **HR wrist yang tidak bisa dipercaya dijadikan null** (gate SQI dikalibrasi terhadap ECG).
- **Rules adalah default demo.** Model WESAD tersedia lewat `Predictor("model")` untuk eksperimen; kalau model gagal dimuat, sistem otomatis kembali ke rules.
- **Pickle WESAD dibuka dengan unpickler allow-list** sehingga file yang dimanipulasi tidak bisa menjalankan kode.

## Batasan

- Semua ambang (rules, kualitas data, baseline) adalah aturan prototipe, bukan kriteria medis.
- Rules dituning di data sintetis yang skenarionya dirancang sendiri; perlu dikalibrasi ulang dengan data perangkat nyata.
- WESAD: 15 subjek sehat, urutan protokol tetap (waktu saja sudah memprediksi stress dengan F1 0,73 pada model non-linear); HR wrist hilang saat stress; HRV wrist belum tervalidasi.
