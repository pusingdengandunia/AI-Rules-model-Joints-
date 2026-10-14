# Run It — PTSD Guardian MVP (Data Engineering + ML)

Panduan menjalankan dan menguji proyek di `D:\jointsHackathon\ptsd-wearable-mvp`.
Semua perintah dijalankan dari folder proyek, di PowerShell atau terminal VS Code.

> Hasil sistem adalah estimasi pola fisiologis, bukan diagnosis PTSD. Data sintetis membuktikan alur dan logika, bukan akurasi klinis.

---

## 0. Persiapan (sekali saja)

```powershell
cd D:\jointsHackathon\ptsd-wearable-mvp
python --version                      # butuh Python 3.10+ (sudah diuji di 3.13)
pip install -r requirements.txt       # numpy, pandas, scipy, matplotlib, scikit-learn, joblib, tzdata, pytest
```

Folder `data/` tidak disimpan di git. Setelah clone, jalankan minimal **Bagian 1 langkah 1** dulu. `models/model.joblib` sudah ikut di repo.

---

## 1. Bangun semua data dan model (urutan wajib)

| # | Perintah | Waktu | Hasil |
|---|---|---|---|
| 1 | `python -m src.pipeline --source synthetic --regenerate` | ~1 menit | data sintetis 24 sesi → `data/raw`, `data/validated`, `data/clean`, `data/features`, `data/baselines` |
| 2 | `python -m src.pipeline --source wesad --wesad-root D:/jointsHackathon/WESAD/WESAD` | ~15 menit, sekali | konversi WESAD (butuh ~2 GB RAM bebas) |
| 3 | `python -m src.ml.dataset` | detik | `data/ml/wesad_dataset.csv` (657 window, 169 stress) |
| 4 | `python -m src.ml.train` | ~1 menit | `models/model.joblib`, `models/model_card.json`, `reports/ml_eval_wesad.md` |

Catatan:
- Langkah 2–4 hanya diperlukan untuk mode `model` dan `mix`. Mode `rules` cukup dengan langkah 1.
- Kalau WESAD sudah pernah dikonversi (ada `data/raw/wesad/WESAD_S*.csv`), cukup jalankan `python -m src.pipeline --source all` untuk membangun ulang keduanya.

---

## 2. Smoke test cepat (± 2 menit)

Untuk memastikan semuanya hidup setelah ada perubahan:

```powershell
python -m pytest -q
python -m src.ml.simulate_stream --sessions SIM01_mixed_demo SIM01_sleep_disruption
```

Yang diharapkan:
- `110 passed`. Test WESAD/model otomatis `skipped` kalau Bagian 1 langkah 2–4 belum dijalankan.
- Gladi: `SIM01_mixed_demo 8/8` dan `SIM01_sleep_disruption 5/5 checks pass`.

---

## 3. Unit test (pytest)

```powershell
python -m pytest -q                         # semua test (~35 detik)
python -m pytest -q tests/test_rules.py     # satu file
python -m pytest -q -k sleep                # test yang namanya mengandung "sleep"
python -m pytest -q -k mix                  # test mode mix
python -m pytest -v tests/test_rules.py     # tampilkan nama setiap test
python -m pytest -x                         # berhenti di kegagalan pertama
```

| File | Yang diuji |
|---|---|
| `tests/test_validate.py` | validasi record: duplikat, timestamp rusak/tanpa zona, nilai di luar batas, data terlambat |
| `tests/test_clean.py` | data hilang tidak menjadi normal, gap, sesi, staleness (sensor baru putus) |
| `tests/test_features.py` | kolom fitur sesuai kontrak, batch = online, baseline dan delta |
| `tests/test_simulator.py` | setiap skenario sintetis berisi pola yang dijanjikan, termasuk skenario tidur |
| `tests/test_signals.py` | deteksi detak PPG, SQI, unpickler aman |
| `tests/test_rules.py` | rules siang (prioritas, olahraga, cooldown, persistensi) dan anomali waktu tidur |
| `tests/test_dataset.py` | filter dataset WESAD, pengaman kebocoran label |
| `tests/test_predict.py` | mode `rules` / `model` / `mix`, format output, `insufficient_data`, fallback |
| `tests/test_stream.py` | deduplikasi notifikasi caregiver |

Test anomali waktu tidur yang diminta:

```powershell
python -m pytest -v tests/test_rules.py -k "test_1_night or test_2_night or test_3_day or brief_wake"
```

| Test | Skenario | Harapan |
|---|---|---|
| `test_1_night_movement_is_sleep_disruption_not_activity` | gerak tinggi jam 02:00 | `monitor` → `elevated`, bukan `activity` |
| `test_2_night_hr_spike_monitor_then_elevated` | HR naik jam 03:00 | window 1 `monitor`, window 2 `elevated` |
| `test_3_day_movement_stays_activity` | gerak tinggi jam 10:00 | `normal` (reason `activity`) |
| `test_brief_wake_on_sliding_windows_stays_monitor` | bangun ~4 menit | maksimal `monitor` |

---

## 4. Gladi bersih offline (alur live tanpa API)

Memutar data sintetis seolah datang dari jam tangan. Setiap 60 detik dijalankan validate → clean → fitur → prediksi → notifikasi, lalu hasilnya dicek otomatis terhadap ground truth.

```powershell
python -m src.ml.simulate_stream --mode rules     # default demo
python -m src.ml.simulate_stream --mode mix       # rules + petunjuk model
python -m src.ml.simulate_stream --mode model     # model WESAD saja (eksperimen)

# satu atau beberapa sesi saja (lebih cepat)
python -m src.ml.simulate_stream --mode rules --sessions SIM01_sleep_disruption
python -m src.ml.simulate_stream --mode mix --sessions SIM01_normal SIM01_high_activity
```

Sesi yang tersedia: `SIM01_normal`, `SIM01_hr_elevation`, `SIM01_hrv_drop`, `SIM01_high_activity`, `SIM01_missing_data`, `SIM01_bad_quality`, `SIM01_mixed_demo`, `SIM01_sleep_disruption` (juga `SIM02_*`, `SIM03_*`).

Waktu: semua sesi ~3–4 menit per mode (sesi tidur berisi 5 jam data).

Hasil yang diharapkan:

| Mode | Cek lulus | Keterangan |
|---|---|---|
| `rules` | **32/32** | default demo |
| `mix` | **32/32** | model hanya menambah `monitor` saat diam di siang hari |
| `model` | 18/32 | wajar: model WESAD tidak mengenal tidur dan membaca gerakan sebagai stres |

Output:
- `reports/stream_rehearsal_<mode>.md` — tabel cek lulus/gagal dan daftar notifikasi.
- `reports/stream/<sesi>_<mode>.png` — plot HR + pita status per menit + arsiran kejadian sebenarnya.
- `reports/stream/<sesi>_<mode>_timeline.csv` — status per menit beserta fitur.

Plot yang paling enak dilihat: `reports/stream/SIM01_mixed_demo_rules.png` dan `reports/stream/SIM01_sleep_disruption_rules.png`.

Cara membaca pita status: hijau = `normal`, kuning = `monitor`, merah = `elevated_physiological_anomaly`, abu-abu = `insufficient_data`. Garis merah putus-putus = notifikasi `alert`, hijau = `resolved`, hitam = `device_check`.

---

## 5. Evaluasi rules dan model

```powershell
python -m src.ml.evaluate_rules           # rules pada data sintetis -> reports/rules_eval_synthetic.md
python -m src.ml.evaluate_rules --tune    # grid search ambang ulang (lama, ~10 menit)
python -m src.ml.train                    # model WESAD leave-one-subject-out -> reports/ml_eval_wesad.md
```

Hasil yang diharapkan:
- Rules: recall arousal, hrv_drop, dan gangguan tidur 100%; bangun singkat tidak pernah `elevated`; 0 false alarm di split tune, test, dan holdout.
- Model terbaik `logreg__eda`: F1 ± 0,78, AUC ± 0,96 (tebakan naif 0,41).

---

## 6. Mencoba prediksi langsung dari Python

```powershell
python
```

```python
import pandas as pd
from src.ml.predict import Predictor

f = pd.read_csv("data/features/synthetic_features.csv", parse_dates=["window_start", "window_end"])
windows = f[f.session_id == "SIM01_mixed_demo"].sort_values("window_end").to_dict("records")

p = Predictor("rules")            # ganti dengan "mix" atau "model"
for w in windows:
    r = p.predict(w)              # satu Predictor dipakai untuk semua window (rules menyimpan riwayat)
    if r["risk_level"] != "normal":
        print(r["window_end"], r["risk_level"], r["reasons"])

p.mode, p.model_version, p.fallback_reason   # mode aktif; fallback_reason terisi kalau model gagal dimuat
```

Format output selalu sama di ketiga mode:

```json
{"patient_id": "SIM01", "window_end": "...", "risk_level": "monitor",
 "reasons": ["hr_above_baseline", "at_rest"], "model_version": "rules_v1", "data_quality": "good"}
```

---

## 7. Stream data ke backend (untuk integrasi nanti)

```powershell
python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30                                      # cetak JSON ke layar
python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --post http://localhost:8000/ingest   # kirim ke endpoint
python -m src.simulator.replay data/raw/synthetic/SIM01_bad_quality.csv --speed 0 > stream.jsonl                       # simpan ke file
```

Kolom `gt_*` (kunci jawaban) otomatis dibuang. `--speed 30` = 30× lebih cepat dari waktu nyata; `--speed 0` = secepat mungkin.

---

## 8. Laporan yang dihasilkan

| File | Isi |
|---|---|
| `reports/data_quality_synthetic.md` | kualitas data sintetis, deteksi gangguan yang disuntikkan |
| `reports/data_quality_wesad.md` | kualitas WESAD, akurasi HR wrist vs ECG |
| `reports/rules_eval_synthetic.md` | evaluasi rules per event dan hasil tuning |
| `reports/wesad_dataset_summary.md` | dataset WESAD untuk training |
| `reports/ml_eval_wesad.md` | perbandingan model vs rules vs naif, probe confound |
| `reports/stream_rehearsal_<mode>.md` | gladi bersih per mode |
| `reports/plots/synthetic/*.png` | sinyal mentah per skenario |
| `reports/stream/*.png` | status per menit per sesi |

---

## 9. Kalau ada masalah

| Gejala | Penyebab / solusi |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | jalankan dari folder `D:\jointsHackathon\ptsd-wearable-mvp` dengan `python -m ...`, bukan `python src\...py` |
| `FileNotFoundError: ... data/raw/synthetic` atau `synthetic_baselines.json` | jalankan Bagian 1 langkah 1 |
| Mode `model`/`mix` keluar `model_version = rules_v1` dan reason `fallback_rules` | model belum dilatih: jalankan Bagian 1 langkah 2–4 |
| Test WESAD/model `skipped` | normal kalau WESAD/model belum dibangun |
| `ZoneInfoNotFoundError` | `pip install tzdata` |
| `MemoryError` saat konversi WESAD | tutup aplikasi lain; konversi per subjek: `python -m src.sources.wesad --root D:/jointsHackathon/WESAD/WESAD --subjects S2 S3` |
| Hasil gladi berbeda dari tabel di atas | data sintetis lama: jalankan ulang `python -m src.pipeline --source synthetic --regenerate` |

---

## 10. Dokumentasi terkait

- `README.md` — ringkasan proyek dan struktur folder.
- `docs/data_contract.md` — kolom data, flag kualitas, fitur, cara pakai fungsi.
- `docs/ml_target.md` — target prediksi, kategori output, anomali waktu tidur (Bagian 8), mode prediksi (Bagian 9).
- `docs/jalur_b_datasets.md` — audit dataset WESAD.

