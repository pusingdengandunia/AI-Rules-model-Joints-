# Serah Terima ke Fullstack Engineer — PTSD Guardian

Dokumen ini menjelaskan apa yang sudah dibuat di sisi Data Engineering + ML, apa yang perlu dibangun di backend/app, dan kontrak data di antaranya. Spec acuan: `PTSD_Guardian_MVP_ML_Data_Engineering.md` Step 11–12 dan Bagian 7–8.

> Sistem ini mengestimasi **pola fisiologis yang menyimpang dari baseline pribadi**. Bukan diagnosis, bukan deteksi episode PTSD. Kata "PTSD", "serangan", "diagnosis", atau "risiko x%" tidak boleh muncul di UI.

---

## 1. Ringkasan 1 menit

- Modul Python di repo ini mengubah **record sensor mentah** (1 per detik per pasien) menjadi **status per pasien setiap 1 menit**.
- Statusnya satu dari empat: `normal`, `monitor`, `elevated_physiological_anomaly`, `insufficient_data`.
- **Belum ada API.** Backend perlu memanggil fungsi Python yang sudah ada, atau membungkusnya jadi service (misalnya FastAPI, spec Step 11).
- Default demo: mode `rules`. Mode `mix` dan `model` tersedia untuk eksperimen.
- Beban: ±26 ms per pasien per menit di laptop (1 core ≈ 2.000 pasien per menit).

---

## 2. Alur yang perlu dibangun backend

```
jam tangan / simulator
   │  POST record (1 per detik, JSON)                     ← Bagian 3
   ▼
backend: simpan + buffer 15 menit terakhir per pasien
   │  setiap 60 detik, per pasien aktif:
   ▼
validate_sensor_data → clean_sensor_data → compute_window_features(window_end=now) → Predictor.predict
   │                                                                                   ← Bagian 4
   ▼
status per pasien (simpan riwayat)  →  AlertDeduplicator → notifikasi caregiver      ← Bagian 6
   ▼
aplikasi caregiver: tampilkan status + teks + reasons                                 ← Bagian 5
```

Kode referensi alur lengkapnya sudah ada dan teruji: fungsi `simulate()` di `src/ml/simulate_stream.py`. Backend tinggal meniru loop itu dengan data live.

---

## 3. Input: record sensor (dari jam tangan / simulator)

Satu JSON per detik per pasien. Contoh dari `python -m src.simulator.replay`:

```json
{"patient_id": "SIM01", "timestamp": "2026-01-11T02:45:00Z", "heart_rate": 92.0,
 "session_id": "SIM01_mixed_demo", "hrv_rmssd": 24.1, "accel_x": 0.61, "accel_y": 0.77,
 "accel_z": 0.13, "eda": 5.2, "skin_temp": 33.4, "source": "simulator",
 "received_at": "2026-01-11T02:45:02Z"}
```

| Field | Wajib | Catatan |
|---|---|---|
| `patient_id` | ya | pseudonim, bukan nama/NIK |
| `timestamp` | ya | ISO-8601 dengan zona waktu (`Z` atau `+07:00`) |
| `heart_rate` | ya (boleh `null`) | `null` = sensor tidak membaca; **jangan diisi 0** |
| `hrv_rmssd`, `accel_x/y/z`, `eda`, `skin_temp` | tidak | `null` kalau perangkat tidak punya |
| `session_id`, `source` | tidak | |
| `received_at` | disarankan | diisi backend saat record diterima (untuk deteksi data terlambat) |

Aturan backend:
- **Simpan semua record apa adanya**, termasuk yang aneh. Validasi tidak membuang data, hanya memberi flag (`quality_flag`).
- Duplikat (patient_id + timestamp sama) boleh datang. Validator memilih yang terakhir diterima.
- Kolom `gt_*` hanya ada di file simulator (kunci jawaban). `replay.py` sudah membuangnya.

Detail lengkap: `docs/data_contract.md` Bagian 1–2.

---

## 4. Memanggil modul ML (inti integrasi)

```python
import pandas as pd
from src.ingestion.validate import validate_sensor_data
from src.preprocessing.clean import clean_sensor_data
from src.features.engineer import compute_window_features, load_baselines
from src.ml.predict import Predictor

PREDICTOR = Predictor("rules")       # SATU instance per proses, dibuat saat service start
BASELINES = load_baselines("data/baselines/synthetic_baselines.json")   # patient_id -> dict

def status_for(patient_id: str, records: pd.DataFrame, now: pd.Timestamp) -> dict:
    """records = record mentah pasien ini yang DITERIMA dalam 15 menit terakhir."""
    clean = clean_sensor_data(validate_sensor_data(records, now=now))
    window = clean[(clean.timestamp_utc >= now - pd.Timedelta(minutes=5)) & (clean.timestamp_utc < now)]
    feats = compute_window_features(window, BASELINES.get(patient_id), window_end=now)
    feats.update(patient_id=patient_id, window_end=now)   # opsional: feats["timezone"] = "Asia/Makassar"
    return PREDICTOR.predict(feats)
```

Wajib diperhatikan:
1. **Satu `Predictor` untuk semua pasien dan semua menit.** Rules menyimpan riwayat per pasien (perlu 3 menit berturut-turut untuk `elevated`, dan ~10 menit untuk gangguan tidur). Membuat `Predictor` baru tiap menit = status tidak pernah naik ke `elevated`.
2. **Panggil tiap 60 detik untuk setiap pasien aktif, termasuk yang tidak mengirim data.** Kalau pasien diam (sensor putus), justru dari panggilan inilah status `insufficient_data` muncul.
3. `now` = waktu server saat itu, timezone-aware UTC.
4. Kalau pasien memulai sesi rekaman baru: `PREDICTOR.reset(patient_id)`.
5. Kalau pakai beberapa worker/proses, pastikan satu pasien selalu diproses oleh worker yang sama (state rules ada di memori).

Pilihan mode (disarankan lewat environment variable, misalnya `PREDICTOR_MODE`):

| Mode | Isi | Status |
|---|---|---|
| `rules` | aturan transparan, termasuk anomali waktu tidur | **default demo** |
| `mix` | rules memutuskan; model hanya boleh menambah `monitor` | eksperimen |
| `model` | model WESAD saja | eksperimen, jangan dipakai untuk notifikasi |

Kalau file model tidak ada, `mix`/`model` otomatis kembali ke `rules` (reason `fallback_rules`), tidak crash.

---

## 5. Output: status per pasien → aplikasi caregiver

```json
{"patient_id": "SIM01", "window_end": "2026-01-11 02:54:00+00:00",
 "risk_level": "elevated_physiological_anomaly", "risk_score": null,
 "reasons": ["hr_above_baseline", "at_rest"], "model_version": "rules_v1", "data_quality": "good"}
```

| Field | Arti untuk UI |
|---|---|
| `risk_level` | status utama (tabel di bawah) |
| `risk_score` | **selalu `null`**. Jangan tampilkan angka/persen risiko |
| `reasons` | kode penjelasan; terjemahkan dengan tabel di bawah |
| `model_version` | tampilkan di layar detail/debug saja |
| `data_quality` | `good` atau `insufficient_data` |
| `window_end` | waktu status ini (UTC) |

### Status → tampilan

| `risk_level` | Teks | Warna saran |
|---|---|---|
| `normal` | Pola fisiologis stabil | hijau |
| `monitor` | Pola fisiologis berubah — pantau | kuning |
| `elevated_physiological_anomaly` | Pola fisiologis abnormal — perlu pemeriksaan caregiver | merah |
| `insufficient_data` | Data sensor tidak cukup — periksa perangkat | abu-abu |

**Aturan keras:** `insufficient_data` tidak boleh ditampilkan sebagai `normal`, dan status lama tidak boleh ditampilkan seolah masih berlaku.

### Reasons → teks

| Reason | Teks untuk caregiver |
|---|---|
| `hr_above_baseline` | Detak jantung di atas pola biasanya |
| `hrv_below_baseline` | Variabilitas detak jantung di bawah pola biasanya |
| `at_rest` | Terjadi saat sedang diam |
| `activity` | Sedang beraktivitas fisik |
| `post_activity` | Masa pemulihan setelah aktivitas |
| `sleep_disruption_candidate` | Kemungkinan gangguan tidur — pantau |
| `sleep_disruption` | Gangguan pola tidur |
| `high_night_restlessness` | Gelisah saat jam istirahat |
| `sustained_night_arousal` | Detak jantung tinggi saat jam istirahat |
| `model_stress_pattern` | Pola mirip stres (eksperimen) |
| `model_agrees` | (tidak perlu ditampilkan) |
| `no_baseline` | Belum ada data pola pribadi — perlu kalibrasi |
| `insufficient_samples` | Data sensor tidak cukup |
| `stale_data` | Sensor tidak mengirim data |
| `fallback_rules` | (internal; tidak perlu ditampilkan) |
| `missing_feature:<nama>` | Data sensor tidak lengkap |

---

## 6. Notifikasi ke caregiver (deduplikasi)

Status keluar tiap menit, tetapi notifikasi tidak boleh dikirim tiap menit. Aturan yang sudah diuji ada di class `AlertDeduplicator` (`src/ml/simulate_stream.py`). Pakai ulang class itu, atau tiru logikanya:

| Notifikasi | Kapan |
|---|---|
| `alert` — "Pola fisiologis abnormal — perlu pemeriksaan caregiver" | saat status **masuk** `elevated`. Episode baru dalam 30 menit setelah alert terakhir tidak dikirim ulang |
| `resolved` — "Pola fisiologis kembali stabil" | saat status turun dari `elevated` ke `normal`/`monitor` |
| `device_check` — "Data sensor tidak cukup — periksa perangkat" | **sekali per gangguan**, saat `insufficient_data` bertahan 2 menit, setelah data pernah bagus |
| (tidak ada) | `monitor` hanya terlihat di status app, tidak dikirim sebagai notifikasi |

Satu `AlertDeduplicator` per pasien.

---

## 7. Hal produk yang harus diketahui

| Hal | Dampak ke backend/app |
|---|---|
| **Warm-up 5 menit** | 5 menit pertama setelah jam dipakai statusnya `insufficient_data` (window belum penuh). Tampilkan "mengumpulkan data". |
| **Latensi alert ±5–10 menit** | Window 5 menit + syarat 3 menit berturut-turut (siang) atau ±10 menit (tidur). Ini disengaja untuk mengurangi alarm palsu. |
| **Baseline pribadi wajib** | Tanpa baseline, status selalu `insufficient_data` (reason `no_baseline`). Pasien baru perlu **kalibrasi**: 30 menit pertama sesi pertamanya dalam kondisi tenang. Untuk demo, baseline pasien `SIM01–SIM03` sudah ada di `data/baselines/synthetic_baselines.json`. Backend perlu menyimpan baseline per pasien (tabel/JSON) dan menyediakan alur onboarding. |
| **Zona waktu pasien** | Anomali tidur dinilai pada jam 22:00–06:00 **waktu lokal pasien**. Default `Asia/Jakarta`. Kirim `feats["timezone"]` kalau pasien di zona lain. |
| **Putus sensor < 2 menit** | Sengaja tidak memicu `insufficient_data` (putus BLE sebentar tidak perlu membuat panik). |
| **Gerak siang vs malam** | Gerak siang = aktivitas (normal). Gerak terus-menerus di jam tidur = gangguan tidur. |
| **Data sensitif** | `patient_id` pseudonim, akses status dibatasi ke caregiver yang berwenang, data hanya dibagikan sesuai persetujuan (spec Step 12). |

---

## 8. Data uji untuk integrasi

Generate data simulasi (±1 menit), lalu putar ke endpoint backend:

```bash
python -m src.pipeline --source synthetic --regenerate
python -m src.simulator.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --post http://localhost:8000/ingest
```

`--speed 30` = 30× lebih cepat (sesi 2 jam selesai ±4 menit). Untuk test cepat logika backend, jam server perlu mengikuti waktu data (gunakan timestamp record terakhir + 1 detik sebagai `now`), atau putar dengan `--speed 1`.

| File (`data/raw/synthetic/`) | Yang harus terlihat di app |
|---|---|
| `SIM01_normal.csv` | hijau terus (skenario A spec) |
| `SIM01_hr_elevation.csv` | kuning → merah + 1 alert, lalu resolved (skenario B) |
| `SIM01_hrv_drop.csv` | kuning → merah + 1 alert |
| `SIM01_high_activity.csv` | hijau walau HR 120 (olahraga) |
| `SIM01_missing_data.csv` | abu-abu saat sensor lepas + device_check (skenario C) |
| `SIM01_bad_quality.csv` | hijau; record rusak tidak membuat crash |
| `SIM01_sleep_disruption.csv` | malam: merah + 1 alert saat gangguan 15 menit; kuning saja saat bangun singkat |
| `SIM01_mixed_demo.csv` | semua kasus dalam 2 jam — **file demo utama** |

Hasil yang benar untuk setiap file (status per menit dan notifikasi) sudah ada di `reports/stream/SIM01_<sesi>_rules_timeline.csv` dan plotnya `reports/stream/SIM01_<sesi>_rules.png`. Backend bisa dibandingkan langsung dengan file itu.

---

## 9. Checklist integrasi (spec Bagian 8)

- [ ] Endpoint health memberikan respons.
- [ ] Record valid diterima dan disimpan; record rusak tidak membuat crash.
- [ ] Status per pasien dihitung tiap 60 detik, termasuk untuk pasien yang berhenti mengirim data.
- [ ] Output sesuai format Bagian 5.
- [ ] App menampilkan 4 status dengan teks Bagian 5; `insufficient_data` tidak pernah tampil sebagai normal.
- [ ] Notifikasi tidak dikirim berulang (Bagian 6).
- [ ] Skenario normal, abnormal, sensor terputus, dan tidur berhasil diuji dengan file Bagian 8.
- [ ] Kegagalan service ML ditangani (tampilkan "data tidak tersedia", bukan normal).
- [ ] Tidak ada kata PTSD/diagnosis/risiko % di UI.

---

## 10. Referensi

| Dokumen | Isi |
|---|---|
| `Runit.md` (folder hackathon) | cara menjalankan dan testing |
| `docs/data_contract.md` | detail kolom, flag kualitas, fitur |
| `docs/ml_target.md` | target, kategori, anomali tidur (Bagian 8), mode (Bagian 9) |
| `src/ml/simulate_stream.py` | contoh alur lengkap yang sudah teruji |
| `reports/stream_rehearsal_rules.md` | hasil gladi: 32/32 cek lulus |
