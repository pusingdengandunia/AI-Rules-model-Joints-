# Backend Neuvana MVP

FastAPI di atas modul ML repo ini (mode `rules` saja). Acuan: PRD Neuvana MVP dan `docs/handoff_fullstack.md`.

## Menjalankan

```powershell
pip install -r requirements.txt
python -m src.pipeline --source synthetic --regenerate     # data simulator + baseline SIM01–SIM03
uvicorn backend.main:app --host 0.0.0.0 --port 8000         # dokumentasi API: http://localhost:8000/docs
```

Putar skenario demo (dengan jeda sesuai data, jadi sensor lepas ikut terlihat abu-abu di aplikasi):

```powershell
python -m backend.tools.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30                       # pasien SIM01
python -m backend.tools.replay data/raw/synthetic/SIM01_mixed_demo.csv --speed 30 --patient P-1A2B3C4D  # akun demo Rani
```

Test: `python -m pytest -q tests/test_backend_api.py tests/test_backend_replay.py`. Test replay membandingkan
kedelapan sesi SIM01 dengan `reports/stream/*_rules_timeline.csv` per menit, termasuk notifikasinya.

| Env | Default | Isi |
|---|---|---|
| `NEUVANA_DB` | `backend_data/neuvana.db` | file SQLite |
| `PUBLIC_WEB_URL` | `http://localhost:8081` | alamat Expo web; tautan undangan `/join/<kode>`, halaman orang terdekat `/w/<token>` |
| `REPLAY_SPEED` | `30` | kecepatan simulator bila header `X-Replay-Speed` tidak dikirim |
| `NOTIFY_MODE` | `live` | `dry_run` = catat di tabel `notifications`, tidak mengirim |
| `INGEST_KEY` | kosong | bila diisi, `POST /ingest` butuh header `X-Ingest-Key` atau token pasien itu sendiri |
| `EXPO_ACCESS_TOKEN` | kosong | bila push Expo memakai access token |
| `WA_TOKEN`, `WA_PHONE_NUMBER_ID` | kosong | WhatsApp Cloud API; tanpa ini pesan WA dicatat `skipped` |
| `WA_TEMPLATE_ALERT/RESOLVED/DEVICE_CHECK/UNSAFE` | `neuvana_alert`, … | nama template yang disetujui; parameter body: `{{1}}` nama, `{{2}}` tautan web |
| `CORS_ORIGINS` | `*` | dipisah koma |

## Cara kerja

- **Satu pintu data:** `POST /ingest` menerima satu record atau list (Health Connect sinkron per batch). Semua
  record disimpan apa adanya dengan `quality_flag`; record rusak tidak membuat crash.
- **Status tiap 60 detik per pasien**, dengan rantai yang sama persis dengan `simulate()` di
  `src/ml/simulate_stream.py`: `validate → clean → compute_window_features(window_end=now) → Predictor("rules") →
  AlertDeduplicator`. Satu `Predictor` untuk semua pasien dalam satu proses. **Jalankan satu worker saja**
  (state rules ada di memori).
- **Dua jam:**
  - `wall` (jam tangan sungguhan): `received_at` diisi server, tick tiap pergantian menit.
  - `replay` (record `source: "simulator"` atau header `X-Replay-Speed`): jam mengikuti `received_at` data. Bila
    data berhenti > 2 detik, jam tetap berjalan sesuai kecepatan replay sehingga abu-abu muncul secara langsung.
    Data yang mundur > 5 menit atau `session_id` baru = sesi baru (`PREDICTOR.reset`).
- **Jarak sampel Health Connect:** kelengkapan window dinilai terhadap jarak sampel perangkat itu sendiri
  (`expected_samples = 300 / jarak median`, seperti `iter_windows`). Syarat data ≤ 60 detik dari akhir window tetap
  berlaku, jadi sinkronisasi yang terlambat tetap tampil abu-abu (keputusan tim no. 7).
- **Pemberitahuan:** `alert`/`resolved` ke orang terdekat, `device_check` ke pengguna dan orang terdekat (aturan
  `AlertDeduplicator`). `unsafe` dikirim setiap kali tombol ditekan. Kuning tidak pernah menjadi push; pengguna
  hanya menerima event `greeting` lewat WebSocket. Setiap pemberitahuan masuk feed in-app, push Expo ke akun yang
  mendaftarkan token, dan WhatsApp ke orang terdekat tanpa aplikasi.

## Status untuk aplikasi

`GET /patients/{id}/status`, `GET /web/{token}`, dan event `status` di WebSocket memakai bentuk yang sama:
field output modul ML (handoff Bagian 5) ditambah:

```json
{"patient_id": "P-1A2B3C4D", "window_end": "2026-01-11 02:54:00+00:00",
 "risk_level": "elevated_physiological_anomaly", "reasons": ["hr_above_baseline", "at_rest"],
 "model_version": "rules_v1", "data_quality": "good",
 "display": {"state": "elevated_physiological_anomaly", "color": "red",
             "text": "Pola tubuh menyimpang, perlu perhatian",
             "reasons_text": ["Detak jantung di atas pola biasanya", "Terjadi saat sedang diam"]},
 "guide": {"active": true, "step": 2, "total": 6},
 "heart_rate": {"value": 96, "timestamp": "2026-01-11T02:53:59Z"},
 "clock_now": "2026-01-11T02:54:01.000Z"}
```

Aplikasi cukup merender `display`. `display.state` berisi salah satu dari empat `risk_level` atau keadaan khusus berikut:

| `state` | Kapan | Teks |
|---|---|---|
| `waiting` | belum pernah ada data | Menunggu data dari jam tangan |
| `collecting` | 5 menit pertama setelah data mulai masuk, atau setelah restart | Mengumpulkan data |
| `calibrating` | kalibrasi berjalan (`calibration_remaining_s` ikut dikirim) | Mengenali pola tubuhmu |
| `needs_calibration` | reason `no_baseline` | Belum ada pola pribadi — perlu kalibrasi |
| `unavailable` | modul ML gagal, atau status terakhir > 2 menit | Data tidak tersedia |

Semua teks ada di `GET /meta`. Isi panduan 6 langkah ada di aplikasi; backend hanya mengirim nomor langkahnya, naik
satu langkah per menit selama merah.

## Endpoint

| Endpoint | Siapa | Fungsi |
|---|---|---|
| `GET /health`, `GET /meta` | publik | cek layanan; teks status dan reasons |
| `POST /accounts` | publik | `{role: patient\|caregiver, nickname, timezone: WIB\|WITA\|WIT, demo_baseline?: "SIM01"}` → token |
| `GET /me` | akun | profil, pasien yang didampingi |
| `POST /devices/push-token` | akun | simpan token Expo |
| `POST /ingest` | perangkat | record atau list record |
| `GET /patients/{id}/status` · `/history` · `/notifications` | pengguna, pendamping | status (sudah difilter privasi), riwayat per menit, feed |
| `GET /patients/{id}`, `PATCH /patients/{id}/settings` | pengguna | nama, zona waktu, `share_reasons`, `share_heart_rate`, `notify_me_first`, kontak darurat |
| `POST /patients/{id}/calibration` | pengguna | `{action: start\|finish\|cancel}`; selesai otomatis setelah 30 menit |
| `POST /patients/{id}/unsafe` | pengguna | beri tahu orang terdekat; mengembalikan kontak darurat |
| `POST /patients/{id}/session/reset` | pengguna | sesi rekaman baru |
| `GET /patients/{id}/notes` | pengguna | catatan untuk terapis |
| `POST /circles/{id}/invites` | pengguna | kode 6 karakter + tautan, berlaku 7 hari, sekali pakai |
| `POST /circles/join` | pendamping / publik | dengan token pendamping: tautkan akun. Tanpa token: `{code, name, phone}` → `web_token` |
| `GET /circles/{id}/members`, `DELETE /circles/{id}/members/{member_id}` | pengguna | lihat dan cabut akses (koneksi live langsung ditutup) |
| `GET /web/{token}` | pemegang tautan | data halaman web orang terdekat |
| `WS /ws/patients/{id}?token=` | token akun atau token web | event `status`, `heart_rate`, `notification`, `greeting` (pengguna), `calibration` (pengguna) |

Pasien simulator `SIM01–SIM03` (data sintetis, tanpa akun) bisa dibaca tanpa token agar demo mudah.

## Belum dibuat

- F12, AI koordinator. Semua teks saat ini memakai teks standar di `backend/texts.py`, dan `is_safe_text` sudah
  tersedia untuk memeriksa teks AI nanti.
- `notify_me_first` baru disimpan; perilakunya belum dijalankan.
- Template WhatsApp harus didaftarkan dan disetujui Meta sebelum demo.
