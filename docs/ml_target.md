# Target Prediksi ML — PTSD Guardian MVP (spec Step 6)

Status: draf untuk disetujui tim. Tanggal: 2026-10-13.

> Sistem ini mengestimasi **pola fisiologis yang menyimpang dari baseline pribadi**. Sistem ini tidak mendeteksi episode PTSD dan bukan alat diagnosis.

## 1. Dua target, dua tujuan

| | Target A — Anomali fisiologis | Target B — Stres vs non-stres |
|---|---|---|
| Tujuan | Mesin utama demo dan aplikasi caregiver | Eksperimen eksploratif: apakah ML memberi nilai tambah |
| Data | Sintetis (`data/features/synthetic_features.csv`) | WESAD (`data/features/wesad_features.csv`) |
| Metode | Rule-based (`rules_v1`) | Model ML (HistGradientBoosting / Random Forest), dibandingkan dengan rules |
| Label | `gt_expected_alert`, `gt_alert_level` | `gt_label == "stress"` = 1; `baseline`, `amusement`, `meditation` = 0 |
| Label dipakai untuk | **Evaluasi saja.** Aturan tidak dilatih dari label | Training + evaluasi, split per subjek |
| Dipakai di demo? | Ya | Hanya jika terbukti lebih baik dari rules di Target B, dan tetap dengan klaim "stres" |

Kenapa tidak melatih ML di data sintetis: labelnya dibuat dari aturan simulator. Model yang dilatih di sana hanya meniru simulator (spec Step 6: "jangan membuat label HR tinggi = PTSD").

## 2. Unit prediksi

- Satu prediksi per **window 5 menit**, dihitung tiap **1 menit**.
- Input: satu baris fitur dari `compute_window_features()` (lihat `docs/data_contract.md` §4).
- Fitur model (`MODEL_FEATURES`, urutan tetap): `hr_mean, hr_std, hr_delta_pct, hrv_rmssd_mean, hrv_delta_pct, accel_magnitude_std, valid_sample_ratio`.
  - Target B membuang `valid_sample_ratio`, karena HR yang hilang di WESAD berkorelasi dengan stress (artefak gerak). Model bisa mempelajari "sensor gagal", bukan fisiologi.
- Rules boleh memakai riwayat status window sebelumnya (persistensi). Model ML hanya memakai satu window.

## 3. Kategori output

| `risk_level` | Arti | Kapan |
|---|---|---|
| `normal` | Pola konsisten dengan baseline, atau perubahan dijelaskan oleh aktivitas fisik | data `good`, tidak ada penyimpangan saat diam |
| `monitor` | Penyimpangan dari baseline saat diam, belum bertahan | 1–2 window abnormal berturut-turut |
| `elevated_physiological_anomaly` | Penyimpangan saat diam yang bertahan | ≥ 3 window abnormal berturut-turut (≈ 7 menit data) |
| `insufficient_data` | Sistem tidak bisa menilai | `data_quality != "good"`, baseline tidak valid, atau fitur wajib kosong |

Aturan prioritas: `insufficient_data` selalu menang. Data hilang tidak boleh tampil sebagai `normal`.

Pemetaan ke label evaluasi sintetis:

| `gt_alert_level` | Ekspektasi minimum |
|---|---|
| `alert` (`arousal`) | mencapai `elevated_physiological_anomaly` |
| `warning` (`hrv_drop`) | mencapai `monitor` |
| `none` | tetap `normal` atau `insufficient_data` |

## 4. Kontrak output (rules dan model sama persis)

```json
{
  "patient_id": "SIM01",
  "window_end": "2026-01-06T02:35:00Z",
  "risk_level": "monitor",
  "risk_score": null,
  "reasons": ["hr_above_baseline", "at_rest"],
  "model_version": "rules_v1",
  "data_quality": "good"
}
```

- `risk_score`: `null` sampai ada skor yang terkalibrasi dan didefinisikan maknanya. Probabilitas model tidak ditampilkan sebagai "risiko".
- `reasons`: kode singkat yang bisa diterjemahkan app, misalnya `hr_above_baseline`, `hrv_below_baseline`, `activity`, `no_baseline`, `insufficient_samples`, `stale_data`.
- `model_version`: `rules_v1` atau versi dari `models/model_card.json`.

## 5. Evaluasi dan kriteria sukses

### Target A (sintetis, rules)

- Dihitung **per event** (pakai `data/raw/synthetic/events.csv`), bukan per window.
  - Event terdeteksi jika ada status yang sesuai (lihat §3) di window yang overlap dengan `start_s .. peak_end_s + 120 s`.
- False alarm = `monitor`/`elevated` di luar window event `arousal`/`hrv_drop` dan masa recovery-nya. Dilaporkan per jam data.
- Pemisahan tuning/tes: ambang disetel di pasien SIM01–SIM02, dilaporkan di SIM03 (atau pasien baru yang di-generate).

| Kriteria (pasien tes) | Target |
|---|---|
| Recall event `arousal` → `elevated` | 100% |
| Recall event `hrv_drop` → ≥ `monitor` | ≥ 80% |
| Alert di `normal`, `high_activity`, `missing_data`, `bad_quality` | 0 |
| Window gap/off-wrist berstatus `insufficient_data` | 100% |

### Target B (WESAD, eksploratif)

- Leave-one-subject-out (15 fold). Window dengan `is_baseline_period=True` dan `gt_label_purity < 1` dikeluarkan.
- Metrik: precision, recall, F1 kelas stress, confusion matrix, false alarm rate (proporsi window non-stres yang diprediksi stres), dilaporkan sebagai rata-rata ± std per fold.
- Pembanding wajib: rules dijalankan pada window yang sama.
- Tidak ada ambang sukses. Hasil dilaporkan apa adanya. Keterbatasan yang sudah diketahui: hanya 11 window stress yang `good` (lihat `reports/data_quality_wesad.md`).

## 6. Batas klaim

Boleh dikatakan:
- "Pola fisiologis menyimpang dari baseline pribadi saat istirahat."
- "Perlu pemeriksaan caregiver."
- "Data sensor tidak cukup untuk menilai."
- Untuk Target B: "Eksperimen klasifikasi stres akut di lab (WESAD), 15 subjek sehat."

Tidak boleh dikatakan:
- "Mendeteksi episode PTSD" / "serangan PTSD" / "diagnosis".
- "Risiko PTSD x%".
- Menyebut status `normal` saat data `insufficient_data`.
- Hasil data sintetis sebagai bukti akurasi.

Teks app yang disarankan:

| `risk_level` | Teks untuk caregiver |
|---|---|
| `normal` | Pola fisiologis stabil |
| `monitor` | Pola fisiologis berubah — pantau |
| `elevated_physiological_anomaly` | Pola fisiologis abnormal — perlu pemeriksaan caregiver |
| `insufficient_data` | Data sensor tidak cukup — periksa perangkat |

## 7. Yang perlu disetujui tim

1. Dua target terpisah seperti di atas, dengan rules sebagai default demo.
2. Empat kategori output dan teks app di §6.
3. Kontrak output di §4, termasuk `risk_score = null`.
4. Kriteria sukses Target A di §5.
