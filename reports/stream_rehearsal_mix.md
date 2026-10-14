# Gladi bersih offline — mode `mix`

> Data sintetis. Membuktikan alur dan logika sistem, bukan akurasi klinis.

Alur per tick 60 s: record diterima (urut `received_at`) → `validate_sensor_data` → `clean_sensor_data` → `compute_window_features(window_end=now)` → `Predictor.predict` → deduplikasi notifikasi. Kolom `gt_*` dibuang sebelum masuk sistem.

## Hasil: 13/13 cek lulus

| sesi | cek | lulus | detail |
|---|---|---|---|
| mixed_demo | B: exercise -> no alert | ya | levels seen: ['normal'] |
| mixed_demo | B: arousal -> elevated | ya | first elevated 300 s after onset |
| mixed_demo | B: one alert per episode (dedup) | ya | 1 alert notification(s) |
| mixed_demo | B: hrv_drop -> >= monitor | ya | levels seen: ['elevated_physiological_anomaly', 'monitor', 'normal'] |
| mixed_demo | C: disconnected -> insufficient_data | n/a | n/a: gap 90 s < 120 s |
| mixed_demo | C: disconnected -> device-check notification | n/a | n/a: gap 90 s < 180 s (0 sent) |
| mixed_demo | C: off_wrist -> insufficient_data | ya | 3/3 ticks, gap 240 s |
| mixed_demo | C: off_wrist -> device-check notification | ya | 1 notification(s) |
| mixed_demo | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| mixed_demo | A: never 'normal' without good data | ya | normal only when data_quality == good |
| sleep_disruption | B: sleep_disruption -> elevated | ya | first elevated 600 s after onset |
| sleep_disruption | B: one alert per episode (dedup) | ya | 1 alert notification(s) |
| sleep_disruption | B: brief wake -> at most monitor, no alert | ya | levels seen: ['monitor', 'normal'], 0 alert(s) |
| sleep_disruption | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| sleep_disruption | A: never 'normal' without good data | ya | normal only when data_quality == good |

## Notifikasi ke caregiver

| sesi | menit | tipe | pesan |
|---|---|---|---|
| SIM01_mixed_demo | 50 | alert | Pola fisiologis abnormal — perlu pemeriksaan caregiver |
| SIM01_mixed_demo | 58 | resolved | Pola fisiologis kembali stabil |
| SIM01_mixed_demo | 84 | alert | Pola fisiologis abnormal — perlu pemeriksaan caregiver |
| SIM01_mixed_demo | 88 | resolved | Pola fisiologis kembali stabil |
| SIM01_mixed_demo | 97 | device_check | Data sensor tidak cukup — periksa perangkat |
| SIM01_sleep_disruption | 130 | alert | Pola fisiologis abnormal — perlu pemeriksaan caregiver |
| SIM01_sleep_disruption | 144 | resolved | Pola fisiologis kembali stabil |

## Aturan notifikasi

- `alert`: saat status masuk `elevated_physiological_anomaly`. Episode baru dalam 30 menit setelah alert terakhir tidak dikirim ulang (mencegah notifikasi berulang saat status naik-turun).
- `resolved`: saat status turun dari `elevated` ke `normal`/`monitor`.
- `device_check`: sekali per gangguan, saat `insufficient_data` bertahan 2 tick, setelah data pernah `good` (tidak saat window pertama masih terisi).
- Gap lebih pendek dari 120 s tidak terlihat sebagai `insufficient_data` (disengaja: putus BLE singkat tidak perlu membuat caregiver panik). Cek seperti itu ditandai n/a.
- `monitor` tidak mengirim notifikasi, hanya terlihat di status app.

Plot dan timeline per sesi: `reports/stream/<sesi>_mix.png` dan `_timeline.csv`.
