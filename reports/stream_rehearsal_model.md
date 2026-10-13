# Gladi bersih offline — mode `model`

> Data sintetis. Membuktikan alur dan logika sistem, bukan akurasi klinis.

Alur per tick 60 s: record diterima (urut `received_at`) → `validate_sensor_data` → `clean_sensor_data` → `compute_window_features(window_end=now)` → `Predictor.predict` → deduplikasi notifikasi. Kolom `gt_*` dibuang sebelum masuk sistem.

## Hasil: 18/32 cek lulus

| sesi | cek | lulus | detail |
|---|---|---|---|
| bad_quality | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| bad_quality | A: never 'normal' without good data | ya | normal only when data_quality == good |
| high_activity | B: exercise -> no alert | **TIDAK** | levels seen: ['monitor', 'normal'] |
| high_activity | A: no alert outside episodes | **TIDAK** | 23 tick(s) with monitor/elevated outside episodes |
| high_activity | A: never 'normal' without good data | ya | normal only when data_quality == good |
| hr_elevation | B: arousal -> elevated | **TIDAK** | max normal |
| hr_elevation | B: one alert per episode (dedup) | **TIDAK** | 0 alert notification(s) |
| hr_elevation | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| hr_elevation | A: never 'normal' without good data | ya | normal only when data_quality == good |
| hrv_drop | B: hrv_drop -> >= monitor | **TIDAK** | levels seen: ['normal'] |
| hrv_drop | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| hrv_drop | A: never 'normal' without good data | ya | normal only when data_quality == good |
| missing_data | C: disconnected -> insufficient_data | ya | 1/1 ticks, gap 120 s |
| missing_data | C: disconnected -> device-check notification | n/a | n/a: gap 120 s < 180 s (1 sent) |
| missing_data | C: off_wrist -> insufficient_data | ya | 4/4 ticks, gap 300 s |
| missing_data | C: off_wrist -> device-check notification | ya | 1 notification(s) |
| missing_data | A: no alert outside episodes | **TIDAK** | 5 tick(s) with monitor/elevated outside episodes |
| missing_data | A: never 'normal' without good data | ya | normal only when data_quality == good |
| mixed_demo | B: exercise -> no alert | **TIDAK** | levels seen: ['monitor', 'normal'] |
| mixed_demo | B: arousal -> elevated | **TIDAK** | max normal |
| mixed_demo | B: one alert per episode (dedup) | **TIDAK** | 0 alert notification(s) |
| mixed_demo | B: hrv_drop -> >= monitor | **TIDAK** | levels seen: ['normal'] |
| mixed_demo | C: disconnected -> insufficient_data | n/a | n/a: gap 90 s < 120 s |
| mixed_demo | C: disconnected -> device-check notification | n/a | n/a: gap 90 s < 180 s (0 sent) |
| mixed_demo | C: off_wrist -> insufficient_data | ya | 3/3 ticks, gap 240 s |
| mixed_demo | C: off_wrist -> device-check notification | ya | 1 notification(s) |
| mixed_demo | A: no alert outside episodes | **TIDAK** | 23 tick(s) with monitor/elevated outside episodes |
| mixed_demo | A: never 'normal' without good data | ya | normal only when data_quality == good |
| normal | A: no alert outside episodes | **TIDAK** | 3 tick(s) with monitor/elevated outside episodes |
| normal | A: never 'normal' without good data | ya | normal only when data_quality == good |
| sleep_disruption | B: sleep_disruption -> elevated | **TIDAK** | max monitor |
| sleep_disruption | B: one alert per episode (dedup) | **TIDAK** | 0 alert notification(s) |
| sleep_disruption | B: brief wake -> at most monitor, no alert | ya | levels seen: ['monitor', 'normal'], 0 alert(s) |
| sleep_disruption | A: no alert outside episodes | ya | 0 tick(s) with monitor/elevated outside episodes |
| sleep_disruption | A: never 'normal' without good data | ya | normal only when data_quality == good |

## Notifikasi ke caregiver

| sesi | menit | tipe | pesan |
|---|---|---|---|
| SIM01_missing_data | 17 | device_check | Data sensor tidak cukup — periksa perangkat |
| SIM01_missing_data | 37 | device_check | Data sensor tidak cukup — periksa perangkat |
| SIM01_mixed_demo | 97 | device_check | Data sensor tidak cukup — periksa perangkat |

## Aturan notifikasi

- `alert`: saat status masuk `elevated_physiological_anomaly`. Episode baru dalam 30 menit setelah alert terakhir tidak dikirim ulang (mencegah notifikasi berulang saat status naik-turun).
- `resolved`: saat status turun dari `elevated` ke `normal`/`monitor`.
- `device_check`: sekali per gangguan, saat `insufficient_data` bertahan 2 tick, setelah data pernah `good` (tidak saat window pertama masih terisi).
- Gap lebih pendek dari 120 s tidak terlihat sebagai `insufficient_data` (disengaja: putus BLE singkat tidak perlu membuat caregiver panik). Cek seperti itu ditandai n/a.
- `monitor` tidak mengirim notifikasi, hanya terlihat di status app.

Plot dan timeline per sesi: `reports/stream/<sesi>_model.png` dan `_timeline.csv`.
