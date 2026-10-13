# Jalur B — Dataset publik untuk eksperimen ML

Diperiksa: 2026-10-09. Verifikasi ulang lisensi di halaman resmi sebelum dipakai.

## Kesimpulan

- Tidak ditemukan dataset publik yang bisa diunduh bebas dengan sinyal wearable **dan** label episode PTSD.
- Dataset PTSD wearable terbaik (Sadeghi et al., 2022) hanya tersedia lewat permintaan ke IRB.
- Dataset stres umum (WESAD, Nurse Stress) bisa dipakai untuk eksperimen pipeline fitur dan model "arousal fisiologis", tetapi **bukan** untuk mengklaim deteksi episode PTSD.

## Kandidat

### 1. PTSD hyperarousal — Sadeghi et al., PLOS ONE 2022 (paling relevan)

- Subjek: 99 veteran AS dengan diagnosis PTSD, beberapa hari, kondisi naturalistik.
- Sinyal: HR dan akselerasi dari smartwatch komersial, 1 Hz.
- Label: hyperarousal **dilaporkan sendiri** (double-tap di jam), bukan label klinisi. Window 60 s, overlap 30 s; window berlabel hyperarousal jika ada event di dalamnya.
- Ukuran: 530 window hyperarousal vs 13.554 non-hyperarousal (sangat imbalanced).
- Hasil acuan: XGBoost, akurasi >83%, AUC 0,70.
- Akses: **tidak publik.** Data mengandung info sensitif dan peserta tidak setuju data dibagikan publik. Permintaan lewat Texas A&M Human Research Protection Program, referensi IRB2017-0210D.
- Catatan: format sinyal (HR + akselerasi 1 Hz) cocok dengan kolom `heart_rate` dan `accel_x/y/z` di skema proyek.

### 2. WESAD — Schmidt et al., ICMI 2018

- Subjek: 15, lab.
- Sinyal: chest RespiBAN (ECG, EDA, EMG, respirasi, suhu, ACC) dan wrist Empatica E4 (BVP 64 Hz, EDA 4 Hz, suhu 4 Hz, ACC 32 Hz).
- Label: kondisi protokol lab (neutral, stress dari Trier Social Stress Test, amusement).
- Lisensi: tidak ada lisensi CC formal yang ditemukan. Sumber sekunder menyebut hanya untuk riset akademik non-komersial. Cek readme di paket data.
- Relevansi PTSD: rendah. Stres akut terinduksi pada subjek sehat, bukan episode PTSD.
- **Status: sudah dipakai di proyek.** Lihat audit lengkap di bawah dan `reports/data_quality_wesad.md`.

### 3. Nurse Stress Dataset — Hosseini et al., Scientific Data 2022 (Dryad)

- Subjek: 15 perawat perempuan, shift rumah sakit selama COVID-19, ~1.250 jam.
- Sinyal: Empatica E4 (EDA, HR, suhu kulit, IBI, BVP, ACC), CSV.
- Label: event stres terdeteksi lalu dikonfirmasi lewat survei smartphone (tingkat stres dilaporkan sendiri + penyebab).
- Akses: file `Stress_dataset.zip` (1,16 GB) tercantum di Dryad, walau abstrak menyebut "available upon request". Dryad umumnya merilis data dengan CC0, tetapi halaman dataset tidak menyebut lisensi secara eksplisit — cek ulang.
- Relevansi PTSD: rendah–sedang. Naturalistik (bagus), tetapi stres kerja, bukan episode PTSD.

### 4. PTSD-in-the-Wild

- Database video, bukan sinyal wearable. Tidak cocok untuk Jalur B.

## Checklist sebelum memakai dataset

| Cek | Pertanyaan |
|---|---|
| Lisensi | Boleh komersial? Boleh redistribusi? Wajib sitasi? |
| Format | Sinyal apa, sampling rate berapa, raw atau sudah diproses? |
| Subjek | Berapa orang, populasi klinis atau sehat, lab atau naturalistik? |
| Definisi label | Siapa yang memberi label (self-report, protokol, klinisi)? Granularitas waktu? |
| Episode PTSD vs stres | Apakah label benar-benar episode PTSD, atau hanya stres umum? |

## Rekomendasi langkah berikutnya

1. Ajukan permintaan data Sadeghi et al. lewat IRB Texas A&M (proses panjang; mulai sekarang).
2. Sambil menunggu, pakai WESAD atau Nurse Stress untuk membangun pipeline fitur (HR, RMSSD, EDA, ACC) dan menguji model "arousal fisiologis vs aktivitas". Laporkan sebagai deteksi stres, bukan PTSD.
3. Pakai data sintetis Jalur A untuk menguji end-to-end sampai aplikasi caregiver.
4. Untuk validasi PTSD sesungguhnya, rencanakan pengumpulan data sendiri dengan persetujuan etik dan label yang jelas.

## Sumber

- [Sadeghi et al. 2022, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267749) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9116643/)
- [Sadeghi et al. 2021, HR reactions to PTSD among veterans](https://acelab.tamu.edu/wp-content/uploads/sites/23/2023/08/sadeghi-et-al-2021-understanding-heart-rate-reactions-to-post-traumatic-stress-disorder-ptsd-among-veterans-a.pdf)
- [WESAD, Uni Siegen](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/) · [UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) · [UCF summary](https://complexity.cecs.ucf.edu/wesad-wearable-stress-and-affect-detection/)
- [Nurse Stress dataset, Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f) · [paper, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9159985/)
- [PTSD-in-the-Wild](https://arxiv.org/pdf/2209.14085)

---

## Audit WESAD (dataset yang dipakai)

Diaudit: 2026-10-12, dari `D:\jointsHackathon\WESAD\WESAD` (15 subjek, `SX.pkl` + `SX_E4_Data.zip` + `SX_quest.csv`). Konversi: `python -m src.sources.wesad --root D:/jointsHackathon/WESAD/WESAD`.

| Pertanyaan audit (spec Step 1) | Jawaban |
|---|---|
| Data dari sensor wearable? | Ya. Model hanya memakai wrist Empatica E4 (BVP, ACC, EDA, TEMP). ECG dada hanya dipakai sebagai referensi kualitas HR. |
| Label menunjukkan stres, arousal, atau episode PTSD? | **Stres akut yang diinduksi protokol** (TSST), plus amusement dan meditation, pada 15 subjek sehat di lab. Bukan episode PTSD. Label = kondisi protokol, bukan penilaian per menit. |
| Timestamp bisa diselaraskan dengan label? | Ya. Label 700 Hz di `SX.pkl` sudah sinkron dengan sinyal wrist di file yang sama. Data wrist di pkl adalah potongan persis dari `ACC.csv` E4, sehingga offset ditemukan dengan exact match dan timestamp absolut UTC bisa dipulihkan. Hasilnya 15/15 subjek `e4_exact_match`. |
| Jumlah subjek, sesi, kelas? | 15 subjek, 1 sesi lab per subjek (87–118 menit), 86.868 detik. Window 5 menit: baseline 291, stress 169, meditation 197, amusement 95, transient 623. |
| Lisensi? | Readme resmi hanya meminta sitasi Schmidt et al. 2018 (ICMI). Sumber sekunder menyebut hanya untuk riset akademik non-komersial. Anggap non-komersial sampai ada konfirmasi tertulis. |
| Potensi data leakage? | Lihat daftar di bawah. |

### Risiko leakage dan bias

1. **Subjek:** window dari subjek yang sama sangat mirip. Split wajib per `patient_id` (GroupKFold / leave-one-subject-out).
2. **Window bertumpuk:** window 5 menit dengan langkah 1 menit bertumpuk 80%. Split acak per window membocorkan data test ke train.
3. **Baseline pribadi:** baseline dihitung dari paruh pertama window `baseline`. Window tersebut ditandai `is_baseline_period=True` dan harus dikeluarkan dari evaluasi, karena delta-nya ≈ 0 secara konstruksi.
4. **Urutan protokol:** baseline selalu di awal, stress setelahnya. Fitur yang berkorelasi dengan waktu (misalnya suhu kulit yang naik perlahan) bisa mempelajari urutan, bukan kondisi.
5. **Missing data tidak acak:** HR wrist hilang terutama saat stress (bicara, gestur). Dari 169 window stress hanya 11 yang `good`. Model yang hanya dilatih pada window `good` hampir tidak melihat stress. Sebaliknya, `valid_sample_ratio` sendiri bisa "memprediksi" stress karena artefak gerak, bukan fisiologi.
6. **Kolom `gt_*`:** `gt_label`, `gt_hr_ecg`, dan `gt_hrv_rmssd_ecg` adalah jawaban atau referensi. Jangan dipakai sebagai fitur.

### Kualitas sinyal wrist

| Metrik (15 subjek) | Nilai |
|---|---|
| Coverage HR setelah gate SQI | 35–68% per subjek (rata-rata ~50%) |
| MAE HR wrist vs ECG | 3,8 bpm; 83% detik dalam ±5 bpm |
| Coverage / MAE HR saat stress | 17% / 19,9 bpm |
| Spearman RMSSD wrist vs ECG | 0,39 (lemah); RMSSD wrist sistematis lebih tinggi |
| Baseline pribadi valid | 13/15 subjek (S10 dan S15 < 5 window baseline yang `good`) |

Kesimpulan untuk ML engineer: WESAD cocok untuk mencoba pipeline fitur dan model "arousal fisiologis saat diam", dengan evaluasi leave-one-subject-out. Hasilnya tidak bisa dipakai sebagai bukti deteksi PTSD. Fitur HRV wrist belum tervalidasi.
