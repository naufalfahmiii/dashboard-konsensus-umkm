# Dashboard Streamlit: Perbandingan Fungsi Konsensus UMKM Jawa Timur

Dashboard pendamping naskah *Comparing Unweighted Consensus Functions for Clustering Mixed-Type MSME Data in East Java*
dan laporan akhir Asistensi Hibah Penelitian BIMA. Dashboard ini menampilkan perbandingan enam fungsi konsensus tanpa bobot
(AHC-average/EAC, AHC-complete, Spectral-CA, PAM-CA, MCLA, K-Modes-label) pada ensemble K-Means, K-Prototypes, dan
K-Medoids (jarak Gower), uji ketahanan tiga lapis, simulasi semi-sintetis, profil klaster, serta penempatan UMKM baru.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`. Python 3.10 atau lebih baru disarankan.

## Menu

1. **Beranda**: jawaban singkat, peringkat enam fungsi (data asli dan simulasi), kartu segmen, alur penelitian.
2. **Data dan Praproses**: data (identitas disamarkan), kualitas dan imputasi, perlakuan variabel, Skenario 1 (versi data),
   dan **unggah data baru**.
3. **Perbandingan Fungsi Konsensus**: definisi fungsi, hasil K = 3 dan 4 (dengan deteksi partisi identik), semua K = 2–10,
   protokol K, kepekaan geometri.
4. **Uji Ketahanan**: peta panas L1–L3, rincian tiap lapis, diagram peringkat Friedman–Nemenyi per uji.
5. **Simulasi Semi-Sintetis**: rancangan, hasil per kondisi, Friedman–Nemenyi, konsensus vs base, unduh 180 dataset.
6. **Profil dan Interpretasi**: kartu profil, sebaran MDS, peran variabel (η² dan porsi variansi), jenis usaha.
7. **Cek UMKM Baru**: menempatkan satu UMKM baru ke klaster dengan logika EAC.

Fungsi konsensus dan K untuk menu 1, 6, dan 7 dipilih di panel kiri (bawaan: AHC-average, K = 4).

## Sumber hasil

- **Data bawaan** (`data/`): hasil notebook `Notebook_Perbandingan_Fungsi_Konsensus_UMKM.ipynb` dengan seed 42.
  - `umkm_anonim.csv`: 1.368 UMKM setelah pembersihan; nama usaha, pemilik, NIB, dan ID dinas dihapus.
  - `label_semua_K.csv`: label 3 base dan 6 fungsi untuk K = 2–10.
  - `hasil_konsensus.xlsx`: workbook hasil (tanpa sheet yang memuat identitas).
  - `Dataset_Semi_Sintetis_180.zip`: 180 dataset simulasi.
- **Mode unggah**: alur utama (praproses, 3 base, 6 fungsi, evaluasi K = 2–10) dijalankan ulang pada berkas pengguna
  (sekitar 40 detik untuk 1.368 UMKM). Format mentah dinas atau format bersih seperti `umkm_anonim.csv` diterima.
  Uji ketahanan dan simulasi tetap memakai hasil data bawaan karena memerlukan ratusan pengulangan alur penuh.

Pengujian: mode unggah dengan data mentah 2.000 baris menghasilkan label dan indeks yang identik dengan notebook
(selisih SC/DBI/CHI < 1e-13), dan uji penempatan ulang UMKM baru memberi kesesuaian 99,34% pada AHC-average, K = 4.

## Kerahasiaan data

Data UMKM bersifat rahasia atas permintaan penyedia data. Walaupun identitas sudah dihapus, jangan menerbitkan dashboard
beserta folder `data/` ke layanan publik (misalnya Streamlit Community Cloud dengan repositori publik). Gunakan repositori
privat atau jalankan secara lokal.
