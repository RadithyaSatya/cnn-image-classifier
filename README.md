# AI Image Classification

Aplikasi web Streamlit untuk mengklasifikasikan gambar menggunakan model CNN hasil pelatihan kelompok. Repository ini menyediakan antarmuka dan integrasi inferensi; tidak melakukan training dan tidak menyertakan model contoh. Tanpa file model, aplikasi tetap terbuka dengan status **Model Not Available**.

## Teknologi

Python 3.11, Streamlit, TensorFlow/Keras, Pillow, NumPy, pandas, dan CSS khusus.

## Struktur

```text
image-classification/
├── training/                # Skrip pelatihan dan optimasi CNN
└── web-app/
    ├── app.py               # Alur halaman dan status aplikasi
    ├── requirements.txt
    ├── .python-version      # Versi Python untuk lingkungan lokal/deployment
    ├── .streamlit/config.toml
    ├── config/model_config.json
    ├── models/
    ├── src/
    └── assets/
```

Struktur aplikasi di dalam `web-app/`:

```text
web-app/
├── app.py                    # Alur halaman dan status aplikasi
├── requirements.txt
├── .python-version           # Versi Python untuk lingkungan lokal/deployment
├── .streamlit/config.toml    # Tema dan batas upload server
├── config/model_config.json  # Kontrak input/output model
├── models/                   # Letakkan model .keras/.h5 di sini
├── src/
│   ├── model_loader.py       # Validasi konfigurasi dan cache model
│   ├── preprocessing.py      # Validasi file dan transformasi gambar
│   ├── predictor.py          # Inferensi dan validasi output
│   └── ui_components.py      # Komponen tampilan
└── assets/                   # Aset tambahan bila diperlukan
```

## Menjalankan secara lokal

Gunakan Python 3.11 untuk kompatibilitas TensorFlow. Jalankan dari folder proyek:

```bash
cd image-classification/web-app
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit. File JPG, JPEG, dan PNG dapat diunggah hingga 10 MB. Validasi aplikasi menggunakan `max_upload_mb` dan server Streamlit menggunakan `.streamlit/config.toml`; jika ingin mengubah batasnya, samakan keduanya.

## Memasang model kelompok

1. Salin model hasil training ke `models/model.keras` (atau file `.h5`). Jangan mengunggah model yang tidak dipercaya karena deserialisasi model dapat menjalankan kode.
2. Sesuaikan `config/model_config.json` dengan kontrak yang diberikan pembuat model:
   - `model_path`: path relatif di folder `models/`.
   - `image_size`: `[tinggi, lebar]` yang diharapkan model.
   - `color_mode`: `rgb` (3 channel) atau `grayscale` (1 channel).
   - `preprocessing`: `none` (nilai piksel 0–255), `rescale_0_1`, atau `minus_one_to_one`. Cocokkan persis dengan pipeline training; bila normalisasi sudah ada di dalam model, pilih `none`.
   - `class_names`: urutan label **persis** sama dengan indeks output training.
   - `output_type`: `multiclass_softmax` untuk satu nilai probabilitas per kelas, atau `binary_sigmoid` untuk satu nilai probabilitas kelas kedua. Untuk binary sigmoid, `class_names[0]` adalah kelas negatif dan `class_names[1]` kelas positif.
   - `max_upload_mb`: batas file aplikasi, maksimal 10 kecuali batas server juga diubah.
3. Muat ulang halaman. Status **Ready** berarti file tersedia; model dimuat ketika tombol **Analyze Image** ditekan. Jika format, dimensi, atau output tidak cocok, aplikasi menampilkan pesan kesalahan.

Konfigurasi bawaan hanya template. Jangan memakai nama kelas atau preprocessing bawaan sebelum mencocokkannya dengan model kelompok. Output yang berupa *logits* tidak didukung oleh konfigurasi saat ini: minta pembuat model menyediakan output softmax/sigmoid, atau ubah kontrak inferensi secara sadar. Aplikasi tidak menambahkan softmax ke output model dan tidak membuat prediksi acak.

## Menguji prediksi

1. Jalankan aplikasi tanpa model untuk memastikan status **Model Not Available** tampil.
2. Setelah memasang model dan menyamakan konfigurasi, unggah gambar valid lalu tekan **Analyze Image**.
3. Periksa label, confidence, grafik probabilitas, dan waktu inferensi. Bandingkan beberapa contoh berlabel dengan hasil dari notebook anggota kelompok. Confidence adalah keyakinan output model, bukan ukuran akurasi yang dijamin.
4. Uji file rusak, file yang bukan gambar, gambar di atas batas ukuran, serta ganti gambar dan prediksi ulang. Aplikasi memvalidasi konten file, bukan hanya ekstensinya.

Waktu inferensi diukur hanya untuk `model.predict`, tanpa waktu memuat model dan preprocessing.

## Deployment ke Streamlit Community Cloud

1. Simpan proyek di repository GitHub. File model dikecualikan oleh `.gitignore`; jika deployment memerlukan model, unggah file model kelompok melalui cara distribusi yang disetujui dan pastikan tersedia di path `model_path` pada runtime. Untuk model kecil, Anda dapat melacaknya di repository dengan `git add -f models/model.keras`, jika hak distribusinya mengizinkan.
2. Di Streamlit Community Cloud, pilih repository, branch, dan entrypoint `image-classification/web-app/app.py`. Jika repository hanya berisi folder `web-app`, gunakan `app.py`.
3. Pastikan runtime memakai Python 3.11 dan memasang `requirements.txt` proyek. Atur working directory/entrypoint sesuai letak folder proyek.
4. Periksa log build terutama jika paket TensorFlow atau ukuran model melebihi batas resource layanan. Setelah deploy, uji upload dan prediksi dengan gambar berlabel.

Jika model belum didistribusikan ke deployment, halaman tetap tampil dengan status **Model Not Available**.
