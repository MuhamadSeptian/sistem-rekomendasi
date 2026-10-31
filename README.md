# Sistem Rekomendasi Film

Proyek ini adalah aplikasi web berbasis Django yang dirancang untuk menampilkan katalog film dan memberikan rekomendasi film secara personal kepada pengguna. Sistem ini menggabungkan dua pendekatan rekomendasi utama:

- Content-Based Filtering (CBF): menggunakan TF-IDF pada data genre dan ringkasan film
- Collaborative Filtering (CF): menggunakan SVD (Singular Value Decomposition)
- Hybrid Recommender: kombinasi skor CBF dan CF dengan bobot 0.4 dan 0.6

Tujuan utama dari proyek ini adalah membantu pengguna menemukan film yang sesuai dengan minat mereka berdasarkan rating yang telah mereka berikan, sekaligus menyediakan antarmuka yang mudah dipakai untuk menjelajahi katalog film.

### Ukuran Dataset
Dataset yang digunakan dalam proyek ini adalah MovieLens `ml-latest-small`, dengan rincian berikut:

- 9.742 film
- 100.836 rating
- 3.683 tag
- 9.742 mapping movie ID ke TMDB ID

Dataset ini digunakan sebagai sumber data utama untuk proses training dan evaluasi sistem rekomendasi.

---

## Fitur Utama

- Katalog film dengan pencarian berdasarkan judul atau deskripsi
- Halaman detail film lengkap dengan informasi genre, poster, dan sinopsis
- Sistem login dan registrasi pengguna
- Fitur memberi rating film dari 1 sampai 5 bintang
- Dashboard rekomendasi personal berdasarkan riwayat rating pengguna
- Kombinasi rekomendasi hybrid untuk hasil yang lebih baik pada kondisi cold-start dan pengguna dengan data rating yang cukup
- Panel admin untuk melihat rekomendasi dan riwayat rating pengguna
- Proses impor data film dari TMDB dan MovieLens
- Script evaluasi dan visualisasi performa model recommender

---

## Teknologi yang Digunakan

- Django 6.x
- SQLite (database default)
- Python
- scikit-learn
- NumPy
- SciPy
- Pandas
- matplotlib / seaborn
- python-dotenv
- TMDB API

---

## Arsitektur Proyek

Struktur utama proyek adalah sebagai berikut:

```text
Sistem rekomendasi/
├── config/                  # konfigurasi Django (settings, urls, wsgi, asgi)
├── movies/                  # aplikasi utama proyek
│   ├── management/          # command custom Django untuk import, evaluasi, visualisasi
│   ├── migrations/          # migrasi database
│   ├── templates/           # template HTML
│   ├── templatetags/        # custom template tags
│   ├── forms.py             # form login dan register
│   ├── models.py            # model Genre, Movie, UserRating
│   ├── recommender.py       # engine rekomendasi hybrid
│   ├── urls.py              # routing aplikasi
│   ├── views.py             # view untuk catalog, detail, auth, dashboard
│   └── tests.py             # test aplikasi
├── static/                  # file CSS/JS frontend
├── templates/               # template global
├── ml-latest-small/         # dataset MovieLens kecil
├── Script/                  # script pendukung, evaluasi, visualisasi, dan eksperimen
├── manage.py                # entry point Django
├── requirements.txt         # daftar dependency
├── db.sqlite3               # database SQLite lokal
├── .env.example             # contoh environment variable (jika dibuat)
├── README.md                # dokumentasi proyek
└── ...
```

### Penjelasan modul utama

- `config/settings.py`  
  Mengatur konfigurasi proyek Django, database, static files, serta variabel lingkungan seperti `SECRET_KEY` dan `TMDB_API_KEY`.

- `movies/models.py`  
  Mendefinisikan model `Movie`, `Genre`, dan `UserRating` yang digunakan untuk menyimpan data film dan rating pengguna.

- `movies/recommender.py`  
  Mengimplementasikan mesin rekomendasi hybrid.  
  Di sini sistem melakukan:
  - pemodelan konten film dari judul, genre, dan overview
  - pembelajaran pola dari rating pengguna
  - penggabungan hasil CBF dan CF secara weighted hybrid

- `movies/views.py`  
  Menangani logika aplikasi seperti katalog film, detail movie, login/register, rating, dan dashboard rekomendasi.

- `movies/management/commands/`  
  Berisi command-command Django seperti import data TMDB, load dataset MovieLens, evaluasi model, serta script visualisasi hasil.

---

## Cara Kerja Sistem Rekomendasi

### 1. Data film
Data film dapat diambil dari TMDB API atau dari dataset MovieLens. Project ini sudah menyediakan folder `ml-latest-small/` serta command untuk meng-import data.

### 2. Content-Based Filtering
Model ini membangun vektor TF-IDF dari:
- nama film
- genre film
- ringkasan/overview film

Setelah profil pengguna dibangun dari film yang disukai, sistem menghitung kemiripan cosine antara profil pengguna dengan semua film lain.

### 3. Collaborative Filtering
Pada pendekatan ini, sistem membangun matriks rating pengguna-film lalu memprosesnya dengan SVD untuk memprediksi rating yang mungkin diberikan pengguna ke film yang belum ditonton.

### 4. Hybrid Recommendation
Skor gabungan dihitung dengan rumus dasar:

```text
hybrid_score = 0.4 * cbf_score + 0.6 * cf_score
```

Dengan pendekatan ini, sistem dapat menggabungkan kekuatan rekomendasi berdasarkan konten dan pola interaksi pengguna.

---

## Persiapan Environment

Sebelum menjalankan aplikasi, pastikan Python sudah terinstal di sistem Anda.

### 1. Buat virtual environment

```bash
python -m venv venv
```

Pada Windows:

```bash
venv\Scripts\activate
```

Pada Linux/macOS:

```bash
source venv/bin/activate
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

### 3. Siapkan variabel lingkungan
Project ini membaca `SECRET_KEY` dan `TMDB_API_KEY` dari environment.

Contoh:

```bash
set SECRET_KEY=your-secret-key
set TMDB_API_KEY=your-tmdb-api-key
```

Pada Linux/macOS:

```bash
export SECRET_KEY=your-secret-key
export TMDB_API_KEY=your-tmdb-api-key
```

> Gunakan API key dari TMDB jika Anda ingin mengambil data film secara otomatis dari layanan TMDB.

---

## Menjalankan Proyek

### 1. Migrasi database

```bash
python manage.py migrate
```

### 2. Import data film
Ada dua cara umum:

#### Opsi A: import dari MovieLens
```bash
python manage.py load_movielens
```

#### Opsi B: import dari TMDB
```bash
python manage.py ingest_tmdb
```

### 3. Jalankan server

```bash
python manage.py runserver
```

Setelah itu, buka browser ke:

```text
http://127.0.0.1:8000/
```

---

## Struktur Routing Aplikasi

Aplikasi memiliki rute utama seperti berikut:

- `/` → katalog film
- `/movie/<tmdb_id>/` → detail film
- `/dashboard/` → rekomendasi pengguna
- `/register/` → registrasi akun
- `/login/` → login pengguna
- `/logout/` → logout
- `/admin-panel/` → panel rekomendasi admin

---

## Penggunaan Umum

1. Daftarkan akun baru atau login
2. Jelajahi katalog film
3. Beri rating pada film yang Anda tonton
4. Buka dashboard untuk melihat rekomendasi yang dibuat berdasarkan preferensi Anda
5. Lihat detail film untuk membaca sinopsis dan memantau rating yang sudah diberikan

---

## Catatan Proyek

Proyek ini bukan hanya aplikasi CRUD biasa, tetapi juga merupakan sistem rekomendasi berbasis machine learning dengan pendekatan hybrid. Struktur dan script yang ada menunjukkan bahwa proyek ini dibuat untuk kebutuhan eksperimen, evaluasi performa, dan visualisasi model rekomendasi.

Folder `Script/` dan `movies/management/commands/` berisi banyak command untuk:

- mengevaluasi model
- menghasilkan chart performa
- memvisualisasikan distribusi rating
- membandingkan strategi rekomendasi
- mengekstraksi dan memvalidasi hasil eksperimen

---

## Kebutuhan Pengembangan Lanjutan

Untuk pengembangan berikutnya, beberapa ide yang dapat ditambahkan adalah:

- API REST untuk integrasi frontend modern
- autentikasi dan otorisasi yang lebih aman
- deployment ke production server
- caching rekomendasi agar performa lebih cepat
- eksperimen model lain seperti NMF, SVD++, atau Neural Collaborative Filtering
- fitur filter berdasarkan genre, tahun, rating, atau popularitas

---

## Penutup

Proyek ini merupakan kombinasi antara aplikasi web Django dan sistem rekomendasi hybrid yang cukup lengkap untuk kebutuhan pembelajaran, eksperimen, maupun pengembangan lebih lanjut. Dengan pendekatan CBF + CF + hybrid, aplikasi ini mampu memberi rekomendasi yang lebih relevan dibandingkan satu metode saja.

Jika Anda ingin, proyek ini juga dapat dikembangkan lebih lanjut menjadi aplikasi recommendation service yang lebih kompleks, misalnya dengan frontend React, API REST, atau deployment ke cloud.
