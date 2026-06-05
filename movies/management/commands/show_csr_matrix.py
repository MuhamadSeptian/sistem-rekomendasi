from django.core.management.base import BaseCommand
from movies.models import UserRating, Movie
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

class Command(BaseCommand):
    help = 'Menampilkan demonstrasi Pemadatan Data (CSR Matrix) kepada Dosen'

    def handle(self, *args, **kwargs):
        self.stdout.write("Mengambil data rating dari database...")
        
        # Ambil data rating
        ratings_qs = UserRating.objects.all().values("user_id", "movie_id", "rating")
        df = pd.DataFrame(ratings_qs)
        
        if df.empty:
            self.stdout.write(self.style.ERROR("Belum ada data rating."))
            return
            
        # Hitung statistik matriks global
        n_users = df['user_id'].nunique()
        n_movies = df['movie_id'].nunique()
        n_ratings = len(df)
        
        total_cells = n_users * n_movies
        sparsity = (1.0 - (n_ratings / total_cells)) * 100
        
        # 1. Estimasi kalkulasi ukuran RAM (dalam Megabytes)
        # Dense (Biasanya Float64 makan 8 bytes per sel)
        dense_mem_mb = (total_cells * 8) / (1024 * 1024)
        # CSR Formula: data_array(8 bytes) + indices_array(4 bytes) + indptr_array(4 bytes)
        csr_mem_mb = ((n_ratings * 8) + (n_ratings * 4) + ((n_users + 1) * 4)) / (1024 * 1024)
        
        self.stdout.write("\n" + "="*75)
        self.stdout.write("1. STATISTIK KEKOSONGAN DATA (SPARSITY) & PENGHEMATAN RAM".center(75))
        self.stdout.write("="*75)
        self.stdout.write(f"Total Kolom (Film)         : {n_movies:,}")
        self.stdout.write(f"Total Baris (User)         : {n_users:,}")
        self.stdout.write(f"Total Sel Matriks Tabel    : {total_cells:,} kotak")
        self.stdout.write(f"Sel Terisi Nilai Rating    : {n_ratings:,} ({100-sparsity:.2f}%)")
        self.stdout.write(f"Sel Kosong (Matriks Sparse): {total_cells - n_ratings:,} ({sparsity:.2f}%)\n")
        
        self.stdout.write(f"Estimasi Makan RAM Tabel Biasa (Dense Matrix) : ~{dense_mem_mb:.2f} MB")
        self.stdout.write(f"Estimasi Makan RAM Menggunakan Tabel CSR      :  ~{csr_mem_mb:.2f} MB")
        self.stdout.write(f">> KESIMPULAN: CSR menghemat penyimpanan hingga {(dense_mem_mb/csr_mem_mb):.1f}x lipat!")
        self.stdout.write("="*75)

        # 2. Mini Demonstrasi wujud matriks (Ambil 5 User x 6 Film saja)
        self.stdout.write("\n" + "="*75)
        self.stdout.write("2. ILUSTRASI BEFORE-AFTER MATRIKS BIASA VS MATRIKS CSR".center(75))
        self.stdout.write("="*75)
        
        # Bangun miniatur data untuk pajangan
        dummy_dense = np.array([
            [4.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, 3.5, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0, 0.0, 2.0, 0.0],
        ])
        dummy_sparse = csr_matrix(dummy_dense)

        self.stdout.write("\n[ BEFORE ] Bentuk Tabel Matriks Biasa (Dense):")
        self.stdout.write("Sebuah array 2 Dimensi biasa, semua angka 0.0 ikut disalin ke memori komputer")
        self.stdout.write(str(dummy_dense))
        
        self.stdout.write("\n[ AFTER ] Bentuk Representasi CSR:")
        self.stdout.write("Array 2 Dimensi di atas dihancurkan dan dipadatkan menjadi 3 jalur indeks:")
        self.stdout.write(f" -> Jalur 'Data' (Isi nilai asli) : {dummy_sparse.data}")
        self.stdout.write(f" -> Jalur 'Indices' (Posisi kolom): {dummy_sparse.indices}")
        self.stdout.write(f" -> Jalur 'Indptr' (Batas baris)  : {dummy_sparse.indptr}")

        self.stdout.write("\n[Penjelasan Argumen untuk Dosen]:")
        self.stdout.write("1. Data   : Menyimpan nilai rating yang eksis saja (ex: 4.0, 5.0, 3.5).")
        self.stdout.write("2. Indices: Mengingat koordinat letak posisinya di kolom film ke berapa.")
        self.stdout.write("3. Indptr : Penunjuk baris yang mengelompokkan rating milik masing-masing user.")
        
        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Konsep CSR berhasil ditampilkan untuk sidang!"))
