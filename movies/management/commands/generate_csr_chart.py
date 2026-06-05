import os
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from movies.models import UserRating
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

class Command(BaseCommand):
    help = 'Menghasilkan gambar PNG visualisasi CSR dan Sparsity'

    def handle(self, *args, **kwargs):
        self.stdout.write("Menghitung data untuk visualisasi...")
        
        ratings_qs = UserRating.objects.all().values("user_id", "movie_id", "rating")
        df = pd.DataFrame(ratings_qs)
        
        n_users = df['user_id'].nunique()
        n_movies = df['movie_id'].nunique()
        n_ratings = len(df)
        total_cells = n_users * n_movies
        
        # Kalkulasi estimasi Memori (dalam MB)
        dense_mem_mb = (total_cells * 8) / (1024 * 1024)
        csr_mem_mb = ((n_ratings * 8) + (n_ratings * 4) + ((n_users + 1) * 4)) / (1024 * 1024)
        
        # Setup Figure Matplotlib (1 Baris, 2 Kolom)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ==========================================
        # 1. KIRI: Visualisasi Bar Chart Penghematan
        # ==========================================
        labels = ['Tanpa Pemadatan\n(Tabel Array Biasa)', 'Dengan Pemadatan\n(Algoritma CSR)']
        values = [dense_mem_mb, csr_mem_mb]
        colors = ['#e74c3c', '#2ecc71'] # Merah dan Hijau
        
        bars = ax1.bar(labels, values, color=colors, width=0.5)
        ax1.set_ylabel('Konsumsi Memori RAM (Megabytes)', fontsize=12)
        ax1.set_title('Perbandingan Efisiensi Beban Memori Komputer', fontsize=14, pad=15)
        
        # Tulis angka MB di atas bar
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} MB', 
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        # ==========================================
        # 2. KANAN: Visualisasi Matriks (Spy Plot)
        # ==========================================
        # Mengambil sampel 200 User pertama dan 200 Film pertama yang dinilai agar titiknya terlihat
        sample_users = df['user_id'].unique()[:100]
        sample_movies = df['movie_id'].unique()[:100]
        
        sample_df = df[(df['user_id'].isin(sample_users)) & (df['movie_id'].isin(sample_movies))]
        
        # Petakan ID asli ke urutan indeks 0-200
        user_cat = sample_df['user_id'].astype('category').cat.codes
        movie_cat = sample_df['movie_id'].astype('category').cat.codes
        
        if not sample_df.empty:
            max_u = user_cat.max() + 1
            max_m = movie_cat.max() + 1
            sample_csr = csr_matrix((sample_df['rating'], (user_cat, movie_cat)), shape=(max_u, max_m))
            
            # ax.spy digunakan secara khusus dalam metmatika untuk melihat matriks bolong/sparse
            ax2.spy(sample_csr, markersize=3, color='#2980b9', aspect='auto')
        
        ax2.set_title('Peta Kekosongan Data Matriks (Sparsity Pattern)\nSampel Ukuran 200 User × 200 Film', fontsize=14, pad=15)
        ax2.set_xlabel('Indeks Kolom (Film)', fontsize=12)
        ax2.set_ylabel('Indeks Baris (User)', fontsize=12)
        
        # Tambahkan legend manual di ujung
        ax2.plot([], [], 'o', color='#2980b9', markersize=6, label='Sel Berisi Nilai (Rating)')
        ax2.plot([], [], 's', color='white', markeredgecolor='gray', markersize=6, label='Sel Kosong (Nol)')
        ax2.legend(loc='lower right', framealpha=1)

        # Rendering dan Save Gambar
        plt.tight_layout()
        output_path = os.path.join(os.getcwd(), 'pemadatan_csr.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.stdout.write(self.style.SUCCESS(f"\n[SUCCESS] Gambar visualisasi berhasil dieksport ke: {output_path}"))
