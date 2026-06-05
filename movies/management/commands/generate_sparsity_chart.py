import os
import matplotlib.pyplot as plt
import pandas as pd
from django.core.management.base import BaseCommand
from movies.models import UserRating

class Command(BaseCommand):
    help = 'Menghasilkan gambar visualisasi Sparsity (Kekosongan) dan Long-Tail Interaksi'

    def handle(self, *args, **kwargs):
        self.stdout.write("Menghitung metrik Sparsity dari database...")
        
        # Ambil data rating
        ratings_qs = UserRating.objects.all().values("user_id", "movie_id")
        df = pd.DataFrame(ratings_qs)
        
        if df.empty:
            self.stdout.write(self.style.ERROR("Belum ada data rating."))
            return
            
        n_users = df['user_id'].nunique()
        n_movies = df['movie_id'].nunique()
        n_ratings = len(df)
        total_cells = n_users * n_movies
        
        kekosongan = total_cells - n_ratings
        persentase_kosong = (kekosongan / total_cells) * 100
        persentase_terisi = 100 - persentase_kosong

        # Hitung seberapa banyak rating yang diberikan oleh tiap user
        user_counts = df['user_id'].value_counts().values
        
        # Setup Figure Matplotlib (1 Baris, 2 Kolom)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ==========================================
        # 1. KIRI: Pie Chart Tingkat Sparsity
        # ==========================================
        labels = [f'Sel Kosong / Tidak Ada Interaksi\n({kekosongan:,})', 
                  f'Sel Terisi Rating\n({n_ratings:,})']
        sizes = [kekosongan, n_ratings]
        colors = ['#ecf0f1', '#e74c3c']
        explode = (0, 0.2)  # Tarik potongan 'Terisi' agar menonjol
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, 
                autopct='%1.2f%%', shadow=True, startangle=140, 
                textprops={'fontsize': 11})
        ax1.set_title('Persentase Kekosongan Data (Sparsity Level)', fontsize=14, pad=15)
        
        # ==========================================
        # 2. KANAN: Grafik Long-Tail (Rating per User)
        # ==========================================
        ax2.plot(range(len(user_counts)), user_counts, color='#2980b9', linewidth=2)
        ax2.fill_between(range(len(user_counts)), user_counts, color='#3498db', alpha=0.3)
        ax2.set_title('Distribusi Jumlah Interaksi Tiap User (Long-Tail)', fontsize=14, pad=15)
        ax2.set_xlabel('Pengguna (Diusut dari yang paling aktif ke pasif)', fontsize=12)
        ax2.set_ylabel('Jumlah Film yang Dinilai', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Rendering dan Save Gambar
        plt.tight_layout()
        output_path = os.path.join(os.getcwd(), 'sparsity_interaksi.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.stdout.write(self.style.SUCCESS(f"\n[SUCCESS] Gambar visualisasi Sparsity & Long-tail berhasil dieksport ke: {output_path}"))
