import os
import matplotlib.pyplot as plt
import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Count
from movies.models import UserRating, Movie
from movies.recommender import HybridRecommender

class Command(BaseCommand):
    help = 'Generate vertical stacked bar chart for Hybrid Recommender using real data'

    def handle(self, *args, **kwargs):
        self.stdout.write("Mencari user paling aktif di database...")
        
        # 1. Cari user dengan rating terbanyak (User dengan 2.698 rating)
        top_user_dict = UserRating.objects.values('user_id').annotate(rating_count=Count('id')).order_by('-rating_count').first()
        
        if not top_user_dict:
            self.stdout.write(self.style.ERROR("Tidak ada data rating di database."))
            return
            
        user_id = top_user_dict['user_id']
        rating_count = top_user_dict['rating_count']
        
        self.stdout.write(self.style.SUCCESS(f"Ditemukan User paling aktif: ID {user_id} dengan {rating_count} rating."))
        self.stdout.write("Sedang melatih model Hybrid secara penuh (SVD + TF-IDF)... Mohon tunggu...")
        
        # 2. Inisialisasi dan Train Model Hybrid
        hybrid = HybridRecommender(cbf_weight=0.4, cf_weight=0.6)
        hybrid.train()
        
        self.stdout.write("Memprediksi Top-8 film rekomendasi...")
        # 3. Prediksi 8 film teratas untuk user aktif
        results = hybrid.predict_detailed(user_id, k=8)
        
        if not results:
            self.stdout.write(self.style.ERROR("Gagal memprediksi rekomendasi."))
            return

        # Array untuk menyimpan data plotting
        movie_titles = []
        cbf_contributions = []
        cf_contributions = []
        total_scores = []
        
        for res in results:
            tmdb_id = res['tmdb_id']
            movie = Movie.objects.filter(tmdb_id=tmdb_id).first()
            title = movie.title if movie else f"Movie {tmdb_id}"
            
            # Memotong judul film (Truncate) agar rapi jika kepanjangan (maks 15 karakter)
            if len(title) > 15:
                title = title[:13] + "..."
                
            movie_titles.append(title)
            
            # Mendapatkan skor yang sudah dinormalisasi dan dikali bobot dari model
            # res['cbf_score'] dan res['cf_score'] sudah dinormalisasi 0-1, kita kalikan bobotnya
            cbf_val = res['cbf_score'] * hybrid.cbf_weight
            cf_val = res['cf_score'] * hybrid.cf_weight
            hybrid_val = res['hybrid_score']
            
            cbf_contributions.append(cbf_val)
            cf_contributions.append(cf_val)
            total_scores.append(hybrid_val)

        # ==========================================
        # 4. PLOTTING VERTICAL STACKED BAR CHART
        # ==========================================
        self.stdout.write("Membuat visualisasi grafik...")
        
        # Mengatur ukuran kanvas mirip dengan referensi gambar (memanjang mendatar)
        plt.figure(figsize=(10, 5))
        
        # Array posisi X
        x_pos = np.arange(len(movie_titles))
        width = 0.55 # Lebar batang
        
        # Plot batang CBF (Posisi Bawah - Warna Biru Cerah)
        plt.bar(
            x_pos, 
            cbf_contributions, 
            width, 
            color='#3498DB', 
            edgecolor='white',
            label='Komponen Skor CBF (40%)'
        )
        
        # Plot batang CF (Posisi Atas - Warna Oranye, bertumpuk di atas CBF)
        plt.bar(
            x_pos, 
            cf_contributions, 
            width, 
            bottom=cbf_contributions, 
            color='#F39C12', 
            edgecolor='white',
            label='Komponen Skor CF (60%)'
        )
        
        # Menambahkan teks skor total di puncak tiap batang (4 angka desimal)
        for i, total in enumerate(total_scores):
            plt.text(
                x_pos[i], 
                total + 0.015, # Jarak teks di atas batang
                f"{total:.4f}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                fontweight='bold', 
                color='black'
            )
            
            # Menambahkan teks bobot CBF di tengah-tengah batang warna Biru
            plt.text(
                x_pos[i], 
                cbf_contributions[i] / 2, 
                f"{cbf_contributions[i]:.4f}", 
                ha='center', 
                va='center', 
                color='white', 
                fontsize=8, 
                fontweight='bold'
            )
            
            # Menambahkan teks bobot CF di tengah-tengah batang warna Oranye
            y_center_cf = cbf_contributions[i] + (cf_contributions[i] / 2)
            plt.text(
                x_pos[i], 
                y_center_cf, 
                f"{cf_contributions[i]:.4f}", 
                ha='center', 
                va='center', 
                color='white', 
                fontsize=8, 
                fontweight='bold'
            )

        # Labeling dan Kosmetik
        plt.ylabel('Skor Relevansi Akhir (Hybrid Score)', fontsize=11)
        plt.title('Pemetaan Kontribusi Skor pada Sistem Hybrid (40% CBF + 60% CF)\nTop-8 Hasil Rekomendasi Teratas', fontsize=12, pad=10)
        
        # Nama film di sumbu X diputar 35 derajat
        plt.xticks(x_pos, movie_titles, rotation=35, ha='right', fontsize=9)
        
        # Batas minimal dan maksimal sumbu Y
        plt.ylim(0, 1.05) 
        
        # Grid garis horizontal putus-putus
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Menempatkan legenda di kanan atas sesuai gambar referensi
        plt.legend(loc='upper right', frameon=True, fontsize=9)
        
        plt.tight_layout()
        
        # Save Gambar
        out_path = os.path.join(os.getcwd(), 'hybrid_vertical_chart.png')
        plt.savefig(out_path, dpi=300)
        self.stdout.write(self.style.SUCCESS(f"Sukses! Grafik tersimpan di {out_path}"))
