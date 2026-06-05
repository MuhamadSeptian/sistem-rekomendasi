import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Generate Stacked Bar Chart for Hybrid Recommendations"

    def handle(self, *args, **kwargs):
        self.stdout.write("Generating Hybrid Stacked Bar Chart...")
        
        # ==========================================
        # 1. DATA SIMULASI (Top 5 Rekomendasi)
        # ==========================================
        # Menggunakan judul film populer untuk representasi visual laporan
        movies = [
            "Inception (2010)", 
            "Interstellar (2014)", 
            "The Dark Knight (2008)", 
            "The Matrix (1999)", 
            "Avatar (2009)"
        ]
        
        # Asumsi Normalisasi skor (Skala 0.0 - 1.0) sebelum dibobotkan
        cbf_norm_scores = np.array([0.95, 0.88, 0.75, 0.82, 0.60])
        cf_norm_scores  = np.array([0.92, 0.89, 0.98, 0.85, 0.95])
        
        # Bobot Model (Sesuai dengan sistem Anda: CBF=0.4, CF=0.6)
        alpha_cbf = 0.4
        beta_cf = 0.6
        
        # Kalkulasi kontribusi akhir untuk panjang bar chart
        cbf_contribution = cbf_norm_scores * alpha_cbf
        cf_contribution = cf_norm_scores * beta_cf
        
        # Hybrid total (digunakan untuk memunculkan teks total di ujung bar)
        hybrid_total = cbf_contribution + cf_contribution
        
        # ==========================================
        # 2. PROSES PLOTTING GRAFIK
        # ==========================================
        plt.figure(figsize=(10, 6))
        # Menggunakan style minimalis agar terkesan akademis
        sns.set_style("white", {'axes.grid': False}) 
        
        # Mengatur posisi Y untuk grafik horizontal
        y_pos = np.arange(len(movies))
        
        # Menggambar Bar Bagian 1 (CBF) di sebelah kiri
        plt.barh(
            y_pos, 
            cbf_contribution, 
            height=0.6, 
            color='#2ECC71', # Hijau
            label='Kontribusi CBF (40%)', 
            edgecolor='white'
        )
        
        # Menggambar Bar Bagian 2 (CF) menumpuk di sebelahnya (left=cbf_contribution)
        plt.barh(
            y_pos, 
            cf_contribution, 
            height=0.6, 
            left=cbf_contribution, 
            color='#3498DB', # Biru
            label='Kontribusi CF (60%)', 
            edgecolor='white'
        )
        
        # ==========================================
        # 3. KOSMETIK & TEKS ANGKA
        # ==========================================
        plt.yticks(y_pos, movies, fontsize=11, fontweight='500')
        plt.xlabel('Skor Total Hybrid (Max 1.0)', fontsize=12)
        plt.title('Proporsi Kontribusi Skor CBF dan CF\npada Pemeringkatan Top-5 Rekomendasi Model Hybrid', fontsize=14, pad=15, fontweight='bold')
        
        # Memasukkan teks angka ke dalam chart agar dosen mudah membacanya
        for i, total in enumerate(hybrid_total):
            # Teks Total di luar kanan bar
            plt.text(total + 0.015, i, f"{total:.3f}", va='center', fontsize=11, fontweight='bold', color='#2C3E50')
            
            # Teks Skor CBF di tengah area hijau
            plt.text(cbf_contribution[i]/2, i, f"{cbf_contribution[i]:.2f}", va='center', ha='center', color='white', fontweight='bold')
            
            # Teks Skor CF di tengah area biru
            plt.text(cbf_contribution[i] + cf_contribution[i]/2, i, f"{cf_contribution[i]:.2f}", va='center', ha='center', color='white', fontweight='bold')

        # Batas X diperlebar sedikit agar teks total tidak terpotong
        plt.xlim(0, 1.1)
        
        # Membalik urutan Y agar ranking 1 (index 0) berada di bagian paling atas grafik
        plt.gca().invert_yaxis()
        
        # Menghilangkan garis batas grafik (spines) atas dan kanan agar lebih bersih
        sns.despine(top=True, right=True)
        
        # Menempatkan legenda di pojok kanan bawah
        plt.legend(loc='lower right', frameon=True, fontsize=10)
        
        plt.tight_layout()
        
        # ==========================================
        # 4. RENDER & SAVE
        # ==========================================
        out_path = os.path.join(os.getcwd(), 'hybrid_stacked_chart_hd.png')
        plt.savefig(out_path, dpi=300)
        self.stdout.write(self.style.SUCCESS(f"Berhasil! Gambar telah disimpan sebagai '{out_path}'"))
