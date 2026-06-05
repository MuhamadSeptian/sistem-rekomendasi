import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from django.core.management.base import BaseCommand
from movies.recommender import CollaborativeRecommender
from movies.models import UserRating

class Command(BaseCommand):
    help = 'Menghasilkan visualisasi Heatmap untuk Matrix Bias CF dan Dekomposisi SVD'

    def handle(self, *args, **kwargs):
        self.stdout.write("Melatih Model CF (SVD)...")
        
        # Inisialisasi CF 
        cf = CollaborativeRecommender(n_factors=10) # 10 factors cukup u/ visualisasi
        cf.train()
        
        if not cf._trained:
            self.stdout.write(self.style.ERROR("Gagal melatih SVD."))
            return
            
        self.stdout.write("Menyiapkan Visualisasi...")
        
        # Kita ambil Top 15 Film & Top 15 Pengguna (yang teraktif)
        # Untuk didemonstrasikan di matriks prediksi SVD
        
        from django.db.models import Count
        top_user_ids = list(
            UserRating.objects.values_list('user_id', flat=True)
            .annotate(cnt=Count('id')).order_by('-cnt')[:15]
        )
        top_movie_ids = list(
            UserRating.objects.values_list('movie__tmdb_id', flat=True)
            .annotate(cnt=Count('id')).order_by('-cnt')[:15]
        )
        
        # Konversi ID ke Index
        user_indices = [cf._user_to_idx[uid] for uid in top_user_ids if uid in cf._user_to_idx]
        movie_indices = [cf._item_to_idx[mid] for mid in top_movie_ids if mid in cf._item_to_idx]
        
        # Bikin Matriks Asli (Sebelum) dan Estimasi (Sesudah) R_hat
        R_asli = np.zeros((len(user_indices), len(movie_indices)))
        R_hat = np.zeros((len(user_indices), len(movie_indices)))
        
        for i, uid in enumerate(top_user_ids):
            for j, mid in enumerate(top_movie_ids):
                # Ambil nilai tebakan SVD (Sesudah)
                R_hat[i, j] = cf._predict_rating(user_indices[i], movie_indices[j])
                
                # Ambil nilai asli dari database (Sebelum)
                rating_obj = UserRating.objects.filter(user_id=uid, movie__tmdb_id=mid).first()
                if rating_obj:
                    R_asli[i, j] = float(rating_obj.rating)
                
        # --- Plotting Heatmap Bersandingan (Before - After) ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        y_labels = [f"User {i+1}" for i in range(len(user_indices))]
        x_labels = [f"Film {i+1}" for i in range(len(movie_indices))]
        
        # 1. Heatmap SEBELUM SVD (Sparse)
        annot_before = np.where(R_asli > 0, R_asli.astype(str), "")
        sns.heatmap(
            R_asli, 
            ax=axes[0],
            annot=annot_before, 
            fmt="", 
            cmap="mako", 
            cbar=False,
            xticklabels=x_labels, 
            yticklabels=y_labels,
            vmin=0.0, vmax=5.0,
            linewidths=0.5
        )
        axes[0].set_title('Matriks Interaksi SEBELUM SVD\n(Kondisi Sparse: Data Rating Asli Pengguna)', fontsize=14, pad=15)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # 2. Heatmap SESUDAH SVD (Dense)
        sns.heatmap(
            R_hat, 
            ax=axes[1],
            annot=True, 
            fmt=".1f", 
            cmap="coolwarm", 
            cbar=True,
            cbar_kws={'label': 'Estimasi Rating SVD (1 - 5)'},
            xticklabels=x_labels, 
            yticklabels=y_labels,
            vmin=1.0, vmax=5.0,
            linewidths=0.5
        )
        axes[1].set_title('Matriks Interaksi SESUDAH SVD\n(Kondisi Dense: Rekonstruksi Tebakan Rating SVD)', fontsize=14, pad=15)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save Gambar
        out_svd_matrix = os.path.join(os.getcwd(), 'cf_svd_heatmap_before_after.png')
        plt.savefig(out_svd_matrix, dpi=300, bbox_inches='tight')
        self.stdout.write(self.style.SUCCESS(f"Saved SVD Heatmap Before-After to {out_svd_matrix}"))
        
        
        # --- Plotting Item & User Bias Scatter ---
        plt.figure(figsize=(8, 6))
        # Ambil sampel acak 100 bias user & item
        samples_u = np.random.choice(cf._user_bias, size=min(100, len(cf._user_bias)), replace=False)
        samples_i = np.random.choice(cf._item_bias, size=min(100, len(cf._item_bias)), replace=False)
        
        plt.scatter(range(len(samples_u)), sorted(samples_u), color='blue', alpha=0.6, label='User Bias (Suka marah/murah hati)')
        plt.scatter(range(len(samples_i)), sorted(samples_i), color='red', alpha=0.6, label='Item Bias (Kualitas inheren film)')
        
        plt.axhline(0, color='gray', linestyle='--')
        plt.title('Distribusi User Bias (Bu) dan Item Bias (Bi)', fontsize=14)
        plt.ylabel('Nilai Bias dari Global Mean', fontsize=12)
        plt.xlabel('Sampel Data (Terurut)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        out_bias = os.path.join(os.getcwd(), 'cf_bias_scatter.png')
        plt.savefig(out_bias, dpi=300, bbox_inches='tight')
        
        self.stdout.write(self.style.SUCCESS(f"Saved Bias Scatter to {out_bias}"))
