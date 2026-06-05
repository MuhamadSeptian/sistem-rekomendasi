import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Count, Avg
from movies.recommender import ContentBasedRecommender
from movies.models import Movie, UserRating
from sklearn.metrics.pairwise import cosine_similarity

class Command(BaseCommand):
    help = 'Menghasilkan gambar Heatmap Cosine Similarity untuk 10 film sampel'

    def handle(self, *args, **kwargs):
        self.stdout.write("Mempersiapkan Model CBF dan ekstraksi Matriks TF-IDF...")
        
        # 1. Inisialisasi dan Train CBF untuk membentuk self._tfidf_matrix
        cbf = ContentBasedRecommender()
        cbf.train()
        
        # 2. Mengambil 10 film populer secara acak yang divalidasi sistem 
        # (Syarat: minimum direview 5 kali dengan rata-rata rating >= 3.5)
        popular_movie_ids = list(
            UserRating.objects
            .values_list("movie_id", flat=True)
            .annotate(count=Count("id"), avg=Avg("rating"))
            .filter(count__gte=5, avg__gte=3.5)
            .order_by("-count")[:50]
        )
        
        if len(popular_movie_ids) >= 10:
            selected_ids = random.sample(popular_movie_ids, 10)
        else:
            selected_ids = popular_movie_ids
            
        sample_movies = list(Movie.objects.filter(id__in=selected_ids))
        
        if len(sample_movies) < 5:
            self.stdout.write(self.style.ERROR("Gagal menemukan sampel film yang memadai. Pastikan database lengkap."))
            return

        # 3. Kumpulkan vektor TF-IDF untuk masing-masing film
        indices = []
        labels = []
        for m in sample_movies:
            idx = cbf._tmdb_to_idx.get(m.tmdb_id)
            if idx is not None:
                indices.append(idx)
                labels.append(m.title)
        
        # 4. Hitung Matrix Item-Item Cosine Similarity antar film-film ini
        sample_vectors = cbf._tfidf_matrix[indices]
        sim_matrix = cosine_similarity(sample_vectors)
        
        # 5. Visualisasi menggunakan Seaborn Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sim_matrix, 
            annot=True,          # Tampilkan angka di dalam kotak
            fmt=".2f",           # Format 2 angka desimal
            cmap="YlGnBu",       # Paduan warna Kuning (0) ke Biru Gelap (1)
            xticklabels=labels, 
            yticklabels=labels,
            cbar_kws={'label': 'Cosine Similarity Score (0.0 - 1.0)'}
        )
        
        plt.title('Matriks Kedekatan Konten antar Film (Cosine Similarity Heatmap)', fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        # 6. Save Gambar
        output_path = os.path.join(os.getcwd(), 'cbf_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.stdout.write(self.style.SUCCESS(f"\n[SUCCESS] Gambar Heatmap CBF berhasil dirender ke: {output_path}"))
