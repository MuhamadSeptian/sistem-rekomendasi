import os
from django.core.management.base import BaseCommand
from movies.models import Movie
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

class Command(BaseCommand):
    help = 'Membuat visualisasi kata kunci TF-IDF terkuat setelah Text Preprocessing'

    def handle(self, *args, **kwargs):
        self.stdout.write("Menyatukan corpus data teks film dari Database...")
        movies = list(Movie.objects.prefetch_related('genres').all())
        
        if not movies:
            self.stdout.write("Data film kosong.")
            return
            
        corpus = []
        for m in movies:
            genres_str = " ".join([g.name for g in m.genres.all()])
            # Format bobot yang persis sama seperti di recommender.py
            text = f"{m.title} {genres_str} {m.overview}"
            corpus.append(text)
            
        self.stdout.write("Mengeksekusi Text Preprocessing (TfidfVectorizer)...")
        # vectorizer ini secara otomatis melakukan Lowercase, Punctuation Removal & Stopword 
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(corpus)
        
        self.stdout.write("Menghitung skor kata tertinggi...")
        # Menjumlahkan bobot tf-idf setiap kata di semua film
        scores = np.array(X.sum(axis=0)).flatten()
        
        # Ambil 15 indeks dengan skor tertinggi
        top_indices = np.argsort(scores)[::-1][:15]
        features = vectorizer.get_feature_names_out()
        
        top_words = [features[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        # Membuat Grafik Bar Horisontal (dibalik urutannya agar yang terbesar ada di paling atas)
        plt.figure(figsize=(9, 6))
        plt.barh(top_words[::-1], top_scores[::-1], color='#8E44AD', edgecolor='lightgrey')
        plt.xlabel('Akumulasi Nilai Bobot TF-IDF')
        plt.ylabel('Kata Dasar (Term)')
        plt.title('Top 15 Kata Paling Berpengaruh Sesudah Text Preprocessing')
        plt.tight_layout()
        
        file_name = 'hasil_text_preprocessing.png'
        plt.savefig(file_name, dpi=300)
        
        self.stdout.write(self.style.SUCCESS(f"Selesai! Gambar berhasi dibuat: {file_name}"))
