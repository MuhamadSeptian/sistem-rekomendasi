from django.core.management.base import BaseCommand
from movies.models import Movie
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Command(BaseCommand):
    help = 'Menampilkan tabel rekap TF-IDF tertinggi di terminal'

    def handle(self, *args, **kwargs):
        self.stdout.write("Menyiapkan data teks seluruh film...")
        movies = list(Movie.objects.prefetch_related("genres").all())
        corpus = []
        for m in movies:
            # Pemberian bobot tambahan fiktif ke judul dan genre seperti di sistem rekomendasi kita.
            genres_str = " ".join([g.name for g in m.genres.all()])
            text = f"{m.title} {m.title} {genres_str} {genres_str} {genres_str} {m.overview}"
            corpus.append(text)

        if not corpus:
            self.stdout.write(self.style.ERROR("Belum ada data film dalam database."))
            return

        self.stdout.write("Memproses Text Preprocessing dan Matriks TF-IDF...")
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(corpus)
        
        scores = np.array(X.sum(axis=0)).flatten()
        top_indices = np.argsort(scores)[::-1][:20]
        features = vectorizer.get_feature_names_out()

        self.stdout.write("\n" + "=" * 45)
        self.stdout.write(f"{'No':<4} | {'Kata (Term)':<20} | {'Akumulasi TF-IDF':<15}")
        self.stdout.write("-" * 45)
        for i, idx in enumerate(top_indices, 1):
            self.stdout.write(f"{i:<4} | {features[idx]:<20} | {scores[idx]:.4f}")
        self.stdout.write("=" * 45)
        
        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Tabel Ranking Kata TF-IDF berhasil ditampilkan!"))
