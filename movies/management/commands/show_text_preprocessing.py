import re
import textwrap
from django.core.management.base import BaseCommand
from movies.models import Movie
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class Command(BaseCommand):
    help = 'Menampilkan tabel demonstrasi Text Preprocessing dengan garis batas'

    def handle(self, *args, **kwargs):
        # 1. Pastikan Toy Story ada di urutan pertama
        toy_story = Movie.objects.filter(title__icontains="Toy Story").first()
        if toy_story:
            other_movies = list(Movie.objects.exclude(id=toy_story.id)[:4])
            movies = [toy_story] + other_movies
        else:
            movies = list(Movie.objects.all()[:5])

        if not movies:
            self.stdout.write(self.style.ERROR("Belum ada data film dalam database."))
            return

        # 2. Siapkan baris data
        data = []
        for i, movie in enumerate(movies):
            genres_list = [g.name for g in movie.genres.all()]
            # Menggabungkan semua teks seperti pada sistem rekomendasi
            raw_text = f"{movie.title} {' '.join(genres_list)} {movie.overview}"
            
            # Tahapan Preprocessing
            case_folded = raw_text.lower()
            no_punctuation = re.sub(r'[^\w\s]', '', case_folded)
            words = no_punctuation.split()
            no_stopwords = [w for w in words if w not in ENGLISH_STOP_WORDS]
            cleaned_text = " ".join(no_stopwords)
            
            data.append((str(i), raw_text, cleaned_text))

        # 3. Buat pembatas tabel (garis vertikal dan horizontal) agar terlihat seperti tabel beneran
        w_idx = 3
        w_raw = 55
        w_clean = 55
        
        def cetak_garis():
            self.stdout.write("+" + "-"*w_idx + "+" + "-"*(w_raw+2) + "+" + "-"*(w_clean+2) + "+")
        
        self.stdout.write("\n")
        cetak_garis()
        self.stdout.write(f"|{'':<{w_idx}}| {'Deskripsi Asli (Raw Text)':<{w_raw}} | {'Hasil Preprocessing (Cleaned Text)':<{w_clean}} |")
        cetak_garis()
        
        for row in data:
            idx, raw, cleaned = row
            
            # Bungkus teks panjang ke beberapa baris
            raw_lines = textwrap.wrap(raw, width=w_raw)
            cleaned_lines = textwrap.wrap(cleaned, width=w_clean)
            
            # Batasi hingga 3 baris per film saja agar tabel tidak terlalu panjang penuh
            raw_lines = raw_lines[:3]
            if len(raw_lines) == 3:
                raw_lines[-1] = raw_lines[-1][:w_raw-3] + "..."
            
            cleaned_lines = cleaned_lines[:3]
            if len(cleaned_lines) == 3:
                cleaned_lines[-1] = cleaned_lines[-1][:w_clean-3] + "..."
            
            max_lines = max(len(raw_lines), len(cleaned_lines))
            if max_lines == 0:
                max_lines = 1
                
            for line_idx in range(max_lines):
                c0 = idx if line_idx == 0 else ""
                c1 = raw_lines[line_idx] if line_idx < len(raw_lines) else ""
                c2 = cleaned_lines[line_idx] if line_idx < len(cleaned_lines) else ""
                
                self.stdout.write(f"|{c0:^{w_idx}}| {c1:<{w_raw}} | {c2:<{w_clean}} |")
            
            cetak_garis()

        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Tabel sampel Text Preprocessing dengan garis berhasil ditampilkan!"))
