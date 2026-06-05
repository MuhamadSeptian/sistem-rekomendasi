import os
from django.core.management.base import BaseCommand
from movies.models import Movie
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Membuat grafik diagram lingkaran (Pie Chart) untuk proporsi data film yang valid vs dibuang'

    def handle(self, *args, **kwargs):
        csv_path = 'ml-latest-small/movies.csv'
        
        self.stdout.write('Menghitung jumlah data CSV film dan Database...')
        
        try:
            # Hitung baris di CSV (dikurangi header)
            total_csv_movies = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f'File {csv_path} tidak ditemukan. Pastikan path benar.'))
            return

        # Hitung valid dan dibuang
        valid_movies = Movie.objects.count()
        dropped_movies = total_csv_movies - valid_movies

        # Setup plot
        labels = [f"Film Valid\n({valid_movies:,})", f"Film Dibuang\n({dropped_movies:,})"]
        sizes = [valid_movies, dropped_movies]
        colors = ["#3498db", "#e67e22"] # Menggunakan warna biru dan oranye agar beda dengan rating
        explode = (0, 0.2)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.2f%%", startangle=140)
        ax.axis("equal")
        plt.title("Proporsi Data Film Setelah Validasi dan Pemetaan tmdbId")
        plt.tight_layout()

        # Save plot
        file_name = "proporsi_film_validasi.png"
        plt.savefig(file_name, dpi=300)

        self.stdout.write(self.style.SUCCESS(f'Total CSV: {total_csv_movies}, Valid: {valid_movies}, Dibuang: {dropped_movies}'))
        self.stdout.write(self.style.SUCCESS(f'Selesai! Gambar grafik berhasil disimpan sebagai {file_name}'))
