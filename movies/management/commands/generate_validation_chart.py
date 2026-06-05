import os
from django.core.management.base import BaseCommand
from movies.models import UserRating
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Membuat grafik diagram lingkaran (Pie Chart) untuk proporsi data rating yang valid vs dibuang'

    def handle(self, *args, **kwargs):
        csv_path = 'ml-latest-small/ratings.csv'
        
        self.stdout.write('Menghitung jumlah data CSV dan Database...')
        
        try:
            # Hitung baris di CSV (dikurangi header)
            total_csv_ratings = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f'File {csv_path} tidak ditemukan. Pastikan path benar.'))
            return

        # Hitung valid dan dibuang
        valid_ratings = UserRating.objects.count()
        dropped_ratings = total_csv_ratings - valid_ratings

        # Setup plot
        labels = [f"Rating Valid\n({valid_ratings:,})", f"Rating Dibuang\n({dropped_ratings:,})"]
        sizes = [valid_ratings, dropped_ratings]
        colors = ["#2ecc71", "#e74c3c"]
        explode = (0, 0.2)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.2f%%", startangle=140)
        ax.axis("equal")
        plt.title("Proporsi Data Rating Setelah Valdasi dan Pemetaan tmdbId")
        plt.tight_layout()

        # Save plot
        file_name = "proporsi_rating_validasi.png"
        plt.savefig(file_name, dpi=300)

        self.stdout.write(self.style.SUCCESS(f'Total CSV: {total_csv_ratings}, Valid: {valid_ratings}, Dibuang: {dropped_ratings}'))
        self.stdout.write(self.style.SUCCESS(f'Selesai! Gambar grafik berhasil disimpan sebagai {file_name}'))
