import os
from django.core.management.base import BaseCommand
from django.db.models import Count
from movies.models import UserRating
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Membuat grafik batang (Bar Chart) untuk distribusi nilai rating pengguna'

    def handle(self, *args, **kwargs):
        self.stdout.write('Mengambil data distribusi rating dari database...')
        
        # Ambil data agregasi jumlah (count) berdasarkan nilai rating
        distribution = UserRating.objects.values('rating').annotate(count=Count('rating')).order_by('rating')
        
        ratings = []
        counts = []
        
        for item in distribution:
            ratings.append(str(item['rating']))  # Jadikan string agar rapi di sumbu X kategori
            counts.append(item['count'])
        
        # Setup plot
        plt.figure(figsize=(8, 5))
        plt.bar(ratings, counts, color='#5499C7', edgecolor='lightgrey', width=0.8)
        
        # Menambahkan label numerik di atas setiap batang grafik
        for i in range(len(ratings)):
            plt.text(i, counts[i] + 300, str(counts[i]), ha='center', va='bottom', fontsize=9)
            
        plt.xlabel('Nilai Rating')
        plt.ylabel('Frekuensi (Jumlah)')
        plt.title('Distribusi Nilai Rating Pengguna')
        
        # Atur batas Y agar ada ruang untuk text label
        if counts:
            plt.ylim(0, max(counts) * 1.1)

        plt.tight_layout()

        # Save plot
        file_name = "distribusi_rating.png"
        plt.savefig(file_name, dpi=300)

        self.stdout.write(self.style.SUCCESS(f'Selesai! Gambar grafik berhasil disimpan sebagai {file_name}'))
