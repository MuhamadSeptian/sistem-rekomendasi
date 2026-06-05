import os
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from movies.management.commands.evaluate_recommender import Command as EvalCommand

class Command(BaseCommand):
    help = "Run actual evaluation and generate metrics comparison chart dynamically"

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.WARNING(
            "Memulai evaluasi aktual... Proses komputasi 5-Fold CV ini akan memakan waktu."
        ))
        
        # 1. Menginstansiasi dan menjalankan script evaluasi utama secara langsung
        eval_cmd = EvalCommand()
        eval_cmd.stdout = self.stdout # Mengarahkan output terminal agar tetap terlihat
        eval_cmd.stderr = self.stderr
        
        # Menjalankan fungsi handle dengan argumen default (folds=5, k=10, sample_users=100)
        # Fungsi ini sekarang akan me-return dictionary hasil summary
        summary = eval_cmd.handle(folds=5, k=10, sample_users=100)
        
        if not summary:
            self.stdout.write(self.style.ERROR("Evaluasi gagal atau data tidak ditemukan."))
            return

        self.stdout.write(self.style.SUCCESS("Evaluasi selesai! Mengekstrak angka untuk digambar..."))
        
        # 2. Mengekstrak data aktual dari return value script evaluasi
        cbf = summary['cbf']
        cf = summary['cf']
        hybrid = summary['hybrid']
        
        # Membulatkan angka agar cantik di grafik (4 angka desimal, atau 3 angka untuk RMSE)
        rmse_scores = [round(cbf['rmse'], 3), round(cf['rmse'], 3), round(hybrid['rmse'], 3)]
        hr_scores = [round(cbf['hr'], 3), round(cf['hr'], 3), round(hybrid['hr'], 3)]
        map_scores = [round(cbf['map'], 3), round(cf['map'], 3), round(hybrid['map'], 3)]
        
        # 3. Proses Plotting Grafik
        labels = ['Content-Based', 'Collaborative', 'Hybrid (40:60)']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Perbandingan Rata-rata K-Fold Cross Validation (k=5)', fontsize=14, fontweight='bold', y=1.05)
        
        colors = ['#BDC3C7', '#BDC3C7', '#3498DB']
        edge_color = 'black'
        
        def create_bar_chart(ax, data, title, y_max):
            bars = ax.bar(labels, data, color=colors, edgecolor=edge_color, linewidth=1.2)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    height + (y_max * 0.02),
                    f"{height}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10, 
                    fontweight='bold',
                    color='black'
                )
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0, y_max)

        # Mengatur batas tinggi grafik secara dinamis (20% lebih tinggi dari nilai maksimal)
        rmse_max = max(rmse_scores) * 1.2
        hr_max = max(hr_scores) * 1.2
        map_max = max(map_scores) * 1.2

        # 4. Merender Subplots
        create_bar_chart(axes[0], rmse_scores, 'RMSE (Lower is Better)', y_max=rmse_max)
        create_bar_chart(axes[1], hr_scores, 'Hit Rate@10 (Higher is Better)', y_max=hr_max)
        create_bar_chart(axes[2], map_scores, 'MAP (Higher is Better)', y_max=map_max)
        
        plt.tight_layout()
        
        # 5. Menyimpan Output
        out_path = os.path.join(os.getcwd(), 'evaluation_comparison_chart_hd.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        
        self.stdout.write(self.style.SUCCESS(f"Sukses! Gambar Grafik Evaluasi ACTUAL telah tersimpan sebagai '{out_path}'"))
