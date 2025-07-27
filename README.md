# Kalp Hastalığı Tahmin Projesi

Bu proje, kalp hastalığı riskini tahmin etmeye yönelik bir makine öğrenmesi uygulamasıdır. Veri analizi, görselleştirme ve sınıflandırma modelleri kullanılarak gerçekleştirilmiştir.

## 🔍 Projenin Amacı
Kalp hastalığı ile ilişkili faktörleri analiz ederek, bireylerin kalp hastalığı riskini tahmin eden bir model geliştirmek.

## 🧠 Kullanılan Yöntemler
- Veri temizleme ve ön işleme (pandas)
- Görselleştirme (seaborn, matplotlib)
- Makine öğrenmesi (scikit-learn)
  - Random Forest sınıflandırıcısı
  - ROC eğrisi, Confusion Matrix gibi değerlendirme metrikleri

## 📊 Uygulanan Analiz ve Grafikler
Bu branch’te aşağıdaki özel görselleştirme ve analiz fonksiyonları geliştirilmiştir:

- `plot_chestpain_vs_disease`: Göğüs ağrısı türüne göre hastalık oranı
- `plot_age_distribution`: Yaş dağılımı (histogram + KDE)
- `plot_confusion_matrix`: Gerçek vs tahmin sınıflandırmaları
- `plot_correlation_heatmap`: Korelasyon ısı haritası
- `plot_feature_importance`: Modelin özellik önem düzeyleri
- `plot_gender_vs_disease`: Cinsiyete göre hastalık oranı
- `plot_roc_curve`: ROC eğrisi ile model başarısı

## ⚙️ Nasıl Çalıştırılır?

```bash
# Sanal ortamı etkinleştirin
source .venv/Scripts/activate  # Windows için
# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Örnek: ROC eğrisi grafiğini çalıştırmak için
python plot_roc_curve.py
z.
