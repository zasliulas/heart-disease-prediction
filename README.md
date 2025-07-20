# Görselleştirme Modülü – `aslı` Branch

Bu modül, kalp hastalığı tahmin projesi kapsamında görselleştirme görevlerini sade, modüler ve yeniden kullanılabilir şekilde organize eder. Amaç; verinin yapısını anlamayı kolaylaştıran ve model çıktılarının yorumlanabilirliğini artıran grafikler üretmektir.

---

## 🎯 Modülün Odak Noktası

- Tüm görseller bağımsız Python dosyalarında yer alır.
- Fonksiyonlar tek parametreyle (genellikle `df`) çağrılır.
- Hem Jupyter hem Streamlit hem de `main.py` entegrasyonlarına uygundur.
- Kod okunabilirliği, estetik ve yorumlanabilirlik ön plandadır.

---

## 📁 Klasör Yapısı – `grafikler/`


---

## 🔍 Fonksiyon Özeti

| Dosya                           | Fonksiyon                          | Açıklama                                     |
|----------------------------------|-------------------------------------|----------------------------------------------|
| `plot_age_distribution.py`       | `plot_age_distribution(df)`        | Yaş histogramı + KDE eğrisi                  |
| `plot_chestpain_vs_disease.py`   | `plot_chestpain_vs_disease(df)`    | Göğüs ağrısı tipi vs hastalık oranları       |
| `plot_confusion_matrix.py`       | `plot_confusion_matrix(y_true, y_pred)` | Model doğruluk matrisi                  |
| `plot_correlation_heatmap.py`    | `plot_correlation_heatmap(df)`     | Korelasyon ısı haritası                      |
| `plot_feature_importance.py`     | `plot_feature_importance(model, df)` | Özellik önem sıralaması                 |
| `plot_gender_vs_disease.py`      | `plot_gender_vs_disease(df)`       | Cinsiyete göre hastalık oranı                |
| `plot_roc_curve.py`              | `plot_roc_curve(model, X_test, y_test)` | ROC eğrisi ve AUC değeri              |

---

## 🧪 Kullanım Örneği

```python
from grafikler.plot_correlation_heatmap import plot_correlation_heatmap
plot_correlation_heatmap(df)
Fonksiyonlar doğrudan çağrılabilir ve farklı platformlara entegre edilebilir.

👩‍💻 Modül Geliştirici: Aslı
Bu modül, aslı branch altında Aslı tarafından geliştirilmiştir. Görevler:

Grafik fonksiyonlarının dosya bazlı modüler yapıda tanımlanması

Kod stilinin okunabilir ve estetik biçimde kurgulanması

Alfabetik dosya sıralaması ile erişim kolaylığı sağlanması

Branch üzerinden main entegrasyon sürecinin yapılandırılması

🔄 Branch Bilgisi
Bu modül yalnızca aslı branch’ta yer almaktadır.

data/heart.csv dosyasına erişim main branch üzerinden sağlanmalıdır.

grafikler/ klasörü PR süreciyle main branch’a entegre edilecektir.
