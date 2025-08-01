# Heart Disease Prediction Projesi

## Proje Hakkında
Kalp hastalığı tahmini için veri analizi ve makine öğrenmesi modelleri geliştirmeyi amaçlayan bir proje.

---

## Dosya ve Klasör Yapısı

- `data/`  
  Veri setleri ve ham dosyaların bulunduğu klasör.

- `notebooks/`  
  Jupyter Notebook dosyalarının bulunduğu klasör.

- `src/`  
  Python kaynak kodlarının bulunduğu klasör.

- `reports/`  
  Proje çıktılarına, grafiklere ve raporlara ait klasör.

- `README.md`  
  Proje açıklaması ve rehberi.

- `requirements.txt`  
  Projede kullanılan Python paketleri listesi.

---

## Branch Çalışma Kuralları

- Main branch’e doğrudan push yapmak yasaktır.  
- Tüm değişiklikler, kendi branch’inde yapılıp Pull Request (PR) ile main’e gönderilmelidir.  
- Branch isimlendirme önerileri:  
  - `feature/özellik-adi`  
  - `bugfix/hata-adi`  
- PR açarken ne yaptığınızı açıklayın ve en az bir ekip üyesinden onay alın.  
- Onaylanan PR’lar main’e merge edilir.  
- Main branch her zaman stabil ve deploy edilebilir olmalıdır.

---

## Nasıl Katkı Yapılır?

1. Projeye klonlayın:  
   ```bash
   git clone https://github.com/kullaniciadi/projeadi.git
Size atanmış branch’e geçin (örneğin ayse-branch):

bash
Kopyala
Düzenle
git checkout ayse-branch
Kodlarınızı yazıp commit edin:

bash
Kopyala
Düzenle
git add .
git commit -m "Yaptığım değişiklik açıklaması"
Branch’inize push yapın:

bash
Kopyala
Düzenle
git push origin ayse-branch
GitHub üzerinden New pull request açın.

Kodlarınız incelenip main’e merge edilir.

