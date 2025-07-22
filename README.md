# Kalp Hastalığı Tahmini Projesi

Bu proje, çeşitli sağlık parametrelerini kullanarak kalp hastalığı riskini tahmin eden bir makine öğrenimi uygulamasıdır.

## Proje Yapısı

- `app.py`: Streamlit web uygulaması
- `train_model.py`: Model eğitim kodu
- `scripts/download_data.py`: Veri setini indirme scripti
- `scripts/explore_data.py`: Veri keşfi ve görselleştirme scripti
- `heart.csv`: Kalp hastalığı veri seti
- `model.pkl`: Eğitilmiş model dosyası (eğitim sonrası oluşturulur)
- `scaler.pkl`: Veri ölçeklendirme dosyası (eğitim sonrası oluşturulur)

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
\`\`\`bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn requests
\`\`\`

2. Veri setini indirin:
\`\`\`bash
python scripts/download_data.py
\`\`\`

3. Veri setini keşfedin (isteğe bağlı):
\`\`\`bash
python scripts/explore_data.py
\`\`\`

4. Modeli eğitin:
\`\`\`bash
python train_model.py
\`\`\`

5. Uygulamayı çalıştırın:
\`\`\`bash
streamlit run app.py
\`\`\`

## Veri Seti

Veri seti aşağıdaki özellikleri içermektedir:

- Age: Yaş
- Sex: Cinsiyet (M: Erkek, F: Kadın)
- ChestPainType: Göğüs ağrısı tipi (ATA, NAP, ASY, TA)
- RestingBP: İstirahat kan basıncı (mmHg)
- Cholesterol: Kolesterol (mg/dl)
- FastingBS: Açlık kan şekeri > 120 mg/dl (1: Evet, 0: Hayır)
- RestingECG: Dinlenme EKG sonucu (Normal, ST, LVH)
- MaxHR: Maksimum kalp atış hızı
- ExerciseAngina: Egzersize bağlı anjina (Y: Evet, N: Hayır)
- Oldpeak: ST segmentindeki düşüş
- ST_Slope: ST eğimi (Up, Flat, Down)
- HeartDisease: Kalp hastalığı (1: Var, 0: Yok)

## Model

Bu projede Random Forest sınıflandırıcı kullanılmıştır. Model, GridSearchCV ile hiperparametre optimizasyonu yapılarak eğitilmiştir.

