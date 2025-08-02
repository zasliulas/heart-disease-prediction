import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Model eğitimi başlatılıyor...")

# Veri setini kontrol et ve indir
def check_and_download_data():
    if not os.path.exists("heart.csv"):
        print("Veri seti bulunamadı. Lütfen 'heart.csv' dosyasını projenin ana dizinine ekleyin.")
        return False
    return True

if not check_and_download_data():
    print("Veri seti bulunamadığı için işlem durduruluyor.")
    exit()

# Veri setini oku
print("Veri seti okunuyor...")
df = pd.read_csv("heart.csv")

# Veri seti hakkında bilgi
print(f"Veri seti boyutu: {df.shape}")
print("\nİlk 5 satır:")
print(df.head())

# Kategorik verileri sayısal hale getir
print("\nKategorik veriler dönüştürülüyor...")
df['Sex'] = df['Sex'].replace({'M': 1, 'F': 0})
df['ChestPainType'] = df['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
df['RestingECG'] = df['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ExerciseAngina'] = df['ExerciseAngina'].replace({'N': 0, 'Y': 1})
df['ST_Slope'] = df['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})

# Özellik ve hedef değişkenleri ayır
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Veriyi ayır
print("Veri eğitim ve test setlerine ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizasyon
print("Veriler ölçeklendiriliyor...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli eğit
print("Model eğitiliyor...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# En iyi parametreleri yazdır
print(f"\nEn iyi parametreler: {grid_search.best_params_}")

# Test seti üzerinde değerlendirme
print("\nModel değerlendiriliyor...")
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk (Accuracy): {accuracy:.4f}")

# Sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)

# Özellik önemliliği
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nÖzellik Önemliliği:")
print(feature_importance)

# Modeli ve scaler'ı kaydet
print("\nModel ve scaler kaydediliyor...")
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model eğitimi tamamlandı!")
