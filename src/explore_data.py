import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_heart_data():
    """
    Heart Disease veri setini keşfeder ve görselleştirir
    """
    # Veri setinin varlığını kontrol et
    if not os.path.exists("heart.csv"):
        print("Veri seti bulunamadı. Lütfen önce 'download_data.py' dosyasını çalıştırın.")
        return False
    
    # Veri setini oku
    df = pd.read_csv("heart.csv")
    
    # Veri seti hakkında genel bilgi
    print("Veri Seti Bilgisi:")
    print(f"Satır sayısı: {df.shape[0]}")
    print(f"Sütun sayısı: {df.shape[1]}")
    print("\nSütun tipleri:")
    print(df.dtypes)
    
    print("\nEksik değerler:")
    print(df.isnull().sum())
    
    print("\nBetimsel istatistikler:")
    print(df.describe())
    
    # Hedef değişken dağılımı
    plt.figure(figsize=(8, 6))
    sns.countplot(x='HeartDisease', data=df)
    plt.title('Kalp Hastalığı Dağılımı')
    plt.xlabel('Kalp Hastalığı (0: Yok, 1: Var)')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('heart_disease_distribution.png')
    print("Kalp hastalığı dağılımı grafiği 'heart_disease_distribution.png' olarak kaydedildi.")
    
    # Yaş dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='HeartDisease', multiple='stack', bins=20)
    plt.title('Yaşa Göre Kalp Hastalığı Dağılımı')
    plt.xlabel('Yaş')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('age_distribution.png')
    print("Yaş dağılımı grafiği 'age_distribution.png' olarak kaydedildi.")
    
    # Cinsiyet ve kalp hastalığı ilişkisi
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sex', hue='HeartDisease', data=df)
    plt.title('Cinsiyete Göre Kalp Hastalığı')
    plt.xlabel('Cinsiyet (F: Kadın, M: Erkek)')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('gender_heart_disease.png')
    print("Cinsiyet ve kalp hastalığı grafiği 'gender_heart_disease.png' olarak kaydedildi.")
    
    # Korelasyon matrisi
    plt.figure(figsize=(12, 10))
    
    # Kategorik değişkenleri sayısal hale getir
    df_corr = df.copy()
    df_corr['Sex'] = df_corr['Sex'].replace({'M': 1, 'F': 0})
    df_corr['ChestPainType'] = df_corr['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df_corr['RestingECG'] = df_corr['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
    df_corr['ExerciseAngina'] = df_corr['ExerciseAngina'].replace({'N': 0, 'Y': 1})
    df_corr['ST_Slope'] = df_corr['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})
    
    corr = df_corr.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.savefig('correlation_matrix.png')
    print("Korelasyon matrisi 'correlation_matrix.png' olarak kaydedildi.")
    
    print("\nVeri keşfi tamamlandı!")
    return True

if __name__ == "__main__":
    explore_heart_data()
