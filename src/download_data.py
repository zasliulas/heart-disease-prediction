import pandas as pd
import requests
import os

def download_heart_data():
    """
    Heart Disease veri setini indirir ve kaydeder
    """
    # Veri seti URL'si
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/heart-nvnpJME60HuRDzmtQgubwZSiTXrizB.csv"
    
    print("Veri seti indiriliyor...")
    
    try:
        # Veriyi indir
        response = requests.get(url)
        response.raise_for_status()  # HTTP hatalarını kontrol et
        
        # Ana dizine kaydet
        with open("heart.csv", "wb") as f:
            f.write(response.content)
        
        # Veri setini kontrol et
        df = pd.read_csv("heart.csv")
        print(f"Veri seti başarıyla indirildi. Boyut: {df.shape}")
        print(f"Sütunlar: {df.columns.tolist()}")
        print(f"İlk 5 satır:\n{df.head()}")
        
        return True
    
    except Exception as e:
        print(f"Veri seti indirilirken hata oluştu: {e}")
        return False

if __name__ == "__main__":
    download_heart_data()
