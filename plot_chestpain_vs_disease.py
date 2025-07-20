import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_chestpain_vs_disease(df: pd.DataFrame) -> None:
    """
    Veri çerçevesi içindeki göğüs ağrısı türlerinin kalp hastalığı oranlarıyla ilişkisini gösterir.
    Ortalama hastalık oranlarına göre bar grafiği sunar.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    chest_disease = df.groupby("ChestPainType")["HeartDisease"].mean().sort_values()
    sns.barplot(x=chest_disease.index, y=chest_disease.values, palette="viridis")

    plt.title("Göğüs Ağrısı Türü vs Kalp Hastalığı Oranı", fontsize=14)
    plt.xlabel("Göğüs Ağrısı Türü")
    plt.ylabel("Kalp Hastalığı Oranı")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")  # Dosya yolunu ihtiyaca göre güncelle
    plot_chestpain_vs_disease(df)
