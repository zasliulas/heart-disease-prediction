import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gender_vs_disease(df: pd.DataFrame) -> None:
    """
    Cinsiyete göre kalp hastalığı oranlarını gösteren bar grafiği üretir.
    Ortalama hastalık oranlarına göre karşılaştırmalı görsel sunar.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))

    gender_disease = df.groupby("Sex")["HeartDisease"].mean().sort_values()
    sns.barplot(x=gender_disease.index, y=gender_disease.values, palette="pastel")

    plt.title("Cinsiyet vs Kalp Hastalığı Oranı", fontsize=14)
    plt.xlabel("Cinsiyet")
    plt.ylabel("Kalp Hastalığı Oranı")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")
    plot_gender_vs_disease(df)

