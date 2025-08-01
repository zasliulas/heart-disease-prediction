import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gender_vs_disease(df: pd.DataFrame):
    """
    Cinsiyete göre kalp hastalığı oranlarını gösteren bar grafiği üretir.
    Ortalama hastalık oranlarına göre karşılaştırmalı görsel sunar.

    Args:
        df (pd.DataFrame): Veri çerçevesi

    Returns:
        matplotlib.figure.Figure: Oluşturulan görsel nesne
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    gender_disease = df.groupby("Sex")["HeartDisease"].mean().sort_values()
    sns.barplot(x=gender_disease.index, y=gender_disease.values,
                color="mediumseagreen", ax=ax)

    ax.set_title("Cinsiyet vs Kalp Hastalığı Oranı", fontsize=14)
    ax.set_xlabel("Cinsiyet")
    ax.set_ylabel("Kalp Hastalığı Oranı")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "heart.csv")
    df = pd.read_csv(csv_path)
    fig = plot_gender_vs_disease(df)
    plt.show()




