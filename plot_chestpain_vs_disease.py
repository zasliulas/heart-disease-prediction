import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_chestpain_vs_disease(df: pd.DataFrame):
    """
    Göğüs ağrısı türlerinin kalp hastalığı oranlarıyla ilişkisini gösterir.
    Ortalama hastalık oranlarına göre bar grafiği üretir.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    chest_disease = df.groupby("ChestPainType")["HeartDisease"].mean().sort_values()
    sns.barplot(x=chest_disease.index, y=chest_disease.values, color="steelblue", ax=ax)

    ax.set_title("Göğüs Ağrısı Türü vs Kalp Hastalığı Oranı", fontsize=14)
    ax.set_xlabel("Göğüs Ağrısı Türü")
    ax.set_ylabel("Kalp Hastalığı Oranı")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")
    fig = plot_chestpain_vs_disease(df)
    plt.show()
