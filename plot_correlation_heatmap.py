import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Sayısal sütunlar arasındaki korelasyonları gösteren ısı haritası üretir.

    Args:
        df (pd.DataFrame): Korelasyon hesaplanacak sayısal veri çerçevesi.
    """
    sns.set_style("white")
    plt.figure(figsize=(10, 7))

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

    plt.title("Korelasyon Matrisi", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")
    plot_correlation_heatmap(df)
