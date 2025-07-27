import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Sayısal sütunlar arasındaki korelasyonları gösteren ısı haritası üretir.

    Args:
        df (pd.DataFrame): Korelasyon hesaplanacak sayısal veri çerçevesi.

    Returns:
        matplotlib.figure.Figure: Oluşturulan korelasyon ısı haritası
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 7))

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)

    ax.set_title("Korelasyon Matrisi", fontsize=14)
    fig.tight_layout()

    return fig

