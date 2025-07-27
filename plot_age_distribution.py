import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_distribution(df: pd.DataFrame):
    """
    Kalp hastalığı veri setindeki yaş değişkeninin istatistiksel dağılımını görselleştirir.
    Hem histogram hem KDE eğrisi ile yaşın genel profilini sunar.

    Args:
        df (pd.DataFrame): Yaş verisini içeren veri çerçevesi

    Returns:
        matplotlib.figure.Figure: Oluşturulan yaş dağılımı grafiği
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Age", bins=15, kde=True, color="salmon", ax=ax)

    ax.set_title("Yaş Dağılımı", fontsize=14)
    ax.set_xlabel("Yaş")
    ax.set_ylabel("Kişi Sayısı")
    fig.tight_layout()

    return fig


