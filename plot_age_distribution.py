import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_age_distribution(df: pd.DataFrame) -> None:
    """
    Veri çerçevesi içindeki yaş değişkeninin dağılımını gösterir.
    Histogram ve KDE eğrisi bir arada sunulur.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", bins=15, kde=True, color="salmon")

    plt.title("Yaş Dağılımı", fontsize=14)
    plt.xlabel("Yaş")
    plt.ylabel("Kişi Sayısı")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")  # Dosya yolunu ihtiyaca göre güncelle
    plot_age_distribution(df)
