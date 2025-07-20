import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Gerçek ve tahmin edilen değerler arasındaki karşılaştırmayı gösteren confusion matrix grafiği üretir.

    Args:
        y_true (pd.Series): Gerçek sınıf etiketleri.
        y_pred (pd.Series): Modelin tahmin ettiği sınıf etiketleri.
    """
    plt.figure(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap="Blues", colorbar=False
    )

    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Örnek veriler (gerçek proje akışında model çıktısı ile beslenmeli)
    # Örnek olarak dosya yüklenip test yapılacaksa:
    df = pd.read_csv("data/heart.csv")
    y_true = df["HeartDisease"]
    y_pred = df["HeartDisease"]  # Burada test amaçlı olarak tahmin = gerçek
    plot_confusion_matrix(y_true, y_pred)
