import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    """
    Gerçek ve tahmin edilen değerler arasındaki karşılaştırmayı gösteren confusion matrix grafiği üretir.

    Args:
        y_true (pd.Series): Gerçek sınıf etiketleri.
        y_pred (pd.Series): Modelin tahmin ettiği sınıf etiketleri.

    Returns:
        matplotlib.figure.Figure: Oluşturulan confusion matrix görseli
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap="Blues", colorbar=False, ax=ax
    )

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    # Örnek kullanım:
    # Gerçek ve tahmin edilen değerlerin örnek serileri
    y_true = pd.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

    fig = plot_confusion_matrix(y_true, y_pred)
    plt.show()

