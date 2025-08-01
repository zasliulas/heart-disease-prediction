import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Verilen modelin ROC eğrisini çizerek sınıflandırma başarısını görselleştirir.

    Args:
        model: Sınıflandırma modeli (fit edilmiş).
        X_test (pd.DataFrame): Test veri özellikleri.
        y_test (pd.Series): Test veri gerçek etiketleri.

    Returns:
        matplotlib.figure.Figure: ROC eğrisi figürü
    """
    display = RocCurveDisplay.from_estimator(
        estimator=model,
        X=X_test,
        y=y_test,
    )

    fig = display.figure_
    ax = display.ax_

    ax.set_title("ROC Eğrisi", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Veri seti yükle
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=42, test_size=0.3
    )
    # Model oluştur ve eğit
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ROC eğrisini çiz ve göster
    fig = plot_roc_curve(model, X_test, y_test)
    plt.show()



