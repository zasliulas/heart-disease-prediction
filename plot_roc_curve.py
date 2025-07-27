import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Verilen modelin ROC eğrisini çizerek sınıflandırma başarısını görselleştirir.

    Returns:
        matplotlib.figure.Figure: ROC eğrisi figürü
    """
    display = RocCurveDisplay.from_estimator(
        estimator=model,
        X=X_test,
        y=y_test,
        color="darkorange"
    )

    fig, ax = plt.subplots()
    display.plot(ax=ax)
    ax.set_title("ROC Eğrisi", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()

    return fig



