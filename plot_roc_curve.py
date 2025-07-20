import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Verilen modelin ROC eğrisini çizerek sınıflandırma başarısını görselleştirir.

    Args:
        model: ROC hesaplamasına uygun sınıflandırma modeli.
        X_test (pd.DataFrame): Test veri kümesi.
        y_test (pd.Series): Gerçek etiketler.
    """
    display = RocCurveDisplay.from_estimator(
        estimator=model,
        X=X_test,
        y=y_test,
        color="darkorange"
    )
    display.plot()
    plt.title("ROC Eğrisi", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    plot_roc_curve(model, X_test, y_test)


