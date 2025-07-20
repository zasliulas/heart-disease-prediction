import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Verilen modelin ROC eğrisini çizerek sınıflandırma başarısını görselleştirir.

    Args:
        model: Tahmin yapan sınıflandırma modeli (predict_proba metodu olmalı).
        X_test (pd.DataFrame): Test veri kümesi.
        y_test (pd.Series): Test etiketleri.
    """
    plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        color="darkorange"
    )

    plt.title("ROC Eğrisi", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/heart.csv")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Kategorik değişkenleri sayısal hale getir
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    plot_roc_curve(model, X_test, y_test)
