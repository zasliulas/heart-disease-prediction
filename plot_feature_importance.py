import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names: list) -> None:
    """
    Modelin özellik önem düzeylerini gösteren yatay bar grafiği üretir.

    Args:
        model: feature_importances_ niteliğine sahip model nesnesi (örn. RandomForestClassifier).
        feature_names (list): Modelde kullanılan özellik isimleri.
    """
    importances = model.feature_importances_
    sorted_indices = importances.argsort()

    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in sorted_indices],
        importances[sorted_indices],
        color="teal"
    )

    plt.title("Özellik Önem Sıralaması", fontsize=14)
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/heart.csv")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Kategorik değişkenleri kodlayalım
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    plot_feature_importance(model, X_encoded.columns.tolist())
