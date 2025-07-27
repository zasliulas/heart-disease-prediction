import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names: list):
    """
    Modelin özellik önem düzeylerini gösteren yatay bar grafiği üretir.

    Args:
        model: feature_importances_ niteliğine sahip model nesnesi (örn. RandomForestClassifier).
        feature_names (list): Modelde kullanılan özellik isimleri.

    Returns:
        matplotlib.figure.Figure: Oluşturulan özellik önem grafiği
    """
    importances = model.feature_importances_
    sorted_indices = importances.argsort()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in sorted_indices],
        importances[sorted_indices],
        color="teal"
    )

    ax.set_title("Özellik Önem Sıralaması", fontsize=14)
    ax.set_xlabel("Önem Skoru")
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    # Örnek kullanım (RandomForestClassifier ile)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    fig = plot_feature_importance(model, feature_names)
    plt.show()


