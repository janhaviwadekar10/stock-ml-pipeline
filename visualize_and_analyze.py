import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
from matplotlib.colors import LinearSegmentedColormap

def make_all_plots():
    # make sure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # load the combined features and drop gaps
    df = pd.read_csv("data/combined_features.csv").dropna()

    # pick the same feature set we’ve been using
    features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
                'daily_return', 'ma_7', 'ma_30', 'ma_90', 'volatility_30']
    X = df[features]

    # correlation heatmap to see who moves together
    cmap = LinearSegmentedColormap.from_list('red_white_green', ['red', 'white', 'green'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap=cmap, center=0)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/heatmap_correlation.png")
    plt.close()

    # quick PCA to squash features into 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.5)
    plt.title("PCA: 2D Projection of Features")
    plt.savefig("outputs/pca_plot.png")
    plt.close()

    # t-SNE for a nonlinear 2D peek at the space
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.5)
    plt.title("t-SNE: 2D Projection of Features")
    plt.savefig("outputs/tsne_plot.png")
    plt.close()

    # k-means to get a quick feel for cluster structure
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # basic cluster quality scores for a gut check
    sil_score = silhouette_score(X, cluster_labels)
    ch_score = calinski_harabasz_score(X, cluster_labels)
    db_score = davies_bouldin_score(X, cluster_labels)
    print(f"Silhouette Coefficient: {sil_score:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")

    # plot clusters on the PCA map for something visual
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', s=10)
    plt.title("KMeans Clusters on PCA Projection")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig("outputs/kmeans_clusters_pca.png")
    plt.close()

if __name__ == "__main__":
    make_all_plots()