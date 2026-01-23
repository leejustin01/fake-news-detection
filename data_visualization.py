from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

X_train = np.load("./data/processed/X_train.npy")
y_train = np.load("./data/processed/y_train.npy")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train,
    alpha=0.5
)
plt.title("PCA of SBERT Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()