from sklearn.decomposition import PCA
import torch
import numpy as np
x = np.random.rand(100,200)
pca = PCA(n_components=x.shape[0])
y = pca.fit_transform(x)
np.cumsum(pca.explained_variance_ratio_)
cummulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
index_99_variance = max(np.argmax(cummulative_variance_ratio >= 0.99), 1)
y = y[:, :index_99_variance]

pca.components_ = pca.components_[:index_99_variance, :]
x_reconstructed = pca.inverse_transform(y)