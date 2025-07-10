#exercice 2

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

print(iris.keys())

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

print("Nouvelle forme des données :", X_pca.shape)
print(pca.components_)



plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for target, color in zip([0, 1, 2], colors):
    plt.scatter(X_pca[iris.target == target, 0],    # PC1
                X_pca[iris.target == target, 1],    # PC2
                label=iris.target_names[target],
                c=color)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP")
plt.legend()
plt.grid(True)
plt.show()

print("\nVariance :")
print(pca.explained_variance_ratio_)
print(f"Variance totale expliquée : {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

cumulative = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative >= 0.95) + 1
print(n_components)


pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(iris.data)
print(X_pca_95.shape[1])
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='--', label='Seuil 95%')
plt.title("Cumul de la variance expliquée par les composantes principales")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.xticks(range(1, len(cumulative) + 1))
plt.grid(True)
plt.legend()
plt.show()


#exercice 3



