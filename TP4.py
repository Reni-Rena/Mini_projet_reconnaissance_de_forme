import matplotlib
matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler


# Q1. Lire une image couleur
img = cv2.imread("red_car.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("voiture rouge")
plt.axis('off')
plt.show()

# Q2. Aplatir l’image en pixels RGB
pixel_data = img_rgb.reshape((-1, 3))

# Q3. Clustering KMeans
k = 1
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_data)

# Q4. Générer l’image segmentée
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
segmented_image = segmented_pixels.reshape(img_rgb.shape)

# Q5. Visualisation
plt.imshow(segmented_image)
plt.title(f"Image segmentée (k={k})")
plt.axis('off')
plt.show()

# ---------------------------------------------------

# Charger les données digits
digits = load_digits()
data = digits.data
labels = digits.target

# Standardisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Réduction de dimension (ACP)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Clustering hiérarchique (méthode de Ward)
linked = linkage(data_pca, method='ward')

# Affichage du dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title("Dendrogramme hiérarchique")
plt.xlabel("Groupes")
plt.ylabel("Distance")
plt.show()

# Découpage en clusters
clusters = fcluster(linked, t=10, criterion='maxclust')

# Visualisation des clusters (en 2D)
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='tab10', s=15)
plt.title("Clusters hiérarchiques sur digits (ACP)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()