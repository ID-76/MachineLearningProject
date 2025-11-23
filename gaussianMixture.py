import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("US_Accidents_processed_for_modeling.csv")

# Drop problematic variables as in kMeans
df_reduced = df.drop(columns=[
    "Start_Time", "End_Time", "City", "County", "State", "Zipcode", "Country",
    "Timezone", "Airport_Code", "Wind_Direction", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight", "Amenity", "Bump", "Give_Way",
    "No_Exit", "Railway", "Roundabout", "Traffic_Calming", "Turning_Loop",
    "Weather_Condition"
])

# Dummies
num_cols = df_reduced.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df_reduced.select_dtypes(include=['object', 'category']).columns
df_cat_dummies = pd.get_dummies(df_reduced[cat_cols], drop_first=False)
df_num = df_reduced[num_cols]
df_processed = pd.concat([df_num, df_cat_dummies], axis=1)

# Escale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)
print("Dimensiones después de dummies:", df_processed.shape)

"""
pca_components = [2, 3, 5, 7, 10]
for n in pca_components:
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # muestreo
    X_sample = resample(X_pca, n_samples=100000, random_state=42)

    db = DBSCAN(eps=0.4, min_samples=5)
    db.fit(X_sample)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("\nPCA =", n)
    print("Clusters detectados:", n_clusters)
    print("Ruido:", list(labels).count(-1))

    if n_clusters >= 2:
        print("Silhouette:", silhouette_score(X_sample, labels))
        print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
        print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))
    else:
        print("No hay suficientes clusters para calcular métricas.")
"""
#When I increase the PCA the data separates in hundred of clusters, so I will leave it in PCA = 2

"""
eps_values = [0.3, 0.4, 0.5, 1.0]
min_samples = 5

for eps in eps_values:
    # Tomamos muestra de 50k puntos para evitar MemoryError
    X_sample = resample(X_pca, n_samples=100000, random_state=43)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X_sample)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nDBSCAN eps={eps}, min_samples={min_samples}")
    print(f"Clusters detectados: {n_clusters}, puntos de ruido: {n_noise}")

    if n_clusters >= 2:
        print("Silhouette:", silhouette_score(X_sample, labels))
        print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
        print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))
    else:
        print("No hay suficientes clusters para calcular métricas.")
"""
#After analyising the result, I'm going to choose eps 0.4 


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# muestreo
X_sample = resample(X_pca, n_samples=100000, random_state=42)

db = DBSCAN(eps=0.4, min_samples=5)
db.fit(X_sample)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Clusters detectados:", n_clusters)
print("Ruido:", list(labels).count(-1))

if n_clusters >= 2:
    print("Silhouette:", silhouette_score(X_sample, labels))
    print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
    print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))
else:
    print("No hay suficientes clusters para calcular métricas.")