import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

df = pd.read_csv("US_Accidents_processed_for_modeling.csv")

#We remove those variables that are categorical and have too many unique values for doing dummies
#Also we drop some boolean variables that has almost 0% of False and 100% of True, as they dont't provide information
df_reduced = df.drop(columns = ["Start_Time", "End_Time", "City", "County", "State", "Zipcode", "Country", "Timezone", "Airport_Code", "Wind_Direction", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight", "Amenity", "Bump", "Give_Way", "No_Exit", "Railway", "Roundabout", "Traffic_Calming", "Turning_Loop", "Weather_Condition"])
#We create dummy variables for categorical variables
num_cols = df_reduced.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df_reduced.select_dtypes(include=['object', 'category']).columns
df_cat_dummies = pd.get_dummies(df_reduced[cat_cols], drop_first=False)
df_num = df_reduced[num_cols]
df_processed = pd.concat([df_num, df_cat_dummies], axis=1)
#We scale the data so variables with high range don't affect too much
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)
print(df_processed.shape)
print(df_processed.columns)

#We reduce the data, as when we have convert labels into numerical 
#the number of dimensions have increased too much, and to improve the clustering results we will do PCA

"""
for k in range(2, 10):
    pca = PCA(n_components=k, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance by ",k," components:", pca.explained_variance_ratio_.sum())
    kmeans = KMeans(init="k-means++", n_clusters=3, random_state=42)
    kmeans.fit(X_pca)

    labels = kmeans.labels_  # los clusters asignados

    sample_data, sample_labels = resample(X_pca, labels, n_samples=50000, random_state=42)

    print(silhouette_score(sample_data, sample_labels))
    print(calinski_harabasz_score(sample_data, sample_labels))
    print(davies_bouldin_score(sample_data, sample_labels))
"""
#With this loop we can see that the best number of components is 2, as while we increase it the performance decrease

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

#n_digits = np.unique(df["Severity"]).size
#Training
#We will consider the Severity measure as a "target" value, 
#and after applying clustering we will use it for validating the model

"""
for k in range (2, 15):
    kmeans = KMeans(init="k-means++", n_clusters=k, random_state=42)
    kmeans.fit(X_pca)

    labels = kmeans.labels_  # los clusters asignados

    sample_data, sample_labels = resample(X_pca, labels, n_samples=50000, random_state=42)

    print(k, silhouette_score(sample_data, sample_labels))
    print(k, calinski_harabasz_score(sample_data, sample_labels))
    print(k, davies_bouldin_score(sample_data, sample_labels))
"""

#After doing the loop for plenty number of clusters, we can see that the one with best results is making three clusters
#We were going to select the variable "Severity" as target, 
# but as the ideal number of clusters don't matches the number of unique labels in the variable we won't do it

kmeans = KMeans(init="k-means++", n_clusters=3, random_state=42)
kmeans.fit(X_pca)

labels = kmeans.labels_  # los clusters asignados

sample_data, sample_labels = resample(X_pca, labels, n_samples=50000, random_state=42)

print(silhouette_score(sample_data, sample_labels))
print(calinski_harabasz_score(sample_data, sample_labels))
print(davies_bouldin_score(sample_data, sample_labels))