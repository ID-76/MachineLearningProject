import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
# ============================================
# CARGA Y PREPROCESADO (tu mismo código)
# ============================================

df = pd.read_csv("US_Accidents_processed_for_modeling.csv")

df_reduced = df.drop(columns=[
    "Start_Time", "End_Time", "City", "County", "State", "Zipcode", "Country",
    "Timezone", "Airport_Code", "Wind_Direction", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight", "Amenity", "Bump", "Give_Way",
    "No_Exit", "Railway", "Roundabout", "Traffic_Calming", "Turning_Loop",
    "Weather_Condition"
])

num_cols = df_reduced.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df_reduced.select_dtypes(include=['object', 'category']).columns

df_num = df_reduced[num_cols]
df_cat = pd.get_dummies(df_reduced[cat_cols], drop_first=False)

df_processed = pd.concat([df_num, df_cat], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

print("Dimensiones del dataset procesado:", X_scaled.shape)


# ============================================
# OPTUNA: OBJETIVE FUNCTION
# ============================================
"""
db_file = "optuna_birch.db"
if os.path.exists(db_file):
    os.remove(db_file)

study = optuna.create_study(
    study_name="birch_optimization",
    storage=f"sqlite:///{db_file}",
    direction="maximize",
    load_if_exists=False  # aseguramos que no carga estudios previos
)

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.1, 5.0)
    branching_factor = trial.suggest_int("branching_factor", 20, 250)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_sample = resample(X_pca, n_samples=50000, random_state=42)

    birch = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=None
    )

    labels = birch.fit_predict(X_sample)
    n_clusters = len(set(labels))

    # We penalize if there are less than 2 clusters or more than 10
    if n_clusters < 2:
        return -10.0 
    if(n_clusters > 10):
        return -10.0

    #We maximize the silhoette score
    silhouette = silhouette_score(X_sample, labels)

    return silhouette   # queremos MAX silhouette


# ============================================
# OPTUNA STUDY
# ============================================

study.optimize(objective, n_trials=40)

print("\n==============================")
print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
print("==============================")
print(study.best_params)

print("\nMejor valor de Silhouette:", study.best_value)
"""
#With this optune study we have seen that the optimal parameters are treshold = 1.976 and branching_factor = 156

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_sample = resample(X_pca, n_samples=50000, random_state=42)

birch = Birch(
    threshold=1.976,
    branching_factor=156,
    n_clusters=None
)

labels = birch.fit_predict(X_sample)
n_clusters = len(set(labels))

print("Num of clusters:", n_clusters)
print("Silhouette:", silhouette_score(X_sample, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))