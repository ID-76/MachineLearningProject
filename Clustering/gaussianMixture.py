import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

# =========================================
# CARGA Y PREPROCESADO
# =========================================
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


# =========================================
# OPTIMIZACIÓN CON OPTUNA (BIC)
# =========================================
db_file = "optuna_birch.db"
if os.path.exists(db_file):
    os.remove(db_file)

study = optuna.create_study(
    study_name="gmm_optimization",
    storage=f"sqlite:///{db_file}",
    direction="minimize",     # BIC: cuanto menor mejor
    load_if_exists=False
)

def objective(trial):
    n_components = trial.suggest_int("n_components", 2, 12)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])

    # PCA a 2 componentes
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Muestra de 50k para más velocidad
    X_sample = resample(X_pca, n_samples=50000, random_state=42)

    # Gaussian Mixture
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42
    )

    gmm.fit(X_sample)

    # ============================
    # BIC como objetivo a minimizar
    # ============================
    bic = gmm.bic(X_sample)

    return bic


# =========================================
# EJECUTAR OPTIMIZACIÓN
# =========================================
study.optimize(objective, n_trials=40)

print("\n==============================")
print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
print("==============================")
print(study.best_params)

print("\nMejor valor de BIC:", study.best_value)


# =========================================
# ENTRENAMIENTO FINAL CON LOS MEJORES PARÁMETROS
# =========================================
best_n = study.best_params["n_components"]
best_cov = study.best_params["covariance_type"]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_sample = resample(X_pca, n_samples=50000, random_state=42)

gmm = GaussianMixture(
    n_components=best_n,
    covariance_type=best_cov,
    random_state=42
)

gmm.fit(X_sample)
labels = gmm.predict(X_sample)

print("\nResultados finales con los mejores parámetros:")
print("Num of clusters:", len(set(labels)))
print("Silhouette:", silhouette_score(X_sample, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))