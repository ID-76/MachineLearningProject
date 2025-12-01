import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

# LOAD DATA
df = pd.read_csv("US_Accidents_processed_for_modeling.csv")

# Drop non-informative / ID / text fields
df_reduced = df.drop(columns=[
    "Start_Time", "End_Time", "City", "County", "State", "Zipcode", "Country",
    "Timezone", "Airport_Code", "Wind_Direction", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight", "Amenity", "Bump", "Give_Way",
    "No_Exit", "Railway", "Roundabout", "Traffic_Calming", "Turning_Loop",
    "Weather_Condition", "Start_Lat", "Start_Lng", "Wind_Chill(C)", "Distance(km)", "Stop",
    "Junction", "Station"
])

print("Columns after reduction:", df_reduced.columns)

num_cols = ['Severity', 'Humidity(%)', 'start_hour', 'start_dayofweek', 'start_month',
            'duration_min', 'Temperature(C)', 'Wind_Speed(m/s)', 'Visibility(km)',
            'Precipitation(mm)', 'Pressure(hPa)']
df_num = df_reduced[num_cols]

bool_cols = ['Crossing', 'Traffic_Signal']
df_bool = df_reduced[bool_cols].astype(int)

# Make dummies for categorical variables
df_sun = pd.get_dummies(df_reduced['Sunrise_Sunset'], prefix='Sunrise_Sunset', drop_first=True)

# Concatenate everything
df_processed = pd.concat([df_num, df_bool, df_sun], axis=1)
print("Processed columns:", df_processed.columns)

# Scale data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

print("Final shape:", X_scaled.shape)
"""
# Optuna search
db_file = "Clustering/optuna_gmm.db"
if os.path.exists(db_file):
    os.remove(db_file)

study = optuna.create_study(
    study_name="gmm_optimization",
    storage=f"sqlite:///{db_file}",
    direction="maximize",     # Silhouette: the higher the better
    load_if_exists=False
)

def objective(trial):
    # Hyperparameters to optimize
    n_components = trial.suggest_int("n_components", 2, 5)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
    
    # Use sample for speed
    X_sample = resample(X_scaled, n_samples=50000, random_state=43)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42
    )

    labels = gmm.fit_predict(X_sample)

    unique_clusters = len(set(labels))

    # Penalize if all points end in 1 cluster
    if unique_clusters < 2:
        return -1.0
    
    score = silhouette_score(X_sample, labels)
    return score

# Execute optimization
study.optimize(objective, n_trials=40)

print("Best found hyperparameters:")
print(study.best_params)

print("\nBest Silhouette:", study.best_value)
"""

# best params from optuna: {'n_components': 2, 'covariance_type': 'spherical'}
best_n_components = 2
best_covariance = "spherical"

X_sample = resample(X_scaled, n_samples=50000, random_state=43)

gmm = GaussianMixture(
    n_components=best_n_components,
    covariance_type=best_covariance,
    random_state=42
)

labels = gmm.fit_predict(X_sample)
unique_clusters = len(set(labels))

print("Clusters:", unique_clusters)
print("Silhouette:", silhouette_score(X_sample, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))