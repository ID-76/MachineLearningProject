import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cluster import Birch
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

#Concatenate everything
df_processed = pd.concat([df_num, df_bool, df_sun], axis=1)
print("Processed columns:", df_processed.columns)

# Scale data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

print("Final shape:", X_scaled.shape)
"""
# Optuna search

db_file = "Clustering/optuna_birch.db"

study = optuna.create_study(
    study_name="birch_optimization",
    storage=f"sqlite:///{db_file}",
    direction="maximize",     # Silhoette: the higher the better
    load_if_exists=False
)

def objective(trial):
    # Hyperparameters to optimize
    threshold = trial.suggest_float("threshold", 0.5, 5.0)
    branching_factor = trial.suggest_int("branching_factor", 20, 250)
    n_clusters = trial.suggest_int("n_clusters", 3, 10)
    
    X_sample = resample(X_scaled, n_samples=50000, random_state=42)
    
    birch = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters
    )
    
    labels = birch.fit_predict(X_sample)

    unique_clusters = len(set(labels))
    
    #Penalize if n_cluster < 2
    if unique_clusters < 2:
        return -1.0  

    score = silhouette_score(X_sample, labels)
    return score

#Execute optimization
study.optimize(objective, n_trials=40)

print("Best found hyperparameters:")
print(study.best_params)

print("\nMejor valor de BIC:", study.best_value)
"""
#Best found hyperparameters: {'threshold': 2.8985317784980644, 'branching_factor': 222, 'n_clusters': 3}
best_threshold = 2.8985317784980644
best_branching = 222

X_sample = resample(X_scaled, n_samples=50000, random_state=42)

birch = Birch(
    threshold=best_threshold,
    branching_factor=best_branching,
    n_clusters=3
)

labels = birch.fit_predict(X_sample)
n_clusters = len(set(labels))

print("Clusters:", n_clusters)
print("Silhouette:", silhouette_score(X_sample, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))
