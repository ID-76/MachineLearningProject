import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
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
db_file = "Clustering/optuna_kmeans.db"
if os.path.exists(db_file):
    os.remove(db_file)

study = optuna.create_study(
    study_name="kmeans_optimization",
    storage=f"sqlite:///{db_file}",
    direction="maximize",     # Silhouette: the higher the better
    load_if_exists=False
)

def objective(trial):
    # Hyperparameters to optimize
    n_clusters = trial.suggest_int("n_clusters", 3, 10)
    n_init = trial.suggest_int("n_init", 5, 30)

    X_sample = resample(X_scaled, n_samples=50000, random_state=42)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=42
    )
    
    labels = kmeans.fit_predict(X_sample)
    unique_clusters = len(set(labels))
    
    # Penalize if n_cluster < 2
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

# Best found hyperparameters with optuna: {'n_clusters': 9, 'n_init': 24}
best_n_clusters = 9
best_n_init = 24

X_sample = resample(X_scaled, n_samples=50000, random_state=42)

kmeans = KMeans(
    n_clusters=best_n_clusters,
    init="k-means++",
    n_init=best_n_init,
    random_state=42
)

labels = kmeans.fit_predict(X_sample)

print("Clusters:", best_n_clusters)
print("Silhouette:", silhouette_score(X_sample, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_sample, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_sample, labels))