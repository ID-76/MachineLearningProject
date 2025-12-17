import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

print("="*80)
print("REGRESIÓN LINEAL MÚLTIPLE: SGD CON ELASTIC NET")
print("="*80)

# 1. CARGAR DATASET
df = pd.read_csv("Data/US_Accidents_processed_for_modeling.csv")
df.columns = [c.replace(";;;;;;;;;;", "").replace(";", "").strip() for c in df.columns]
print(f"\n Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# 2. CREAR VARIABLE OBJETIVO
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
df = df.dropna(subset=["Start_Time", "End_Time"]).reset_index(drop=True)
df["duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0
print(f" Variable objetivo creada: duration_min")

# 3. FILTRAR OUTLIERS
mask = (df["duration_min"] >= 1) & (df["duration_min"] <= 240)
df = df[mask].reset_index(drop=True)
print(f" Outliers filtrados: {df.shape[0]:,} filas")

# 4. SAMPLING
USE_SAMPLING = True
SAMPLE_SIZE = 50000

if USE_SAMPLING and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f" Muestra: {SAMPLE_SIZE:,} filas")

# 5. SEPARAR Y y X
y_raw = df["duration_min"].copy()

cols_to_drop = [
    "ID", "Description", "Street", "City", "County", "State", 
    "Zipcode", "Country", "Timezone", "Airport_Code", "Weather_Timestamp",
    "End_Time", "End_Lat", "End_Lng", "Severity", "Distance(mi)", "duration_min"
]

X = df.drop(columns=cols_to_drop, errors='ignore')

# 6. FEATURES TEMPORALES
X["Month"] = X["Start_Time"].dt.month
X["DayOfWeek"] = X["Start_Time"].dt.dayofweek
X["Hour"] = X["Start_Time"].dt.hour
X["Hour_Sin"] = np.sin(2 * np.pi * X["Hour"] / 24)
X["Hour_Cos"] = np.cos(2 * np.pi * X["Hour"] / 24)
X["IsWeekend"] = (X["DayOfWeek"] >= 5).astype(int)
X = X.drop(columns=["Start_Time", "Hour"])
print(f" Features temporales creadas")

# 7. ONE-HOT ENCODING
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
TOP_N_CATEGORIES = 20

for col in categorical_cols:
    if X[col].nunique() > TOP_N_CATEGORIES:
        top_cats = X[col].value_counts().nlargest(TOP_N_CATEGORIES).index
        X[col] = X[col].where(X[col].isin(top_cats), other='Other')

X = pd.get_dummies(X, drop_first=True, dtype=float)
print(f" One-hot encoding: {X.shape[1]} features")

# 8. IMPUTAR
feature_names = X.columns.tolist()
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
print(f" Imputación completada")

# 9. TRANSFORMACIÓN LOGARÍTMICA
X = X.reset_index(drop=True)
y_raw = y_raw.reset_index(drop=True)
y = np.log1p(y_raw)
print(f" Transformación logarítmica aplicada")

# 10. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f" Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# 11. ESCALADO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f" Escalado completado")


# 12. CONFIGURACIÓN DE MODELOS
modelos = {
    "SGD + Elastic Net": SGDRegressor(
        penalty='elasticnet', alpha=0.01, l1_ratio=0.5, 
        learning_rate='adaptive', eta0=0.01, max_iter=5000, random_state=42
    ),
    "SGD Puro (Sin Regularización)": SGDRegressor(
        penalty=None, # Aquí quitamos Elastic Net
        learning_rate='adaptive', eta0=0.01, max_iter=5000, random_state=42
    )
}

resultados = {}

# 13. ENTRENAMIENTO Y EVALUACIÓN COMPARTIVA
for nombre, model in modelos.items():
    model.fit(X_train_scaled, y_train)
    y_pred_log = model.predict(X_test_scaled)
    
    # Clip para estabilidad y des-transformación
    y_pred_log = np.clip(y_pred_log, a_min=None, a_max=np.log1p(240))
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    
    # Métricas
    resultados[nombre] = {
        'R2': r2_score(y_test_real, y_pred),
        'MAE': mean_absolute_error(y_test_real, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_real, y_pred))
    }

# 14. MOSTRAR COMPARATIVA EN TABLA
df_res = pd.DataFrame(resultados).T
print("\n" + "="*40)
print("COMPARATIVA DE RESULTADOS")
print("="*40)
print(df_res.to_string())

# 15. VISUALIZACIÓN DE COMPARATIVA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparativa: Elastic Net vs SGD Puro', fontsize=16, fontweight='bold')

metrics = ['R2', 'MAE', 'RMSE']
colors = ['skyblue', 'salmon']

for i, metric in enumerate(metrics):
    axes[i].bar(df_res.index, df_res[metric], color=colors, edgecolor='black', alpha=0.8)
    axes[i].set_title(f'Comparación de {metric}')
    axes[i].grid(axis='y', linestyle='--', alpha=0.6)
    if metric == 'R2':
        axes[i].set_ylim(-0.1, 0.1) # Ajustado para ver pequeñas diferencias cerca de 0

plt.tight_layout()
plt.show()
