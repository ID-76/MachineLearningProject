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

# 12. ENTRENAR SGD CON ELASTIC NET (AJUSTADO PARA ESTABILIDAD)
print(f"\nENTRENAMIENTO")

start_time = time.time()

model = SGDRegressor(
    loss='squared_error',
    penalty='elasticnet',
    alpha=0.0001,        # Reducido para evitar subajuste
    l1_ratio=0.15,       # Ratio estándar de Elastic Net
    max_iter=5000,       # Más iteraciones para asegurar convergencia
    tol=1e-3,
    learning_rate='adaptive', # CAMBIO CLAVE: Más estable que 'optimal'
    eta0=0.01,                # Tasa inicial
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# 13. PREDICCIONES
y_pred_log = model.predict(X_test_scaled)
# Clip para evitar valores extremos antes de expm1
y_pred_log = np.clip(y_pred_log, a_min=None, a_max=np.log1p(240)) 
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

# 14. EVALUACIÓN
r2 = r2_score(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))

print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.2f} min")
print(f"RMSE: {rmse:.2f} min")

# 18. VISUALIZACIÓN SIMPLIFICADA (SOLO MÉTRICAS)
metrics = ['R²', 'MAE (min)', 'RMSE (min)']
values = [r2, mae, rmse]

plt.figure(figsize=(12, 5))

# Subplot para R2 (Escala 0 a 1 usualmente)
plt.subplot(1, 2, 1)
plt.bar(['R²'], [r2], color='skyblue', edgecolor='black')
plt.ylim(0, 1) # El R2 ideal es 1
plt.title('Coeficiente de Determinación')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot para Errores (MAE y RMSE)
plt.subplot(1, 2, 2)
plt.bar(['MAE', 'RMSE'], [mae, rmse], color=['orange', 'salmon'], edgecolor='black')
plt.title('Métricas de Error (Minutos)')
plt.ylabel('Minutos')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()