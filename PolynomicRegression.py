import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Cargar y Limpiar Nombres
df = pd.read_csv("Data/US_Accidents_March23_sampled_500k.csv")

# --- LIMPIEZA DE NOMBRES DE COLUMNAS (NUEVO) ---
# Quitamos espacios y esos extraños ;;;;; del final
df.columns = [c.replace(";;;;;;;;;;", "").strip() for c in df.columns]
print("Nombres de columnas limpiados:", df.columns.tolist())

# 2. Fechas y Target
df = df.dropna(subset=["Start_Time", "End_Time"])
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# Calcular duración
df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0

# Filtro estricto de Outliers (quedamos con accidentes entre 1 min y 4 horas)
mask = (df["Duration_min"] >= 1) & (df["Duration_min"] <= 240)
df = df[mask]

# --- TRUCO MATEMÁTICO: TRANSFORMACIÓN LOGARÍTMICA ---
# En lugar de predecir minutos, predecimos el Logaritmo de los minutos.
# Esto suele mejorar drásticamente modelos de tiempo/dinero.
y = np.log1p(df["Duration_min"]) 

# 3. Limpieza de Features
cols_drop_texto = [
    "ID", "Description", "Street", "City", "County", "State", "Zipcode", "Country", 
    "Timezone", "Airport_Code", "Weather_Timestamp", "Duration_min"
]

# AQUI ESTÁ LA CLAVE: Añadimos Severity y Distance a la lista negra
cols_leakage = [
    "End_Time", "End_Lat", "End_Lng", 
    "Severity", "Distance(mi)"  # <--- IMPORTANTE: Si no quitas esto, el modelo hace trampa
]

# Borramos todo
X = df.drop(columns=cols_drop_texto + cols_leakage, errors='ignore')
# 4. Ingeniería de Fechas (Cíclica)
X["Month"] = X["Start_Time"].dt.month
X["DayOfWeek"] = X["Start_Time"].dt.dayofweek
X["Hour"] = X["Start_Time"].dt.hour

# Transformación Seno/Coseno para la hora (Para que las 23h esté cerca de las 00h)
X["Hour_Sin"] = np.sin(2 * np.pi * X["Hour"] / 24)
X["Hour_Cos"] = np.cos(2 * np.pi * X["Hour"] / 24)

X = X.drop(columns=["Start_Time", "Hour"]) # Borramos la original

# 5. Dummies e Imputer
X = pd.get_dummies(X, drop_first=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# Guardamos nombres antes de perderlos
feature_names = X.columns.tolist()
X = imputer.fit_transform(X)

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Selección de Variables + Polinomio
# Bajamos a las 15 mejores para no saturar con ruido
selector = SelectKBest(score_func=f_regression, k=15)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

poly = PolynomialFeatures(degree=2 , include_bias=False)
X_train_poly = poly.fit_transform(X_train_sel)
X_test_poly = poly.transform(X_test_sel)

# 8. Escalado
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_poly)
X_test_final = scaler.transform(X_test_poly)

# 9. Modelo (Usamos RIDGE en vez de LinearRegression pura para reducir ruido)
model = Ridge(alpha=1.0)
model.fit(X_train_final, y_train)

# 10. Predicción y DES-TRANSFORMACIÓN
y_pred_log = model.predict(X_test_final)

# --- IMPORTANTE: DESHACEMOS EL LOGARITMO ---
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test) # Tus datos reales en minutos

# 11. Métricas
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("\n" + "="*40)
print(" RESULTADOS FINALES (LOG-TRANSFORM)")
print("="*40)
print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.2f} minutos")

from sklearn.ensemble import HistGradientBoostingRegressor

# -------------------------------------------------------------------
# EL MODELO DEFINITIVO: HISTOGRAM GRADIENT BOOSTING
# -------------------------------------------------------------------
# Este modelo es mucho más potente que la regresión lineal.
# No necesita escalar datos, maneja nulos y encuentra patrones complejos.

print("\nEntrenando Gradient Boosting (La artillería pesada)...")

# Usamos los mismos datos X_train, y_train (el logaritmo) del paso anterior
model_gb = HistGradientBoostingRegressor(
    max_iter=200,       # Número de árboles
    max_depth=10,       # Profundidad máxima
    learning_rate=0.1,  # Velocidad de aprendizaje
    random_state=42
)

model_gb.fit(X_train, y_train) # Usamos X_train completo (sin el polinomio, GB no lo necesita)

# Predicciones
y_pred_log_gb = model_gb.predict(X_test)
y_pred_gb = np.expm1(y_pred_log_gb) # Deshacer logaritmo
y_test_real = np.expm1(y_test)

# Métricas
r2_gb = r2_score(y_test_real, y_pred_gb)
mae_gb = mean_absolute_error(y_test_real, y_pred_gb)

print("\n" + "="*40)
print(" RESULTADOS GRADIENT BOOSTING")
print("="*40)
print(f"R²:  {r2_gb:.4f}")
print(f"MAE: {mae_gb:.2f} minutos")

# Gráfico comparativo
plt.figure(figsize=(10,6))
plt.scatter(y_test_real, y_pred_gb, alpha=0.3, color='purple', label='Gradient Boosting')
plt.plot([0, 240], [0, 240], "r--", label='Predicción Perfecta')
plt.xlabel("Minutos Reales")
plt.ylabel("Minutos Predichos")
plt.title(f"Gradient Boosting (R2: {r2_gb:.2f})")
plt.legend()
plt.show()