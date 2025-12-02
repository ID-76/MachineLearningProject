import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------------
# 1. Cargar dataset
# -------------------------------------------------------------------
df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")

print("Dataset cargado correctamente.")
print(df.head())
print("\nColumnas:", df.columns.tolist())


# -------------------------------------------------------------------
# 2. Seleccionar variable objetivo
# -------------------------------------------------------------------
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# Eliminar filas con fechas inválidas
df = df.dropna(subset=["Start_Time", "End_Time"])

df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 3606   
y = df["Duration_min"]

# -------------------------------------------------------------------
# 3. Limpiar columnas problemáticas
# -------------------------------------------------------------------

cols_texto_largo = [
    "ID", "Description", "Street", "City", "Timezone", "Airport_Code",
    "Zipcode", "Weather_Timestamp", "County", "State", "Country" 
]

df = df.drop(columns=cols_texto_largo)

cols_leakage = [
    "Start_Time", "End_Time", "Weather_Timestamp",
    "Start_Time_year", "Start_Time_month", "Start_Time_day", "Start_Time_hour",
    "End_Time_year", "End_Time_month", "End_Time_day", "End_Time_hour"
]   



# -------------------------------------------------------------------
# 4. Convertir columnas de fecha a valores numéricos
# -------------------------------------------------------------------
def procesar_fecha(df, columna):
    df[columna] = pd.to_datetime(df[columna], errors="coerce")
    df[columna + "_year"] = df[columna].dt.year
    df[columna + "_month"] = df[columna].dt.month
    df[columna + "_day"] = df[columna].dt.day
    df[columna + "_hour"] = df[columna].dt.hour
    df.drop(columns=[columna], inplace=True)

procesar_fecha(df, "Start_Time")
procesar_fecha(df, "End_Time")


# -------------------------------------------------------------------
# 5. Preparar X
# -------------------------------------------------------------------
X = df.drop(columns=["Duration_min"])


# -------------------------------------------------------------------
# 6. Convertir categóricas pequeñas en dummies
# -------------------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)


# -------------------------------------------------------------------
# 7. Imputar valores faltantes
# -------------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)


# -------------------------------------------------------------------
# 8. Dividir train/test
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------------------------
# 9. Escalar datos
# -------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------------------------
# 10. Entrenar modelo
# -------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nModelo entrenado correctamente.")


# -------------------------------------------------------------------
# 11. Predicciones
# -------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\nPrimeras predicciones:")
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Real: {real} | Predicho: {pred:.5f}")

"""
# 1. Ver si la columna objetivo está en las features
print("¿Está el target en las columnas?")
print(y.name in X.columns)
"""
# 2. Correlaciones fuertes sospechosas (>0.999)
# Para cada feature, calcular correlación con y

# 3. Ver si has usado accidentalmente X_train para generar X_test
print("Dimensiones:")
print("X:", X.shape, "y:", y.shape)
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# -------------------------------------------------------------------
# 12. Gráfico
# -------------------------------------------------------------------


plt.scatter(y_test, y_pred + np.random.normal(0, 1, size=len(y_pred)), alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Regresión Lineal – Real vs Predicho (con jitter)")
plt.show()