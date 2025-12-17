import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------------------------------
# 1. Cargar dataset & LIMPIEZA DE NOMBRES (Fix Critical)
# -------------------------------------------------------------------
df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")

# --- FIX: Cleaning dirty column names (removing ;;;;;) ---
df.columns = [col.replace(";", "").strip() for col in df.columns]

print("Dataset cargado correctamente.")
print("\nColumnas limpias:", df.columns.tolist())


# -------------------------------------------------------------------
# 2. Seleccionar variable objetivo
# -------------------------------------------------------------------

"""
We are going to try and predict the duration of accidents in minutes. 
For that, we need to calculate the duration from Start_Time and End_Time.
We use pd.to_datetime to convert the columns to datetime objects, and then calculate the difference and
convert it to minutes by dividing the total seconds by 60.
"""

df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# Eliminar filas con fechas inválidas
df = df.dropna(subset=["Start_Time", "End_Time"])

# Note: Dividing by 60 for minutes (3606 was a typo in original code)
df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0
y = df["Duration_min"]

"""
Then, we filter some of the most extreme outliers to make the model more stable.
Before the filter, the dataset had outliers that heavyly skewed the distribution of durations.
After applying a filter to keep only durations between 0 and 240 minutes (4 hours),
"""

mask = (df["Duration_min"] > 0) & (df["Duration_min"] < 240)
df = df[mask]

# Actualizamos 'y' después del filtro
y = df["Duration_min"]

print(f"Filas después de eliminar outliers extremos: {df.shape[0]}")


# -------------------------------------------------------------------
# 3. Limpiar columnas problemáticas
# -------------------------------------------------------------------

""""
We will remove columns that have little to no useful information for the model.
These include long text columns that are unlikely to help predict duration, such as ID, STREET
or Descrpition, that without any further information are not useful.
"""

cols_texto_largo = [
    "ID", "Description", "Street", "City", "Timezone", "Airport_Code",
    "Zipcode", "Weather_Timestamp", "County", "State", "Country" 
]

df = df.drop(columns=cols_texto_largo, errors='ignore')


"""
Finally, we need to define which columns are "leakage" columns.
These are columns that contain information that would not be available at the time of prediction.
Of course, we have to remove all of those columns that refeer to when the accident ended.
If we give the model the starting time, but then ALSO the ending time, it can trivially calculate the duration.
Therefore, it will not predict anything useful, just memorize the data.
"""

# Identificamos cualquier columna que empiece por "End_" (Time, Lat, Lng)
cols_leakage = [c for c in df.columns if c.startswith("End_")]
print(f"Variables de Leakage detectadas y eliminadas: {cols_leakage}")


# -------------------------------------------------------------------
# 4. Convertir columnas de fecha a valores numéricos
# -------------------------------------------------------------------

"""
We want to extract useful information from the Start_Time column.
We will create new columns for year, month, day, and hour.
In this way, the model can learn patterns based on these time features.
This can be very useful, as accidents may have different durations depending on the time of day or month.
"""

def procesar_fecha(df, columna):
    df[columna] = pd.to_datetime(df[columna], errors="coerce")
    df[columna + "_year"] = df[columna].dt.year
    df[columna + "_month"] = df[columna].dt.month
    df[columna + "_day"] = df[columna].dt.day
    df[columna + "_hour"] = df[columna].dt.hour
    df.drop(columns=[columna], inplace=True)

# SOLO procesamos Start_Time (End_Time es leakage y se borrará)
procesar_fecha(df, "Start_Time")


# -------------------------------------------------------------------
# 5. Preparar X (FIXED: REMOVING ALL LEAKAGE)
# -------------------------------------------------------------------

"""
Now that we have cleaned the dataset, we can prepare the feature matrix X.
We will drop the target variable Duration_min and the leakage columns defined earlier.
CRITICAL FIX: We must also drop 'Distance(mi)' and 'Severity'. 
- Distance is measured after the fact (leakage).
- Severity is often calculated based on duration (leakage).
If we keep them, the model gets an R2 of 1.0, which is cheating.
"""

# 1. Define strict leakage columns
cols_leakage_strict = [
    "Distance(mi)", 
    "Severity",
    "End_Lat", 
    "End_Lng"
]

# 2. Combine with previous time leakage (columns starting with End_)
cols_time_leakage = [c for c in df.columns if c.startswith("End_")]

# 3. Create the final drop list
# Combine: Target + Time Leakage + Strict Leakage
drop_list = ["Duration_min"] + cols_time_leakage + cols_leakage_strict

print(f"Dropping these columns to prevent cheating: {drop_list}")

# 4. Drop them
X = df.drop(columns=drop_list, errors='ignore')


"""
Lastly, before continuing, we will perform a quick check to ensure that no leakage columns remain in X.
This is a safety check to avoid accidental inclusion of these columns.
"""
# Verificación de seguridad
print(f"Columnas en X: {X.shape[1]}")
if "Distance(mi)" in X.columns or "Severity" in X.columns:
    print("¡ALERTA! Distance o Severity siguen ahí. El modelo hará trampa.")
else:
    print("Correcto: El modelo ahora es 'ciego' al futuro y a la magnitud del accidente.")

# -------------------------------------------------------------------
# 6. Convertir categóricas pequeñas en dummies
# -------------------------------------------------------------------

"""
To not waste some potentially useful information, we will convert categorical variables 
with a small number of unique values into dummy/indicator variables.
This way, the model can learn from these categorical features without being overwhelmed 
by too many dummy variables
"""
X = pd.get_dummies(X, drop_first=True)

# --- ¡NUEVO! Guardamos los nombres aquí, mientras X sigue siendo un DataFrame ---
feature_names = X.columns.tolist() 


# -------------------------------------------------------------------
# 7. Imputar valores faltantes
# -------------------------------------------------------------------

"""
We will use a SimpleImputer to fill in any missing values in the dataset.
We will use the median strategy, which is robust to outliers.
"""
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X) # AQUI X SE CONVIERTE EN NUMPY ARRAY Y PIERDE NOMBRES


# -------------------------------------------------------------------
# 8. Dividir train/test
# -------------------------------------------------------------------

"""
The moment of truth: we will split the dataset into training and testing sets.
We will use 80% of the data for training and 20% for testing.
Of course, this is done to provide an unbiased evaluation of the model's performance on unseen data.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------------------------
# 9. Escalar datos
# -------------------------------------------------------------------

"""
To ensure that all features contribute equally to the model, we will standardize the feature values.
This is especially important for linear regression, as it is sensitive to the scale of the input features
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------------------------
# 10. Entrenar modelo
# -------------------------------------------------------------------

"""
Training time has arrived! We will use a simple Linear Regression model from scikit-learn.
This model will learn the relationship between the features and the target variable (duration in minutes).
We assume it will probably not be the most powerful model, but it will give us a good baseline to compare 
against more complex models later.
With seeing how the data reaacts to a simple linear model, we can get insights into its structure, features and 
overall behavior of the dataset in more depth.
"""
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nModelo entrenado correctamente.")

# 1. El Intercepto
print(f"\nIntercepto (Beta 0): {model.intercept_:.4f}")

# 2. Los Coeficientes
coeficientes = pd.DataFrame({
    'Variable': feature_names,
    'Peso (Coeficiente)': model.coef_
})

# Ordenar por valor absoluto para ver las más influyentes
coeficientes['Importancia'] = coeficientes['Peso (Coeficiente)'].abs()
coeficientes = coeficientes.sort_values(by='Importancia', ascending=False)

print("\nTop 10 variables más influyentes en la ecuación:")
print(coeficientes.head(10))


# -------------------------------------------------------------------
# 11. Predicciones
# -------------------------------------------------------------------

"""
With the model trained, it is now time to make predictions on the test set.
We will use the trained model to predict the duration of accidents in minutes for the test data.
"""
y_pred = model.predict(X_test_scaled)

print("\nPrimeras predicciones:")
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Real: {real} | Predicho: {pred:.5f}")


# -------------------------------------------------------------------
# 13. Evaluación del Modelo & Gráfico
# -------------------------------------------------------------------

""""
Lastly, we will evaluate the model's performance using common regression metrics.
These metrics will help us understand how well the model is performing and where it can be improved.
Of course, we will also take a look at some visualizations to get a better sense of the predictions 
versus the actual values.
"""

# 1. Calcular métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # La raíz cuadrada del MSE

# 2. Mostrar resultados
print("\n" + "="*40)
print("     EVALUACIÓN DEL MODELO")
print("="*40)
print(f"R² (Coeficiente de determinación): {r2:.4f}")
print(f"MAE (Error promedio absoluto):     {mae:.2f} minutos")
print(f"RMSE (Error cuadrático medio):     {rmse:.2f} minutos")
print("-" * 40)

# Interpretación automática simple
if r2 < 0.1:
    print(">> Interpretación: El modelo tiene muy poca capacidad predictiva.")
    print("   Probablemente necesitas más variables relevantes o limpiar outliers.")
elif r2 < 0.5:
    print(">> Interpretación: El modelo captura algunas tendencias, pero hay mucho ruido.")
else:
    print(">> Interpretación: El modelo está encontrando patrones sólidos.")


plt.scatter(y_test, y_pred + np.random.normal(0, 1, size=len(y_pred)), alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Regresión Lineal – Real vs Predicho (con jitter)")
plt.show()

""""
So, conclusions are clear. After we cleaned the model, the linear regression model performs decent, but
not great. We can see that there is still a lot of room for improvement. Of course, that was expected, as
linear regression is a simple model and is not the most fit solution for a dataset lke this.

Apart from the grapgh that is veary revealing, we can check the metrics:
========================================
     EVALUACIÓN DEL MODELO
========================================
R² (Coeficiente de determinación): 0.2659
MAE (Error promedio absoluto):     28.98 minutos
RMSE (Error cuadrático medio):     39.13 minutos
----------------------------------------
"""