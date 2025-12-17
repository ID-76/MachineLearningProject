import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

## DATA FROM model_metrics.md ##

# 1. DATOS ACTUALIZADOS (Solo Clase 4)
data = {
    'Model': ['ID3', 'ID3', 'CART', 'CART', 'Random Forest', 'Random Forest'],
    'Strategy': ['Baseline', 'SMOTE', 'Baseline', 'SMOTE', 'Baseline', 'SMOTE'],
    'Recall_Class4': [0.3357, 0.8196, 0.4161, 0.7750, 0.0643, 0.8768] # Datos extraídos de tu output
}

df = pd.DataFrame(data)

# 2. CONFIGURACIÓN DEL GRÁFICO
fig, ax = plt.subplots(figsize=(10, 6))

models = df['Model'].unique()
x = np.arange(len(models))
width = 0.35

# Barras
rects1 = ax.bar(x - width/2, df[df['Strategy']=='Baseline']['Recall_Class4'], width, 
                label='Baseline', color='#A8B6CC', edgecolor='black')
rects2 = ax.bar(x + width/2, df[df['Strategy']=='SMOTE']['Recall_Class4'], width, 
                label='SMOTE', color='#E63946', edgecolor='black')

# 3. TEXTOS Y ETIQUETAS (INGLÉS)
ax.set_ylabel('Recall (Class 4 Only)', fontsize=12)
ax.set_title('Impact of SMOTE on Class 4 Accident Detection', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(title='Strategy', loc='upper left')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Anotaciones de valores
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

# 1. DATOS
# He añadido \n (saltos de línea) para que los nombres no se choquen entre sí
data = {
    'Model': ['Linear SVM', 'ID3\n(SMOTE)', 'CART\n(SMOTE)', 'Random\nForest'],
    'Severe_Recall': [0.7154, 0.8321, 0.8412, 0.8822],
    'Color': ['#FF9F1C', '#2EC4B6', '#3A86FF', '#E63946']
}

df = pd.DataFrame(data)

# 2. CONFIGURACIÓN
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df['Model']))
width = 0.6

# Crear barras
rects = ax.bar(x, df['Severe_Recall'], width, color=df['Color'], edgecolor='black', alpha=0.9)

# 3. ZOOM Y ESTÉTICA
ax.set_ylim(0.65, 0.92) 

ax.set_ylabel('Severe Recall', fontsize=12)
ax.set_title('Severe Recall Comparison: SVM vs Tree Models', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)

# ETIQUETAS EJE X: Sin rotación (0), pero con saltos de línea para que quepan bien
ax.set_xticklabels(df['Model'], fontsize=11, rotation=0) 

ax.grid(axis='y', linestyle='--', alpha=0.3)

# 4. VALORES ENCIMA DE LAS BARRAS
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects)

plt.tight_layout()
plt.show()

# 1. DATOS DE MATRICES DE CONFUSIÓN (Normalizados)
svm_cm = np.array([
    [0.01, 0.14, 0.50, 0.35], 
    [0.05, 0.15, 0.50, 0.30], 
    [0.05, 0.15, 0.68, 0.12], 
    [0.01, 0.09, 0.19, 0.71]  
])

id3_cm = np.array([
    [0.05, 0.10, 0.45, 0.40],
    [0.05, 0.08, 0.47, 0.40],
    [0.02, 0.05, 0.83, 0.10], 
    [0.00, 0.03, 0.15, 0.82]  
])

cart_cm = np.array([
    [0.08, 0.12, 0.40, 0.40],
    [0.05, 0.12, 0.43, 0.40],
    [0.02, 0.05, 0.85, 0.08], 
    [0.00, 0.02, 0.21, 0.77]  
])

rf_cm = np.array([
    [0.02, 0.05, 0.40, 0.53], 
    [0.01, 0.05, 0.44, 0.50], 
    [0.00, 0.02, 0.88, 0.10], 
    [0.00, 0.00, 0.12, 0.88]  
])

matrices = [svm_cm, id3_cm, cart_cm, rf_cm]
names = ['Linear SVM', 'ID3 (SMOTE)', 'CART (SMOTE)', 'Random Forest (SMOTE)']

# 2. GENERAR GRÁFICO
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    # Heatmap azul limpio
    sns.heatmap(matrices[i], annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
                vmin=0, vmax=1)
    
    ax.set_title(names[i], fontsize=13, fontweight='bold')
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')

plt.suptitle('Normalized Confusion Matrices: Prediction Patterns by Model', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()