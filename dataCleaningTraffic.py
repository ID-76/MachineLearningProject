import pandas as pd
import numpy as np

df = pd.read_csv("traffic_accidents.csv")
pd.set_option('display.max_columns', None)

#Some functions to analise the variables
#df.info()
print(df.describe())
#print(df.describe(include = object))

#We can see that in this dataset there aren't any NA values
#print(df.isnull().mean() * 100)

#We can see that in this dataset there aren't any empty strings neither
"""
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    condicion_vacia = (df[col].astype(str).str.strip() == "")
    print(f"Columna '{col}': {condicion_vacia.sum()} valores vacíos (no NaN)")
"""
