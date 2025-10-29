import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("US_Accidents_March23.csv")
pd.set_option('display.max_columns', None)


#Some functions to analise the variables
#df.info()
#print(df.shape)
#print(df.describe())
#print(df.describe(include = object))
