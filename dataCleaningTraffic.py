import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("traffic_accidents.csv")
pd.set_option('display.max_columns', None)

#Some functions to analise the variables
#df.info()
#print(df.shape)
#print(df.describe())
#print(df.describe(include = object))

#First we are going to convert the float type variables into int, as they are discrete variables
float_columns = ['injuries_total', 'injuries_fatal', 'injuries_incapacitating',
    'injuries_non_incapacitating', 'injuries_reported_not_evident',
    'injuries_no_indication']

df[float_columns] = df[float_columns].astype(int)
#df.info()

#We are going to remove the crash_date column as it isn't useful
df = df.drop(columns= "crash_date")

#We can see that in this dataset there aren't any NA values
#print(df.isnull().mean() * 100)

#We can see that in this dataset the string "empty" data is put as "UNKNOWN", 
#so we are going to replace them by null values
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    condicion_vacia = (df[col].astype(str).str.strip() == "UNKNOWN")
    #print(f"Columna '{col}': {condicion_vacia.sum()} valores vacíos (no NaN)")
    df.replace("UNKNOWN", np.nan, inplace=True)    

#After replacing the "UNKNOWN" values we can see that are null values in the dataset
#print(df.isnull().sum().sort_values(ascending=False))

#As the number of na values in each class isn't too high 
# we are going to replace the values by doing a decision tree
#print(df.isnull().mean() * 100)

columns_with_na = ["traffic_control_device", "weather_condition", "lighting_condition", "roadway_surface_cond", "road_defect", "trafficway_type"]
for target in columns_with_na:
    
    train_df = df[df[target].notna()]
    test_df = df[df[target].isna()]

    #print("\nDistribution of values before data inputation (original data):")
    #print(df[target].value_counts())

    X_train = pd.get_dummies(train_df.drop(columns=[target]))
    y_train = train_df [target]

    X_test = pd.get_dummies(test_df.drop(columns=[target]))

    #Necesary funtion after applying the dummies function, so both dataframes have same columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    clf = DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    df.loc[df[target].isna(), target] = preds

    #print("Distribution of target after inputation:")
    #print(df[target].value_counts())

#print(df.isnull().mean() * 100)

#Looking at the description of the dataset we can see that are outliers, but they don't look as being errors, 
#so for this first data cleaning we are not going to remove outliers, 
#as we will remove them or not depending of each model we are doing

#We store the cleaned dataframe as a csv to work with it in the future
#df.to_csv("traffic_accidents_cleaned.csv", index = False)