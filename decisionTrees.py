import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("traffic_accidents_cleaned.csv")

# 2. Select target column
target = "crash_type"

X = data.drop(columns=[target])
y = data[target]

# 3. Convert categorical variables to dummy/one-hot encoding
X = pd.get_dummies(X)

# Convert target to numeric labels (if categorical)
y = y.astype("category").cat.codes

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# 5. Train an ID3-style Decision Tree (entropy)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# 6. Predict
y_pred = clf.predict(X_test)

# 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred)) # In our case, around 0.79 of accuracy

# 8. Visualize the tree (uncomment to see the plot)
# plt.figure(figsize=(25,15))
# tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
# plt.show()

# 9. Train a CART Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# 10. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)) # In our case, around 0.79 of accuracy, similar to the previous one in almost every time we run it

# 11. Visualize the tree (uncomment to see the plot)
# plt.figure(figsize=(25,15))
# tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
# plt.show()

# 12. Train a Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,        # número de árboles
    max_depth=None,         # crecer hasta que cada hoja sea pura
    random_state=42,
    n_jobs=-1               # usa todos los núcleos del CPU
)

model.fit(X_train, y_train)

# 6. Predicción
y_pred = model.predict(X_test)

# 7. Performance
print("Accuracy:", accuracy_score(y_test, y_pred)) # In our case, around 0.83 of accuracy