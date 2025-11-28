import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
Here are some results with different target columns using ID3, CART and Random Forest:
- Target: "alignment" (The alignment of the road where the accident occurred)
    - ID3: ~0.95 accuracy
    - CART: ~0.95 accuracy
    - Random Forest: ~0.98 accuracy
- Target: "crash_type" (The overall type of the crash)
    - ID3: ~0.79 accuracy
    - CART: ~0.79 accuracy
    - Random Forest: ~0.83 accuracy
- Target: "damage" (The extent of the damage caused by the accident)
    - ID3: ~0.60 accuracy
    - CART: ~0.60 accuracy
    - Random Forest: ~0.70 accuracy
- Target: "traffic_control_device" (The type of traffic control device involved (e.g., traffic light, sign))
    - ID3: ~0.55 accuracy
    - CART: ~0.55 accuracy
    - Random Forest: ~0.67 accuracy
- Target: "first_crash_type" (The initial type of the crash (e.g., head-on, rear-end))
    - ID3: ~0.47 accuracy
    - CART: ~0.47 accuracy
    - Random Forest: ~0.56 accuracy
""" 

# 1. Load dataset
data = pd.read_csv("traffic_accidents_cleaned.csv")

# 2. Select target column
target = "alignment"  # Change this to try different target columns

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
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Visualize the tree (uncomment to see the plot)
# plt.figure(figsize=(25,15))
# tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
# plt.show()

# 9. Train a CART Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# 10. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Visualize the tree (uncomment to see the plot)
# plt.figure(figsize=(25,15))
# tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
# plt.show()

# 12. Train a Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,        
    max_depth=None,         
    random_state=42,
    n_jobs=-1               
)

model.fit(X_train, y_train)

# 13. Predicción
y_pred = model.predict(X_test)

# 14. Performance
print("Accuracy:", accuracy_score(y_test, y_pred))

# 15. Visualize one of the trees inside the Random Forest
# plt.figure(figsize=(25, 15))
# tree.plot_tree(
#     model.estimators_[0], 
#     filled=True,
#     feature_names=X.columns,
#     class_names=[str(c) for c in model.classes_]
# )
# plt.title("Árbol 0 del Random Forest")
# plt.show()