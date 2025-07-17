# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load dataset
df = pd.read_csv("bank-additional-full.csv", sep=';')  # LOCAL FILE

# Step 3: Encode categorical features using LabelEncoder
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Step 4: Prepare feature matrix (X) and target vector (y)
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

# Step 5: Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train full decision tree classifier
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(x_train, y_train)

# Step 7: Train simplified tree with max_depth = 4
clf_small = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_small.fit(x_train, y_train)

# Step 8: Plot Confusion Matrix
y_pred = clf_full.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='red')
plt.tight_layout()
plt.show()

# Step 9: Plot Feature Importance
importances = clf_full.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Step 10: Plot Simplified Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf_small, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree (max_depth=4)")
plt.show()