# Implement Adaboost classifier using scikit-learn. Use the Iris dataset.

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score

import pandas as pd

# Load the Iris dataset
df = pd.read_csv("SMILES_only_filtered.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier with a DecisionTree base estimator
adaboost = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),  # Weak learner
    n_estimators=50,  # Number of boosting rounds
    learning_rate=1.0,
    random_state=42
)

# Train the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f"R2-score : {r2:.4f}")

























# # Print classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=iris.target_names))