import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

def loocv_weekly():
    # Load Weekly dataset
    data = pd.read_csv('/home/ibab/datasets_IA2/Weekly.csv')

    # Extract relevant columns
    X = data[['Lag1', 'Lag2']].values
    y = data['Direction'].values

    # Encode target (Up/Down) as 1/0
    y = LabelEncoder().fit_transform(y)

    n = len(X)
    correct = 0  # to count correct predictions

    # Perform LOOCV
    for i in range(n):
        # Leave the i-th observation out
        X_train = X[np.arange(n) != i]
        y_train = y[np.arange(n) != i]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict on the left-out observation
        y_pred = model.predict(X_test)

        # Compare with true label
        if y_pred == y_test:
            correct += 1

    # Calculate and print LOOCV accuracy
    accuracy = correct / n
    print(f"LOOCV Test Accuracy: {accuracy:.4f}")

# Run the function
loocv_weekly()
