# Import Libraries

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# Preprocess - Flatten and Normalize
X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
X_test_flat = X_test.reshape(len(X_test), -1) / 255.0

# Train kNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_flat, y_train)

# Predict and Evaluate
y_pred = knn.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
