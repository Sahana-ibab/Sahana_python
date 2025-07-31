import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "Weekly.csv"  # Update with the correct path to the file
data = pd.read_csv(file_path)

# EDA
# 1. Overview of the dataset
print("Dataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# 2. Target distribution (Direction column)
plt.figure(figsize=(6, 4))
sns.countplot(data['Direction'], palette="pastel")
plt.title("Target Distribution")
plt.show()

# 3. Correlation heatmap (Numerical Columns)
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Distribution of 'Today' feature
plt.figure(figsize=(6, 4))
sns.histplot(data['Today'], kde=True, bins=30, color="skyblue")
plt.title("Distribution of 'Today'")
plt.show()

# Feature Transformation
# 1. Scale numerical features
scaler = StandardScaler()
numerical_features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 2. Encode the target variable
label_encoder = LabelEncoder()
data['Direction'] = label_encoder.fit_transform(data['Direction'])  # 'Down' -> 0, 'Up' -> 1

# Save the transformed dataset
data.to_csv("Weekly_Transformed.csv", index=False)

print("\nTransformed Data (First 5 Rows):")
print(data.head())
