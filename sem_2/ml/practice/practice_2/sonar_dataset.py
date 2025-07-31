import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
file_path = "Sonar.csv"
columns = [f"Feature_{i+1}" for i in range(60)] + ["Target"]
data = pd.read_csv(file_path, header=None, names=columns)

# EDA
# 1. Overview of the dataset
print(data.info())
print(data.describe())
print("\nClass Distribution:")
print(data['Target'].value_counts())

# 2. Visualize feature distributions
data.iloc[:, :-1].hist(figsize=(15, 15), bins=20)
plt.tight_layout()
plt.show()

# 3. Correlation heatmap
correlation_matrix = data.iloc[:, :-1].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap of Features")
plt.show()

# 4. Boxplot to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.iloc[:, :-1], orient="h", palette="Set3")
plt.title("Boxplot of Features")
plt.show()

# Feature Transformation
# 1. Normalize features (Min-Max Scaling)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.iloc[:, :-1])
data.iloc[:, :-1] = scaled_features

# 2. Encode the target variable
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

# Save the transformed dataset to a new CSV file
data.to_csv("Sonar_Transformed.csv", index=False)

print("\nData after transformations:")
print(data.head())

