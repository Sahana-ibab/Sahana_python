import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
file_path = "wdbc_data.csv"  # Adjust the file path as needed
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# EDA
# 1. Overview of the dataset
print("\nDataset Info:")
print(data.info())
print("\nDataset Summary:")
print(data.describe())

# 2. Check for null values
print("\nMissing Values:")
print(data.isnull().sum())

# 3. Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data['diagnosis'], palette="pastel")
plt.title("Target Distribution (Diagnosis)")
plt.show()

# 4. Visualize feature distributions (numerical columns only)
data.iloc[:, 2:].hist(figsize=(20, 15), bins=20)
plt.tight_layout()
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = data.iloc[:, 2:].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature Transformation
# 1. Normalize continuous features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.iloc[:, 2:])
data.iloc[:, 2:] = scaled_features

# 2. Encode target (diagnosis)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Display the processed data
print("\nTransformed Data (First 5 Rows):")
print(data.head())

# Save the transformed dataset
data.to_csv("wdbc_transformed.csv", index=False)
print("\nTransformed dataset saved as 'wdbc_transformed.csv'")
