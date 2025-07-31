import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "/mnt/data/Admission_Predict_Ver1.1.csv"
data = pd.read_csv(file_path)

# Display basic info
print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Summary:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Histogram of numerical features
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Encoding & Transformation
scaler = StandardScaler()
numerical_features = ['GRE Score', 'TOEFL Score', 'CGPA']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("\nNormalized Data Sample:")
print(data.head())

# Save transformed dataset
transformed_file = "/mnt/data/Admission_Predict_Processed.csv"
data.to_csv(transformed_file, index=False)
print(f"\nProcessed data saved to: {transformed_file}")
