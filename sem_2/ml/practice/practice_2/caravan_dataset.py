import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "/mnt/data/Caravan.csv"
data = pd.read_csv(file_path)

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Display dataset info
print("\nDataset Info:")
print(data.info())

# Summary statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handling categorical features
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns:", categorical_cols)

# Encoding categorical columns using Label Encoding (if applicable)
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Histogram of numerical features
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Scaling (Normalization)
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("\nNormalized Data Sample:")
print(data.head())

# Save processed dataset
processed_file = "/mnt/data/Caravan_Processed.csv"
data.to_csv(processed_file, index=False)
print(f"\nProcessed data saved to: {processed_file}")
