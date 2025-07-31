import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "/mnt/data/Ch12Ex13.csv"
data = pd.read_csv(file_path)

# Display basic information
print("Dataset Info:")
data.info()
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Visualizing distributions
plt.figure(figsize=(12, 6))
sns.histplot(data, kde=True)
plt.title("Feature Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Feature Engineering
# Handling categorical features with OneHotEncoding (if any exist)
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    data = data.drop(columns=categorical_features)
    data = pd.concat([data, encoded_df], axis=1)

# Standardizing numerical features
scaler = StandardScaler()
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Display transformed data
print("\nProcessed Data Sample:")
print(data.head())
