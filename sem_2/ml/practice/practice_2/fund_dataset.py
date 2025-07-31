import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "/mnt/data/Fund.csv"
data = pd.read_csv(file_path)

# Display basic info
print("Dataset Info:")
data.info()
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizing distributions
plt.figure(figsize=(12, 6))
sns.histplot(data, bins=30, kde=True)
plt.title("Feature Distributions")
plt.show()

# Check for correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Standardization (Scaling) of numerical features
scaler = StandardScaler()
numeric_features = data.select_dtypes(include=[np.number]).columns
data_scaled = data.copy()
data_scaled[numeric_features] = scaler.fit_transform(data[numeric_features])

print("\nScaled Data Sample:")
print(data_scaled.head())

# Check for highly correlated features (Threshold: 0.9)
corr_matrix = data_scaled.corr().abs()
high_corr_var = np.where(corr_matrix > 0.9)
high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

if high_corr_var:
    print("\nHighly Correlated Feature Pairs (>0.9):", high_corr_var)
else:
    print("\nNo highly correlated features found.")
