import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("/mnt/data/Default.csv")

# Display basic information
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualizing target variable distribution
sns.countplot(x='default', data=df)
plt.title("Distribution of Target Variable")
plt.show()

# Checking correlation between numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Engineering
# Encoding categorical features (if any)
if df.select_dtypes(include=['object']).shape[1] > 0:
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Scaling numerical features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nProcessed Data (First 5 Rows):")
print(df.head())
