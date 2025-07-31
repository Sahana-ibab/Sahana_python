import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("/mnt/data/data.csv")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 32", "id"], inplace=True)

# Check for missing values
missing_values = df.isnull().sum()

# Summary statistics
summary_stats = df.describe()

# Visualizing the distribution of numeric features
plt.figure(figsize=(10, 6))
sns.histplot(df.drop(columns=['diagnosis']), bins=30, kde=True)
plt.title("Feature Distributions")
plt.show()

# Encode categorical variable
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# Scale numerical features
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Display processed dataset
print(df.head())
