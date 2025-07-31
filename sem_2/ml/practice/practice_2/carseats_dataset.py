# Basic information about the dataset
print("Dataset Info:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check unique values in categorical columns
categorical_columns = ['ShelveLoc', 'Urban', 'US']
for col in categorical_columns:
    print(f"\nUnique values in {col}: {df[col].unique()}")
# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handling missing values (example: replacing missing values with the column mean)
df.fillna(df.mean(), inplace=True)
import matplotlib.pyplot as plt

# List of numerical features
numerical_features = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']

# Plot histograms
for col in numerical_features:
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# One-hot encoding for nominal variables
df = pd.get_dummies(df, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)
# Ordinal encoding for ShelveLoc
shelveloc_mapping = {'Bad': 0, 'Medium': 1, 'Good': 2}
df['ShelveLoc'] = df['ShelveLoc'].map(shelveloc_mapping)
from sklearn.preprocessing import StandardScaler

# Scaling numerical features
scaler = StandardScaler()
numerical_features = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
