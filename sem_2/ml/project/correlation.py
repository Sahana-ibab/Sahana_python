import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# Load Dataset
def load_data():
    df = pd.read_csv("SMILES_filtered.csv")
    X = df.iloc[:, :5]  # First 5 columns as features
    y = df.iloc[:, 6]   # 6th column as target variable
    return X, y

# Main function
def main():
    X, y = load_data()

    # Encode categorical features
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    X_encoded_df = pd.DataFrame(X_encoded, columns=X.columns)  # Convert to DataFrame

    # Compute correlation
    correlation = X_encoded_df.corrwith(y, method='pearson')  # Pearson Correlation
    print("Feature Correlations with Target (y):\n", correlation)

    # Create subplots
    num_features = X.shape[1]
    fig, axes = plt.subplots(1, num_features, figsize=(20, 5))  # 1 row, n columns

    # Plot each feature separately
    for i, col in enumerate(X.columns):
        axes[i].scatter(X_encoded_df[col], y, alpha=0.5)
        axes[i].set_title(f"{col} vs Target\nCorrelation: {correlation[col]:.2f}")
        axes[i].set_xlabel("Encoded Feature Values")
        axes[i].set_ylabel("Target Variable (y)")

    plt.tight_layout()  # Adjust layout
    plt.show()

if __name__ == '__main__':
    main()
