import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    print("\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def encode_categorical(df):
    """Encode categorical variables using OneHotEncoder"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    oh = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
    encoded_data = oh.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=oh.get_feature_names_out(categorical_cols))

    # Combine the encoded features with the original dataframe (drop original categorical columns)
    df_encoded = df.drop(categorical_cols, axis=1).join(encoded_df)
    return df_encoded


def plot_hist(df, target_col='A1BG'):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_col], kde=True, bins=30)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.show()

def summary_statistics(df):
    print("\n=== Summary Statistics ===")
    print(df.describe())


def EDA(df):

    # EDA: Plot histogram for the target variable
    plot_hist(df, target_col='A1BG')

    # Display summary statistics
    summary_statistics(df)


def main():
    # Load the dataset
    df = load_data("de_train_with_embeddings.csv")

    # Apply one-hot encoding for categorical columns
    df_encoded = encode_categorical(df)

    # Perform EDA on the encoded dataset
    EDA(df_encoded)


if __name__ == '__main__':
    main()
