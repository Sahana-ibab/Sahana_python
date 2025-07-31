# importing required modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


# Function: To Load-Data:
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Function: To split the df into given proportions:
def split_data(df, test_size=0.3):
    X = df.drop('A1BG', axis=1)
    y = df['A1BG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


# Function: Data preprocessing :
def preprocessing(df_train, df_test):
    # One-hot encoding (fit on train, transform both)
    sm_encoder = OneHotEncoder(drop='first', sparse_output=False).fit(df_train[['sm_name']])
    cell_encoder = OneHotEncoder(drop='first', sparse_output=False).fit(df_train[['cell_type']])

    sm_train = sm_encoder.transform(df_train[['sm_name']])
    sm_test = sm_encoder.transform(df_test[['sm_name']])
    cell_train = cell_encoder.transform(df_train[['cell_type']])
    cell_test = cell_encoder.transform(df_test[['cell_type']])

    # Extract numeric columns
    df_train_numerical = df_train.drop(['sm_name', 'cell_type', 'A1BG'], axis=1)
    df_test_numerical = df_test.drop(['sm_name', 'cell_type', 'A1BG'], axis=1)

    # Scale numeric columns (fit only on train)
    scaler = StandardScaler().fit(df_train_numerical)
    scaled_train = scaler.transform(df_train_numerical)
    scaled_test = scaler.transform(df_test_numerical)

    # Combine features
    X_train = np.hstack([scaled_train, sm_train, cell_train])
    X_test = np.hstack([scaled_test, sm_test, cell_test])
    y_train, y_test = df_train['A1BG'], df_test['A1BG']

    return X_train, X_test, y_train, y_test


# Function: k-fold cross-validation with LinearRegression:
def k_fold_cross_validation(df, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    plt.figure(figsize=(10, 8))

    for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
        print(f"\nFold {fold}/{n_splits}")
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocessing(df_train, df_test)

        # Training model
        print("\n---------Training-----------\n")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate metrics
        metrics = {
            'Fold': fold,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred)
        }
        print("R2-score: ", metrics['R2'])
        print("MSE: ", metrics['MSE'])
        results.append(metrics)

        # Plot predictions
        plt.scatter(y_test, y_pred, alpha=0.5, label=f'Fold {fold}')

    # Plotting all folds
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("True A1BG")
    plt.ylabel("Predicted A1BG")
    plt.title("Linear Regression Scatter (All Folds)")
    plt.legend()
    plt.show()

    # Display results
    results_df = pd.DataFrame(results)

    # Mean metrics
    print("\nMean Metrics: ")
    print(results_df.drop(columns='Fold').mean(numeric_only=True))


# Main function
def main():
    df = load_data("de_train_with_embeddings.csv")
    k_fold_cross_validation(df, n_splits=10)

if __name__ == "__main__":
    main()
