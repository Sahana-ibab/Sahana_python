#  importing required modules:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from xgboost import XGBRegressor


# Function: To Load-Data:
#  - Parameter: path of the file
#  - Output: Loaded data in the form of Pandas DataFrame

def load_data(file_path):
    df = pd.read_csv(file_path)
    # print(df.columns)
    return df

# Function: To split the df into given proportions :
#  - Parameter: df, test_size
#  - Returns: training and test sets

def split_data(df, test_size=0.3):

    X = df.drop('A1BG', axis=1)
    print(X.shape)
    y = df['A1BG']
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Function: Scaling and PCA:
#  - Parameters: train-set and test-set( without first two col
#               (We need to keep these columns even if PCA
#               doesn't select them.), target) ,
#               n_components- ( percentage of variance captured )
#
#  - Returns: The training and testing data are first scaled,
#             and then PCA is applied to reduce their
#             dimensionality, dfs are returned.

def apply_pca(df_train_numerical , df_test_numerical, n_components=0.95):

    # Scale the numeric columns (fit only on training set)
    scaler = StandardScaler().fit(df_train_numerical)

    # Apply PCA (fit only on training set )
    pca = PCA(n_components=n_components).fit(scaler.transform(df_train_numerical))

    # printing no of features selected by PCA:
    print(f"Number of components selected after PCA: {pca.n_components_}")

    # # print the variance captured form each component:
    # print(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")

    # Transform train/test datasets:-
    scaled_train = scaler.transform(df_train_numerical)
    scaled_test = scaler.transform(df_test_numerical)

    scaled_train_pca = pca.transform(scaled_train)
    scaled_test_pca = pca.transform(scaled_test)
    # print(scaled_train_pca.shape)
    return scaled_train_pca, scaled_test_pca


# Function: Data preprocessing:
#  - Parameters: training and test DataFrames
#  - Returns: does OneHotEncoding(for first two categorical cols),
#             calls PCA function, finally returns X_train, X_test,
#             y_train, y_test DataFrames.

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

    # Apply PCA using the separate function
    scaled_train_pca, scaled_test_pca = apply_pca(df_train_numerical, df_test_numerical)

    # Combine features
    X_train = np.hstack([scaled_train_pca, sm_train, cell_train])
    X_test = np.hstack([scaled_test_pca, sm_test, cell_test])
    y_train, y_test = df_train['A1BG'], df_test['A1BG']
    # print(X_train.shape)
    # print(X_test.shape)
    return X_train, X_test, y_train, y_test


# Function: Hyperparameter tuning for XGBoost model
#  - Parameters: X_train, y_train (training features and labels)
#  - Uses: GridSearchCV to perform 3-fold cross-validation
#          on a predefined grid of XGBoost parameters
#  - Returns: dictionary of -best parameters (max_depth, learning_rate, n_estimators)

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200]
    }
    model = XGBRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_


# Function: k-fold cross-validation with XGBoost and PCA
#  - Parameters:
#       df: full DataFrame containing features and target
#       n_splits: number of folds to split the data (default = 10)
#  - Process:
#       - Splits the data into k folds using KFold
#       - For each fold:
#           - Preprocesses data (scaling, one-hot, PCA)
#           - Tunes hyperparameters using GridSearchCV
#           - Trains XGBoost with -best parameters
#           - Evaluates predictions (MSE, R², MAE, Explained Variance)
#           - Plots predicted vs true values of target at last
#       - After all folds:
#           - Displays a combined scatter plot
#           - Prints per-fold evaluation metrics
#           - Prints mean values of evaluation metrics across folds
#  - Returns: None (prints results and displays plot within the function itself)

def k_fold_cross_validation(df, n_splits=10):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    plt.figure(figsize=(10, 8))

    for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
        print(f"\nFold {fold}/{n_splits}")
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        # Preprocess data (avoid leakage)
        X_train, X_test, y_train, y_test = preprocessing(df_train, df_test)

        # Hyperparameter tuning
        best_params = tune_hyperparameters(X_train, y_train)
        print(f"Best params: {best_params}")

        # Training  model:
        print("\n---------Training-----------\n")
        model = XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate metrics: storing it as lists of dictionaries
        metrics = {
            'Fold': fold,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred)
        }
        print("R2-score: ", metrics['R2'])
        print("MSE: ", metrics['MSE'])
        results.append(metrics)

        # plot- Combining all 10-folds
        plt.scatter(y_test, y_pred, alpha=0.5, label=f'Fold {fold}')

    # Displaying overall plot(of 10-folds):
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("True A1BG")
    plt.ylabel("Predicted A1BG")
    plt.title("XGBoost Regression Scatter (All Folds)")
    plt.legend()
    plt.show()

    # Displaying results
    results_df = pd.DataFrame(results)
    # print("\nCross-Validation Results: ")
    # print(results_df)

    # Remove the 'Fold' column before calculating the mean
    results_df_without_fold = results_df.drop(columns='Fold')

    # Calculating and printing the mean metrics
    print("\nMean Metrics: ")
    print(results_df_without_fold.mean(numeric_only=True))


# Main function:
#  - calls load_data() function to load DataFrame from file
#  - Calls k_fold_cross_validation() to perform 10-fold CV
#    with preprocessing, PCA, hyperparameter tuning, and evaluation...

def main():
    # calling Function to - Load the dataset
    df = load_data("de_train_with_embeddings.csv")
    # print(df.columns)
    # Perform k-fold cross-validation
    k_fold_cross_validation(df, n_splits=10)


if __name__ == "__main__":
    main()