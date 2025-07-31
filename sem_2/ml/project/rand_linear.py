from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Loading Dataset:
def load_data():
    df = pd.read_csv("SMILES_only_filtered.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    # print(y)
    # print(df.describe())
    return X, y


# Select features based on Random Forest Regressor feature importances:
def select_important_features(X, y, threshold=0.9):
    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances and sort them
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Calculate cumulative importance
    cumulative_importance = feature_importance_df['Importance'].cumsum()

    # Select features based on cumulative importance threshold
    feature_importance_df["Cumulative"] = cumulative_importance
    selected_features = feature_importance_df[feature_importance_df["Cumulative"] <= threshold]

    print(f"Selected {len(selected_features)} features covering {threshold * 100}% of total importance.")

    # Return X with only selected features
    selected_columns = selected_features['Feature'].values
    X_selected = X[selected_columns]
    return X_selected


# Split dataset:
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("train", X_train.shape)
    print("test", X_test.shape)
    return X_train, X_test, y_train, y_test


def main():
    X, y = load_data()

    # Feature selection using Random Forest Regressor
    X_selected = select_important_features(X, y, threshold = 0.9)

    # Cross-validation setup
    kf = KFold(n_splits = 10, shuffle=True, random_state = 42)
    i = 1
    R2_scores = []

    for train_index, test_index in kf.split(X_selected):
        print("\nFold: ", i)
        X_train = X_selected.iloc[train_index]
        X_test = X_selected.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Scaling the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("train", X_train_scaled.shape)
        print("test", X_test_scaled.shape)

        # Make predictions and calculate R2 score
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 score for fold {i}: {r2}")
        i += 1
        R2_scores.append(r2)

    # Calculate mean and standard deviation of R2 scores
    mean_r2 = np.mean(R2_scores)
    std_r2 = np.std(R2_scores)
    print("Mean R2 score: ", mean_r2)
    print("Standard deviation: ", std_r2)


if __name__ == '__main__':
    main()
