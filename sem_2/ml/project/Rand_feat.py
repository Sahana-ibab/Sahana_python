import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the Diabetes dataset
def load_data():
    df = pd.read_csv("SMILES_only_filtered.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    return X, y

# Split the dataset into training and testing sets
def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X, y = load_data()

    rf = RandomForestRegressor()
    rf.fit(X, y)

    importances = rf.feature_importances_
    rf_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    rf_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(rf_df.head(20))

    rf_df['Cumulative'] = rf_df['Importance'].cumsum()
    selected = rf_df[rf_df['Cumulative'] <= 0.90]
    print(f"Selected {len(selected)} features covering 90% importance")


if __name__ == '__main__':
    main()
