import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    data=pd.read_csv("/home/ibab/data.csv")
    X=data[['id', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
    # y=data['diagnosis']
    le = LabelEncoder()
    data['diagnosis_c'] = le.fit_transform(data['diagnosis'])
    y = data['diagnosis_c']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    scaler= StandardScaler()
    scaler= scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("------training------\n")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(y_pred)
    score= accuracy_score(y_test, y_pred)
    print("Accuracy score: ",score)
    print("coefficients:", model.coef_)
    print("Intercept: ", model.intercept_)
if __name__ == '__main__':
    main()
