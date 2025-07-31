# Compute SONAR classification results with and without data pre-processing (data
# normalization). Perform data pre-processing with your implementation and with
# scikit-learn methods and compare the results.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from norm import data_normalization

# without scaling

def Without_scaling():
    data=pd.read_csv("/home/ibab/Sonar.csv")
    # print(data.columns)
    X= data[['Freq_1', 'Freq_2', 'Freq_3', 'Freq_4', 'Freq_5', 'Freq_6', 'Freq_7',
       'Freq_8', 'Freq_9', 'Freq_10', 'Freq_11', 'Freq_12', 'Freq_13',
       'Freq_14', 'Freq_15', 'Freq_16', 'Freq_17', 'Freq_18', 'Freq_19',
       'Freq_20', 'Freq_21', 'Freq_22', 'Freq_23', 'Freq_24', 'Freq_25',
       'Freq_26', 'Freq_27', 'Freq_28', 'Freq_29', 'Freq_30', 'Freq_31',
       'Freq_32', 'Freq_33', 'Freq_34', 'Freq_35', 'Freq_36', 'Freq_37',
       'Freq_38', 'Freq_39', 'Freq_40', 'Freq_41', 'Freq_42', 'Freq_43',
       'Freq_44', 'Freq_45', 'Freq_46', 'Freq_47', 'Freq_48', 'Freq_49',
       'Freq_50', 'Freq_51', 'Freq_52', 'Freq_53', 'Freq_54', 'Freq_55',
       'Freq_56', 'Freq_57', 'Freq_58', 'Freq_59', 'Freq_60']]

    le=LabelEncoder()
    data['Label']=le.fit_transform(data['Label'])
    y=data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    print("--------training--------")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(y_pred)
    accuracy=accuracy_score(y_test, y_pred)
    return accuracy


def With_scaling():
    data=pd.read_csv("/home/ibab/Sonar.csv")
    # print(data.columns)
    X= data[['Freq_1', 'Freq_2', 'Freq_3', 'Freq_4', 'Freq_5', 'Freq_6', 'Freq_7',
       'Freq_8', 'Freq_9', 'Freq_10', 'Freq_11', 'Freq_12', 'Freq_13',
       'Freq_14', 'Freq_15', 'Freq_16', 'Freq_17', 'Freq_18', 'Freq_19',
       'Freq_20', 'Freq_21', 'Freq_22', 'Freq_23', 'Freq_24', 'Freq_25',
       'Freq_26', 'Freq_27', 'Freq_28', 'Freq_29', 'Freq_30', 'Freq_31',
       'Freq_32', 'Freq_33', 'Freq_34', 'Freq_35', 'Freq_36', 'Freq_37',
       'Freq_38', 'Freq_39', 'Freq_40', 'Freq_41', 'Freq_42', 'Freq_43',
       'Freq_44', 'Freq_45', 'Freq_46', 'Freq_47', 'Freq_48', 'Freq_49',
       'Freq_50', 'Freq_51', 'Freq_52', 'Freq_53', 'Freq_54', 'Freq_55',
       'Freq_56', 'Freq_57', 'Freq_58', 'Freq_59', 'Freq_60']]

    le=LabelEncoder()
    data['Label']=le.fit_transform(data['Label'])
    y=data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    scaler = StandardScaler()
    # scaler = scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("--------training--------")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # print(y_pred)
    accuracy=accuracy_score(y_test, y_pred)
    return accuracy

def With_scaling_Own_function():
    data=pd.read_csv("/home/ibab/Sonar.csv")
    # print(data.columns)
    X= data[['Freq_1', 'Freq_2', 'Freq_3', 'Freq_4', 'Freq_5', 'Freq_6', 'Freq_7',
       'Freq_8', 'Freq_9', 'Freq_10', 'Freq_11', 'Freq_12', 'Freq_13',
       'Freq_14', 'Freq_15', 'Freq_16', 'Freq_17', 'Freq_18', 'Freq_19',
       'Freq_20', 'Freq_21', 'Freq_22', 'Freq_23', 'Freq_24', 'Freq_25',
       'Freq_26', 'Freq_27', 'Freq_28', 'Freq_29', 'Freq_30', 'Freq_31',
       'Freq_32', 'Freq_33', 'Freq_34', 'Freq_35', 'Freq_36', 'Freq_37',
       'Freq_38', 'Freq_39', 'Freq_40', 'Freq_41', 'Freq_42', 'Freq_43',
       'Freq_44', 'Freq_45', 'Freq_46', 'Freq_47', 'Freq_48', 'Freq_49',
       'Freq_50', 'Freq_51', 'Freq_52', 'Freq_53', 'Freq_54', 'Freq_55',
       'Freq_56', 'Freq_57', 'Freq_58', 'Freq_59', 'Freq_60']]

    le=LabelEncoder()
    data['Label']=le.fit_transform(data['Label'])
    y=data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    X_train_scaled = data_normalization(X_train)
    X_test_scaled = data_normalization(X_test)
    print("--------training--------")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # print(y_pred)
    accuracy=accuracy_score(y_test, y_pred)
    return accuracy

def main():
    print("Accuracy score with scaling: ",With_scaling())
    print("Accuracy score without scaling: ",Without_scaling())
    print(With_scaling_Own_function())
if __name__ == '__main__':
    main()

