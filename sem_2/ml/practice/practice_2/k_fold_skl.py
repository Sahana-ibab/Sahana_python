import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/data.csv')
    data=data.drop(['id','Unnamed: 32'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
    X=data.drop(columns=['diagnosis'])
    y=data['diagnosis']
    return X,y
def main():
    X,y=load_data()
    kf=KFold(n_splits=10,shuffle=True,random_state=369)
    accuracies=[]
    for train_index,test_index in kf.split(X):
        X_train,X_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        model=LogisticRegression(max_iter=10000)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        accuracies.append(accuracy)
        print(f"Fold accuracy:{accuracy}")
    accuracy_array=np.array(accuracies)
    print(f"Mean accuracy: {accuracy_array.mean()}")
    print(f"Std dev:{accuracy_array.std()}")
main()