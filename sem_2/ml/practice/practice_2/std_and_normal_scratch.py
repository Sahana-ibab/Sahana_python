import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
# def load_data():
#     data=pd.read_csv('/home/ibab/Downloads/data.csv')
#     data=data.drop(['id','Unnamed: 32'],axis=1)
#     data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
#     X=data.drop(columns=['diagnosis'])
#     y=data['diagnosis']
#     return X,y
# def normalization(X):
#     X_min=X.min()
#     X_max=X.max()
#     X_normalized=(X-X_min)/(X_max-X_min)
#     return X_normalized
# def main():
#     X,y=load_data()
#     X_normalized=normalization(X)
#     kf=KFold(n_splits=10,shuffle=True,random_state=369)
#     accuracies=[]
#     for fold_index, (train_index,test_index) in enumerate(kf.split(X_normalized),start=1):
#         X_train,X_test=X.iloc[train_index],X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         # initializing the model eg linear vector
#         clf = LogisticRegression(max_iter=5000)
#         # fitting the model
#         clf.fit(X_train, y_train)
#         scores = clf.score(X_test, y_test)
#         accuracies.append(scores)
#         print(f"Fold Accuracy {fold_index}: {scores:.3f}")
#     accuracy_array=np.array(accuracies)
#     print(f"Mean accuracy:{accuracy_array.mean()}")
#     print(f"Std dev:{accuracy_array.std()}")
# main()
#STANDARDIZATION
