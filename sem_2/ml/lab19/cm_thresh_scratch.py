
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sem_2.ml.lab19.Evaluation_metrix_functions import eval_metrix
from sklearn.metrics import roc_curve, auc

def load_data():
    df = pd.read_csv("/home/ibab/datasets/heart.csv")
    X = df.drop(columns=['output'])
    y = df['output']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def main():

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--------training--------")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cm_list = {thresh: [] for thresh in thresholds}

    # inside the fold loop:
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        print("\ny_pred_thresh: ",y_pred_thresh)
        acc = accuracy_score(y_test, y_pred_thresh)
        cm = confusion_matrix(y_test, y_pred_thresh)
        print("\nThreshold: ",thresh)
        eval_metrix(cm)
        cm_list[thresh].append((acc, cm))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()