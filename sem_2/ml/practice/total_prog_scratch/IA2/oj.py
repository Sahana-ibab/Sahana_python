import pandas as pd
import numpy as np
from ISLP import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# --------------------------
# Preprocessing Function
# --------------------------
def preprocess_data(df, target_col='Purchase', train_size=1000, random_state=42):
    # Split into train and test
    train_df, test_df = train_test_split(df, train_size=train_size, stratify=df[target_col], random_state=random_state)

    # Encode target
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])  # Only transform

    # One-hot encode features
    X_train = pd.get_dummies(train_df.drop(columns=[target_col]), drop_first=True)
    X_test = pd.get_dummies(test_df.drop(columns=[target_col]), drop_first=True)

    # Align test with train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, le

# --------------------------
# Main Pipeline
# --------------------------
def main():
    # Load dataset
    df = load_data('OJ')

    # Preprocess
    X_train, X_test, y_train, y_test, le = preprocess_data(df)

    # (b) LinearSVC with C=0.01
    svc_linear = LinearSVC(C=0.01, max_iter=10000)
    svc_linear.fit(X_train, y_train)
    acc_train_linear = accuracy_score(y_train, svc_linear.predict(X_train))
    acc_test_linear = accuracy_score(y_test, svc_linear.predict(X_test))
    print(f"(b) Linear SVC (C=0.01): Train Acc = {acc_train_linear:.3f}, Test Acc = {acc_test_linear:.3f}")

    # (c) GridSearchCV for best C
    param_grid = {'C': np.linspace(0.01, 10, 50)}
    grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']
    print(f"(c) Best C from GridSearchCV: {best_C}")

    # (d) LinearSVC with best C
    svc_best = LinearSVC(C=best_C, max_iter=10000)
    svc_best.fit(X_train, y_train)
    acc_train_best = accuracy_score(y_train, svc_best.predict(X_train))
    acc_test_best = accuracy_score(y_test, svc_best.predict(X_test))
    print(f"(d) Linear SVC (C={best_C}): Train Acc = {acc_train_best:.3f}, Test Acc = {acc_test_best:.3f}")

    # (e) RBF Kernel SVC
    svc_rbf = SVC(kernel='rbf')  # Default gamma
    svc_rbf.fit(X_train, y_train)
    acc_train_rbf = accuracy_score(y_train, svc_rbf.predict(X_train))
    acc_test_rbf = accuracy_score(y_test, svc_rbf.predict(X_test))
    print(f"(e) RBF SVM: Train Acc = {acc_train_rbf:.3f}, Test Acc = {acc_test_rbf:.3f}")

    # Summary
    print("\nSummary of Results:")
    print(f"Linear SVC (C=0.01): Train={acc_train_linear:.3f}, Test={acc_test_linear:.3f}")
    print(f"Linear SVC (best C={best_C}): Train={acc_train_best:.3f}, Test={acc_test_best:.3f}")
    print(f"SVM with RBF kernel: Train={acc_train_rbf:.3f}, Test={acc_test_rbf:.3f}")

if __name__ == "__main__":
    main()
