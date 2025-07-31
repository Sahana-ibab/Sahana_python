import matplotlib.pyplot as plt
import pandas as pd

def plot(X_vector, y_vector,y_pred_gd, y_pred_ne, y_pred_sklearn, Xs):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_vector, y_vector,color = 'blue', label = 'Actual Data')
    plt.plot(X_vector, y_pred_gd, color = "black", label = "Gradient Descent Line", linewidth = 5)
    plt.plot(X_vector, y_pred_ne, color = 'red', label = 'Normal Equation Line', linewidth=2)
    plt.plot( Xs, y_pred_sklearn, color='green', label = 'Scikit-learn Line', linewidth= 2)
    plt.xlabel("age")
    plt.ylabel('Target (Scaled Disease Score)')
    plt.title(f'Regression Line Comparison: age')
    plt.legend()
    plt.show()

import numpy as np
def main():
    y_pred_gd=[0.26734763, - 0.81929112,  1.35398638, - 0.7804826,   0.96590111, - 1.16856786,0.73304995,  0.810667, - 0.43120585, - 0.08192911, - 1.09095081,  0.38377321,0.38377321,0.53900732, -0.43120585,-1.09095081,-1.01333376,1.47041196]
    y_pred_ne=[ 0.26734763, -0.81929112,  1.35398638, -0.7804826, 0.96590111, -1.16856786, 0.73304995, 0.810667, -0.43120585, -0.08192911, -1.09095081, 0.38377321, 0.38377321, 0.53900732, -0.43120585, -1.09095081, -1.01333376, 1.47041196]
    y_pred_sklearn=[765.30955962,751.92066812, 783.16141496, 785.39289688, 760.84659579,856.80031823, 879.1151374, 747.45770429, 894.73551082, 861.26328206,816.63364372,876.88365548, 807.70771605,742.99474045,827.7910533,863.49476398, 805.47623413, 805.47623413]
    Xs=[20, 14, 28, 29, 18, 61, 71, 12, 78, 63, 43, 70, 39, 10, 48, 64, 38, 38]

    y_pred_gd=np.array(y_pred_gd)
    y_pred_ne=np.array(y_pred_ne)
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')
    # print(df.describe())
    # X_vector = df[["age", "BMI", "BP", "blood_sugar", "Gender"]]
    X = df[["age"]]
    y = df["disease_score"]
    split_index = int(0.7 * len(X))
    X_train, X_vector = X[:split_index], X[split_index:]
    y_train, y_vector = y[:split_index], y[split_index:]
    X_vectorl=X_vector
    y_vectorl=y_vector
    plot(X_vectorl, y_vectorl, y_pred_gd, y_pred_ne, y_pred_sklearn, Xs)
    print(X_vector)

if __name__ == '__main__':
    main()