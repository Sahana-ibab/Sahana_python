
def get_conf_matrix_elements(cm):
    TP = cm[0][0]
    TN = cm[0][1]
    FP = cm[1][0]
    FN = cm[1][1]
    T = TN + TP + FN + FP
    return TP, TN, FP, FN, T


def Accuracy(cm):
    TP, TN, FP, FN, T = get_conf_matrix_elements(cm)
    return ( TP + TN )/T

def Precision(cm):
    TP, TN, FP, FN, T = get_conf_matrix_elements(cm)
    return TP/( TP + FP ) if (TP + FP) != 0 else 0

def Sensitivity(cm):
    TP, TN, FP, FN, T = get_conf_matrix_elements(cm)
    return TP/( TP + FN ) if (TP + FN) != 0 else 0

def Specificity(cm):
    TP, TN, FP, FN, T = get_conf_matrix_elements(cm)
    return TN/( TN + FN ) if (TN + FN) != 0 else 0

def F1_score(cm):
    precision = Precision(cm)
    sensitivity = Precision(cm)
    return 2 * (precision * sensitivity)/(precision + sensitivity) if ( precision + sensitivity ) != 0 else 0

def eval_metrix(cm):
    print("Accuracy score: ", Accuracy(cm))
    print("Precision score: ", Precision(cm))
    print("Sensitivity score: ", Sensitivity(cm))
    print("Specificity score: ", Specificity(cm))
    print("F1 score: ", F1_score(cm))



