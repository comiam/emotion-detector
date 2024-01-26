from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def calculate_metrics(y_true, y_pred):
    f1_new = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return f1_new, precision, recall, accuracy
