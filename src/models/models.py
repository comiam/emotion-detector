from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_available_models():
    return [
        MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
        LogisticRegression(random_state=42),
        SVC(kernel='rbf', C=1, random_state=42)
    ]
