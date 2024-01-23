from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC


def get_available_models():
    return [
        MLPRegressor(hidden_layer_sizes=(100, 200, 200, 150, 100, 50, 1),
                     max_iter=100000,
                     solver='adam',
                     random_state=42),
        LogisticRegression(random_state=42, max_iter=100000),
        SVC(kernel='rbf', C=1, random_state=42, max_iter=100000)
    ]
