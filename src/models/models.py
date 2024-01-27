import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

allowed_models_to_load = ['LogisticRegression', 'MLPClassifier']


def serialize_model(model):
    return pickle.dumps(model)


def deserialize_model(bin_weights):
    return pickle.loads(bytes(bin_weights))


def get_new_models():
    models = []

    for model_type in allowed_models_to_load:
        model_params = {}
        model = None

        if model_type == 'MLPClassifier':
            layer_sizes = tuple(np.random.randint(50, 200, np.random.randint(3, 8)))
            model_params['hidden_layer_sizes'] = ((100,)
                                                  + tuple(sorted(layer_sizes, reverse=True))
                                                  + (1,))
            model_params['activation'] = np.random.choice(['relu', 'tanh', 'logistic'])
            model_params['alpha'] = 10 ** np.random.uniform(-5, -0.5)
            model_params['solver'] = np.random.choice(['sgd', 'lbfgs', 'adam'])
            model_params['learning_rate_init'] = 10 ** np.random.uniform(-3, -1)
            model_params['max_iter'] = np.random.randint(1000, 10000)
            model = MLPClassifier(**model_params)

        elif model_type == 'LogisticRegression':
            model_params['C'] = np.random.uniform(0, 3)
            model_params['solver'] = np.random.choice(['liblinear', 'lbfgs'])
            model_params['penalty'] = np.random.choice(['l1', 'l2']) if model_params['solver'] == 'liblinear' else 'l2'
            model_params['max_iter'] = np.random.randint(1000, 10000)
            model = LogisticRegression(**model_params)

        models.append(model)

    return models
