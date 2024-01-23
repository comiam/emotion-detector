import logging

import numpy as np

from models.models import get_available_models
from util.database import connect_to_database, load_train_dataset, save_model
import ast


def train_models():
    """
    Обучает модели на данных из train_test_splits.
    """
    connection = connect_to_database()

    logging.warning(f'Loading training data...')
    df = load_train_dataset(connection)

    logging.warning(f'Fetched {len(df)} training data.')

    max_dataset_id = df['dataset_id'].max()
    X = np.array(df['embedding'].apply(ast.literal_eval).tolist())
    y = df['sentiment'].to_numpy()

    logging.warning(f'Begin training model...')
    for model in get_available_models():
        logging.warning(f'Training model: {model}...')
        model.fit(X, y)

        logging.warning(f'Saving model {model} to DB...')
        save_model(connection, model, max_dataset_id)

    logging.warning(f'Training complete.')


if __name__ == "__main__":
    train_models()
