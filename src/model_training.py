import logging

import numpy as np

from models.models import get_new_models, deserialize_model
from db.database import connect_to_database, load_dataset_data, save_model, get_trained_models_last_dataset_versions, \
    load_best_models
import ast


def retrain_and_update_models(connection, X_train_new, y_train_new):
    logging.warning(f"Begin retraining best models.")
    best_model_info = load_best_models(connection)

    if len(best_model_info) == 0:
        logging.warning(f"Nothing to retrain! Go out!")
        return

    # Дообучение моделей на новых данных
    for model_info in best_model_info:
        model_id, model_name, current_f1, weights = model_info

        logging.warning(f"Retraining model: id {model_id}, model_name: {model_name}")
        model = deserialize_model(weights)
        model.fit(X_train_new, y_train_new)

        save_model(connection, model, np.max(X_train_new))
        logging.warning(f"Model: id {model_id}, model_name: {model_name} retrained and saved new copy.")

    connection.commit()


def train_models():
    connection = connect_to_database()

    logging.warning(f'Loading training data...')
    df = load_dataset_data(connection, 0)

    logging.warning(f'Fetched {len(df)} training data rows.')

    max_dataset_id = df['dataset_id'].max()

    X = np.array(df['embedding'].apply(ast.literal_eval).tolist())
    y = df['sentiment'].to_numpy()

    logging.warning(f'Begin training model...')
    for model in get_new_models():
        logging.warning(f'Training model: {model}...')
        model.fit(X, y)

        logging.warning(f'Saving model {model} to DB...')
        save_model(connection, model, max_dataset_id)

    connection.commit()
    logging.warning(f'New models training complete.')

    trained_dataset_ids = get_trained_models_last_dataset_versions(connection)

    if max_dataset_id in trained_dataset_ids or max_dataset_id < min(trained_dataset_ids, default=0):
        logging.warning(f'No new data for retraining, last dataset mark is {max_dataset_id}.')
        connection.close()
        return

    retrain_and_update_models(connection, X, y)
    connection.close()


if __name__ == "__main__":
    train_models()
