import ast
import logging

import numpy as np

from models.metrics import calculate_metrics
from db.database import save_best_model, connect_to_database, load_trained_models, load_dataset_data, truncate_models, \
    count_best_models


def test_and_select_best_model(connection, models, X_test, y_test):
    model_data = {}

    for i, model_pair in enumerate(models):
        model_id, model = model_pair
        logging.warning(f'Testing model: id={model_id}, name={model}')

        y_pred = model.predict(X_test)
        f1, precision, recall, accuracy = calculate_metrics(y_test, y_pred)

        logging.warning(f'Metrics for model id={model_id}, name={model}:\n'
                        f'f1={f1}, precision={precision}, recall={recall}, accuracy={accuracy}\n')

        model_metrics = {}
        model_data[model_id] = model_metrics

        model_metrics['precision'] = precision
        model_metrics['recall'] = recall
        model_metrics['accuracy'] = accuracy
        model_metrics['f1_score'] = f1

    total_best_models = count_best_models(connection)
    top_1, top_2 = sorted(model_data.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:2]

    save_best_model(connection, top_1[0], top_1[1])
    logging.warning(f"Best model chosen and saved successfully. "
                    f"Best model id: {top_1[0]}.\n")

    if total_best_models == 0:
        save_best_model(connection, top_2[0], top_2[1])
        logging.warning(f"Additionally saved successfully best model model by metrics. "
                        f"Second best model id: {top_2[0]}.\n")


def test_models():
    connection = connect_to_database()

    logging.warning(f'Loading testing data...')
    df = load_dataset_data(connection, 1)

    logging.warning(f'Fetched {len(df)} testing data rows.')

    X = np.array(df['embedding'].apply(ast.literal_eval).tolist())
    y = df['sentiment'].to_numpy()

    models = load_trained_models(connection)
    test_and_select_best_model(connection, models, X, y)
    truncate_models(connection)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    test_models()
