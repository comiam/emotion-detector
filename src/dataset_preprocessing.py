import logging
import time

import sys

from db.database import connect_to_database, fetch_diff_between_datasets, save_preprocessed_data
from util.embeddings import calculate_embedding


def get_embeddings(batch):
    """
    Получаем эмбэддинги текстовых данных.
    """
    return [calculate_embedding(el) for el in batch]


def preprocess_batch(batch):
    """
    Обработка батча новых данных.
    """
    text_data = batch['comment'].tolist()
    emb_batch = get_embeddings(text_data)

    return batch['sentiment'], emb_batch


def preprocess_dataset(timeout_min):
    connection = connect_to_database()

    batch_size = 256
    count, dataset_generator = fetch_diff_between_datasets(connection, batch_size)
    logging.warning(f"Rows fetched. Processing in stream. "
                    f"Total rows: {count}.")

    try:
        # Проход по батчам и предобработка
        start_time = time.time()
        i = 0
        for batch in dataset_generator:
            if batch.empty:
                break

            sentiment_batch, embedding_batch = preprocess_batch(batch)

            # Сохраняем предобработанные данные
            save_preprocessed_data(connection, embedding_batch,
                                   sentiment_batch, batch['version'])

            time_passed = time.time() - start_time
            i += len(batch)

            logging.warning(f'Processed rows: {i + batch_size}. '
                            f'Time spent: {time_passed}')
            if time_passed > int(timeout_min) * 60:
                logging.warning('Finish by timeout')
                break
    finally:
        # Закрытие соединения
        connection.close()


if __name__ == "__main__":
    timeout = 1
    if len(sys.argv) > 1:
        timeout = int(sys.argv[1])
    preprocess_dataset(timeout)
