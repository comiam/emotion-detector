import logging
import torch
import time
import numpy as np
import sys

from transformers import AutoTokenizer, AutoModel

from util.database import connect_to_database, fetch_diff_between_datasets, save_preprocessed_data

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
MAX_SEQUENCE_LENGTH = 2048
model.max_seq_length = MAX_SEQUENCE_LENGTH


def get_embeddings(batch):
    """
    Получаем эмбэддинги текстовых данных.
    """
    tokens = tokenizer(batch, max_length=MAX_SEQUENCE_LENGTH,
                       padding="max_length",
                       truncation=True)["input_ids"]
    tokens = torch.from_numpy(np.array(tokens))
    with torch.no_grad():
        embeddings = model(tokens)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


def preprocess_batch(batch):
    """
    Обработка батча новых данных.
    """
    text_data = batch['comment'].tolist()

    sentiment_mapping = {0: 0, 1: 0.5, 2: 1}
    sentiment_batch = batch['sentiment'].map(sentiment_mapping)

    emb_batch = get_embeddings(text_data)

    return sentiment_batch, emb_batch


def preprocess_dataset(timeout_min):
    connection = connect_to_database()
    dataset_df = fetch_diff_between_datasets(connection)

    try:
        # Проход по батчам и предобработка
        batch_size = 32
        start_time = time.time()
        for i in range(0, len(dataset_df), batch_size):
            batch = dataset_df.iloc[i:i + batch_size]
            sentiment_batch, embedding_batch = preprocess_batch(batch)

            # Сохраняем предобработанные данные
            save_preprocessed_data(connection, embedding_batch, sentiment_batch, batch['version'])

            time_passed = time.time() - start_time
            logging.info(f'Processed rows: {batch_size} of {dataset_df.shape[0]}. Time spent: {time_passed}')
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
