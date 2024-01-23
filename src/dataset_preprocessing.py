import logging
import time

import nltk
import numpy as np
import sys

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize

from util.database import connect_to_database, fetch_diff_between_datasets, save_preprocessed_data

nltk.download('punkt')
model = Word2Vec.load("src/w2v_model.bin")


def calculate_embedding(text, model):
    # Токенизация текста
    tokens = word_tokenize(text)
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]

    if word_vectors:
        average_embedding = normalize([np.mean(word_vectors, axis=0)])
        return average_embedding
    else:
        return None


def get_embeddings(batch):
    """
    Получаем эмбэддинги текстовых данных.
    """
    return [calculate_embedding(el, model) for el in batch]


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

    if dataset_df.empty:
        connection.close()
        return

    try:
        # Проход по батчам и предобработка
        batch_size = 32
        start_time = time.time()
        for i in range(0, len(dataset_df), batch_size):
            batch = dataset_df.iloc[i:i + batch_size]
            if batch.empty:
                break

            sentiment_batch, embedding_batch = preprocess_batch(batch)

            # Сохраняем предобработанные данные
            save_preprocessed_data(connection, embedding_batch, sentiment_batch, batch['version'])

            time_passed = time.time() - start_time
            logging.warning(f'Processed rows: {i + batch_size} of {dataset_df.shape[0]}. Time spent: {time_passed}')
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
