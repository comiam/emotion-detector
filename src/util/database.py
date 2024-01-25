import json
import logging
import os
import pickle
from datetime import datetime

import numpy
import pandas as pd
import psycopg2
from psycopg2 import extras

from psycopg2.extensions import register_adapter, AsIs


def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


register_adapter(numpy.float64, adapt_numpy_float64)
register_adapter(numpy.int64, adapt_numpy_int64)


def connect_to_database():
    """
    Создаёт соединение с БД.
    """
    logging.info('Connecting to database...')
    user = os.getenv("DB_USERNAME")
    pwd = os.getenv("DB_PASSWORD")
    conn = psycopg2.connect(
        f"postgres://{user}:{pwd}@dpg-cmndg16g1b2c7397qpr0-a.frankfurt-postgres.render.com/emotional")
    return conn


def init_db_schema(connection):
    """
    Создаёт необходимые таблицы.
    """
    with connection.cursor() as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset (
                id SERIAL PRIMARY KEY,
                comment VARCHAR,
                sentiment INTEGER,
                version INTEGER,
                load_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessed_dataset (
                id SERIAL PRIMARY KEY,
                embedding TEXT,
                sentiment INTEGER,
                version INTEGER,
                load_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_test_splits (
                id SERIAL PRIMARY KEY,
                dataset_id INTEGER,
                split_type INTEGER
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trained_models (
                id SERIAL PRIMARY KEY,
                model_name TEXT,
                weights BYTEA,
                training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset_id INTEGER
            );
        ''')
    connection.commit()


def get_latest_dataset_version(connection):
    """
    Получает текущую максимальную версию в dataset из базы данных.
    """
    select_query = "SELECT COALESCE(MAX(version), 0) FROM dataset;"
    with connection.cursor() as cursor:
        cursor.execute(select_query)
        latest_version = cursor.fetchone()[0]
    return latest_version


def fetch_diff_between_datasets(connection):
    """
    Получает информацию, какие данные в dataset не были обработаны.
    """
    query = f'''
        SELECT d.*
        FROM dataset d
        LEFT JOIN preprocessed_dataset p ON d.version = p.version
        WHERE p.version IS NULL;
    '''
    return pd.read_sql_query(query, connection)


def save_new_data(df, connection):
    """
    Загружает данные из CSV-файла в базу данных с версионированием и временем загрузки.
    """
    logging.warning('Saving new dataset data to DB...')
    df = df[['Comment', 'Sentiment']]
    df.rename(columns={'Comment': 'comment'}, inplace=True)
    df.rename(columns={'Sentiment': 'sentiment'}, inplace=True)

    # Получаем текущую максимальную версию
    latest_version = get_latest_dataset_version(connection)

    # Увеличиваем версию для новых данных
    df['version'] = latest_version + 1
    df['load_time'] = datetime.now()

    # Записываем данные в базу данных
    insert_data_query = f'''
        INSERT INTO dataset ({', '.join(df.columns)}) VALUES %s;
    '''

    with connection.cursor() as cursor:
        values = [tuple(row) for row in df.itertuples(index=False, name=None)]
        extras.execute_values(cursor, insert_data_query, values)

    connection.commit()


def save_preprocessed_data(connection, preprocessed_data, sentiment, version):
    """
    Сохраняет ембэддинги и sentiment в таблицу preprocessed_dataset.
    """
    insert_query = '''
        INSERT INTO preprocessed_dataset (embedding, sentiment, version, load_time)
        VALUES (%s, %s, %s, %s);
    '''
    load_time = datetime.now()

    with connection.cursor() as cursor:
        for i in range(len(preprocessed_data)):
            if preprocessed_data[i] is None:
                continue
            embedding_bytea = json.dumps(preprocessed_data[i].tolist()[0])
            cursor.execute(insert_query, (embedding_bytea, int(sentiment.iloc[i]), int(version.iloc[i]), load_time))

    connection.commit()


def get_unsplitted_dataset(connection):
    """
    Получаем данные из таблицы preprocessed_dataset, которые еще не включены в train_test_splits.
    """
    query = '''
            SELECT p.id
            FROM preprocessed_dataset p
            LEFT JOIN train_test_splits t ON p.id = t.dataset_id
            WHERE t.id IS NULL;
        '''
    return pd.read_sql_query(query, connection)


def save_splitted_dataset(connection, values_train, values_test):
    """
    Сохраняем разделения в таблицу train_test_splits.
    """
    insert_query = '''
            INSERT INTO train_test_splits (dataset_id, split_type)
            VALUES (%s, %s);
        '''
    with connection.cursor() as cursor:
        cursor.executemany(insert_query, values_train)
        cursor.executemany(insert_query, values_test)
    connection.commit()


def load_train_dataset(connection):
    """
    Получаем тренировочные данные из train_test_splits.
    """
    query = '''
            SELECT embedding, sentiment, preprocessed_dataset.id AS dataset_id
            FROM preprocessed_dataset
            JOIN train_test_splits ON preprocessed_dataset.id = train_test_splits.dataset_id
            WHERE split_type = 0;
        '''
    return pd.read_sql_query(query, connection)


def save_model(connection, model, dataset_id):
    """
    Сохраняем обученную модель в trained_models.
    """
    # Сохранение весов модели в бинарном формате
    weights = psycopg2.Binary(pickle.dumps(model))

    with connection.cursor() as cursor:
        cursor.execute('''
            INSERT INTO trained_models (model_name, weights, dataset_id)
            VALUES (%s, %s, %s)
        ''', (str(model), weights, dataset_id))
    connection.commit()


def get_trained_models_last_dataset_versions(connection):
    """
    Получаем все последние марки trained датасета, до которых обучались модели в БД.
    """
    query = 'SELECT DISTINCT(dataset_id) FROM trained_models'

    with connection.cursor() as cursor:
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
