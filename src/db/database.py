import json
import logging
import os
from datetime import datetime

import numpy
import pandas as pd
import psycopg2
from psycopg2 import extras

from psycopg2.extensions import register_adapter, AsIs

from models.models import serialize_model, deserialize_model


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
            CREATE INDEX idx_dataset_id ON dataset(id);
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessed_dataset (
                id SERIAL PRIMARY KEY,
                embedding TEXT,
                sentiment INTEGER,
                version INTEGER,
                load_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_preprocessed_dataset_id ON preprocessed_dataset(id);
            CREATE INDEX idx_preprocessed_sentiment ON preprocessed_dataset(sentiment);
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS train_test_splits (
                id SERIAL PRIMARY KEY,
                dataset_id INTEGER,
                split_type INTEGER
            );
            CREATE INDEX idx_train_test_splits_id ON train_test_splits(id);
            CREATE INDEX idx_train_test_splits_split_type ON train_test_splits(split_type);
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_validated_models (
                id SERIAL PRIMARY KEY,
                precision NUMERIC,
                recall NUMERIC,
                accuracy NUMERIC,
                f1_score NUMERIC,
                validation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_id INTEGER
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deploy_models (
                id SERIAL PRIMARY KEY,
                model_id INTEGER,
                deploy_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def fetch_diff_between_datasets(connection, batch_size):
    """
    Получает информацию, какие данные в dataset не были обработаны.
    Возвращает общее количество строк для обработок, генератор df.
    """
    query = f'''
        SELECT d.*
        FROM dataset d
        LEFT JOIN preprocessed_dataset p ON d.version = p.version
        WHERE p.version IS NULL
        ORDER BY d.id ASC;
    '''
    query_count = f'''
        SELECT COUNT(d.*)
        FROM dataset d
        LEFT JOIN preprocessed_dataset p ON d.version = p.version
        WHERE p.version IS NULL;
    '''
    with connection.cursor() as cursor:
        cursor.execute(query_count)
        count = cursor.fetchone()[0]

    return count, pd.read_sql_query(query, connection, chunksize=batch_size)


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
            SELECT p.id, p.sentiment
            FROM preprocessed_dataset p
            LEFT JOIN train_test_splits t ON p.id = t.dataset_id
            WHERE t.id IS NULL;
        '''
    return pd.read_sql_query(query, connection)


def save_splitted_dataset(connection, splitted_dataset):
    """
    Сохраняем разделения в таблицу train_test_splits.
    """
    insert_query = '''
        INSERT INTO train_test_splits (dataset_id, split_type)
        VALUES %s;
    '''

    with connection.cursor() as cursor:
        extras.execute_values(cursor, insert_query, splitted_dataset)

    connection.commit()


def load_dataset_data(connection, split_type):
    """
    Получаем тренировочные данные из train_test_splits.
    """
    query = f'''
            SELECT embedding, sentiment, preprocessed_dataset.id AS dataset_id
            FROM preprocessed_dataset
            JOIN train_test_splits ON preprocessed_dataset.id = train_test_splits.dataset_id
            WHERE split_type = {split_type};
        '''
    return pd.read_sql_query(query, connection)


def save_model(connection, model, dataset_id):
    """
    Сохраняем обученную модель в trained_models.
    """
    # Сохранение весов модели в бинарном формате
    weights = psycopg2.Binary(serialize_model(model))

    with connection.cursor() as cursor:
        query = '''
            INSERT INTO trained_models (model_name, weights, dataset_id)
            VALUES (%s, %s, %s)
        '''
        cursor.execute(query, (str(model), weights, dataset_id))
    connection.commit()


def get_trained_models_last_dataset_versions(connection):
    """
    Получаем все последние марки trained датасета, до которых обучались модели в БД.
    """
    query = 'SELECT DISTINCT(dataset_id) FROM trained_models'

    with connection.cursor() as cursor:
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]


def load_trained_models(connection):
    """
    Загрузка моделей из БД.
    """
    models = []
    with connection.cursor() as cursor:
        cursor.execute('SELECT id, weights FROM trained_models')
        for id, weights in cursor.fetchall():
            model = deserialize_model(weights)
            models.append((id, model))

    return models


def save_best_model(connection, best_model_id, best_metrics):
    """
    Сохранение лучшей модели в БД.
    """
    with connection.cursor() as cursor:
        # Дабы избежать дубликатов
        cursor.execute('DELETE FROM best_validated_models WHERE model_id = %s', (best_model_id,))
        query = '''
                    INSERT INTO best_validated_models ("precision", recall, accuracy, f1_score, model_id)
                    VALUES (%s, %s, %s, %s, %s)
        '''
        cursor.execute(query, (best_metrics['precision'], best_metrics['recall'],
                               best_metrics['accuracy'], best_metrics['f1_score'], best_model_id))


def load_best_models(connection):
    """
    Загрузить лучшие модели из БД.
    """
    best_model_info = []

    with connection.cursor() as cursor:
        query = '''
            SELECT b.model_id, t.model_name, b.f1_score, t.weights
            FROM best_validated_models b
            JOIN trained_models t ON b.model_id = t.id
        '''
        cursor.execute(query)
        for model_id, model_name, f1_score, weights in cursor.fetchall():
            best_model_info.append((model_id, model_name, f1_score, weights))

    return best_model_info


def count_best_models(connection):
    """
    Возвращает количество сохранённых лучших моделей БД.
    """
    with connection.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) FROM best_validated_models')
        count = cursor.fetchone()[0]
    return count


def load_best_model(connection):
    """
    Извлечение лучшей модели из таблицы best_validated_models.
    """
    with connection.cursor() as cursor:
        query = '''
                SELECT model_id, model_name
                FROM best_validated_models
                INNER JOIN trained_models ON best_validated_models.model_id = trained_models.id
                ORDER BY f1_score DESC
                LIMIT 1;
        '''
        cursor.execute(query)
        best_model = cursor.fetchone()

    return best_model


def truncate_models(connection):
    """
    Сохранить 2 лучшие модели в БД, остальные удалить.
    """
    with connection.cursor() as cursor:
        # Топ 2 модели по метрике f1
        cursor.execute('''
            SELECT tm.id
            FROM trained_models tm
            INNER JOIN best_validated_models bvm ON tm.id = bvm.model_id
            ORDER BY bvm.f1_score DESC
        ''')
        top2_models = []
        for idx in cursor.fetchmany(2):
            top2_models.append(int(idx[0]))

        top2_models = tuple(top2_models)

        # Удаляем остальные модели из trained_models и best_validated_models
        cursor.execute('''
            DELETE FROM best_validated_models WHERE model_id NOT IN %s
        ''', (top2_models,))
        cursor.execute('''
            DELETE FROM trained_models WHERE id NOT IN %s
        ''', (top2_models,))

    logging.warning("Top 2 models saved and others deleted successfully.")


def get_deployed_model(connection):
    """
    Получение записей моделей из deploy_models.
    """
    with connection.cursor() as cursor:
        cursor.execute('SELECT model_id FROM deploy_models')
        existing_deploy_model_id = cursor.fetchone()

    return existing_deploy_model_id


def get_deployed_model_with_weight(connection):
    """
    Получение записей моделей из deploy_models, но с весами.
    """
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT d.model_id, t.weights
            FROM deploy_models d
            JOIN trained_models t ON d.model_id = t.id
        ''')
        deployed_model_info = cursor.fetchone()

    return deployed_model_info


def add_new_deploy_model(connection, model_id):
    with connection.cursor() as cursor:
        cursor.execute('DELETE FROM deploy_models')
        cursor.execute('INSERT INTO deploy_models (model_id) VALUES (%s);', (model_id,))
