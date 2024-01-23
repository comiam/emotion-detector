import os
from datetime import datetime

import pandas as pd
import psycopg2


def connect_to_database():
    """
        Создаёт соединение с БД.
    """
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
                embedding BYTEA,
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
        INSERT INTO dataset (comment, sentiment, load_time, version) VALUES (%s, %s, %s, %s);
    '''

    with connection.cursor() as cursor:
        for index, row in df.iterrows():
            cursor.execute(insert_data_query, (row['comment'], row['sentiment'], row['load_time'], row['version']))

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
            embedding_bytea = psycopg2.Binary(preprocessed_data[i].numpy().tobytes())
            cursor.execute(insert_query, (embedding_bytea, sentiment[i], version[i], load_time))

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
