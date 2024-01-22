from sklearn.model_selection import train_test_split

from src.util.database import get_unsplitted_dataset, connect_to_database, save_splitted_dataset


def create_splits():
    """
    Создает разделения данных на train и test наборы и сохраняет в таблицу train_test_splits.
    """
    connection = connect_to_database()
    df_to_split = get_unsplitted_dataset(connection)

    # Разделяем данные на train и test в соотношении 80/20
    train_ids, test_ids = train_test_split(df_to_split['id'], test_size=0.2, random_state=42)

    values_train = [(dataset_id, 0) for dataset_id in train_ids]  # 0 для обучающего набора
    values_test = [(dataset_id, 1) for dataset_id in test_ids]  # 1 для тестового набора

    save_splitted_dataset(connection, values_train, values_test)
    connection.close()


if __name__ == "__main__":
    create_splits()
