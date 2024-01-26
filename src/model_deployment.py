import logging
import os

from db.database import connect_to_database, load_best_model, get_deployed_model, add_new_deploy_model, \
    get_deployed_model_with_weight


def choose_and_mark_to_deploy_best_model(connection):
    best_model = load_best_model(connection)
    result = 0

    if best_model:
        model_id, model_name = best_model

        existing_deploy_model_id = get_deployed_model(connection)

        if existing_deploy_model_id:
            # Если уже есть запись в deploy_models, проверяем, является ли текущая модель лучшей
            existing_deploy_model_id = existing_deploy_model_id[0]

            if existing_deploy_model_id != model_id:
                # Обновляем запись в deploy_models
                add_new_deploy_model(connection, model_id)
                logging.warning(f"Model with id={model_id}, name={model_name} marked for deployment.")
            else:
                logging.warning(f"Model with id={model_id}, name={model_name} still the best for deployment. Skipping.")
                result = 1
        else:
            # Если записи в deploy_models нет, то валяй деплоим
            add_new_deploy_model(connection, model_id)
            logging.warning(f"Model with id={model_id}, name={model_name} marked for deployment.")
    else:
        logging.warning(f"Nothing to deploy. Skipping.")
        result = 1

    connection.commit()
    return result


def save_deployed_model(connection):
    save_path = 'deployed/deploy_model.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    deployed_model_info = get_deployed_model_with_weight(connection)

    if deployed_model_info:
        model_id, weights = deployed_model_info

        with open(save_path, 'wb') as file:
            file.write(weights.tobytes())

        logging.warning(f"Marked for deploy model with id={model_id} saved successfully to {save_path}.")
        return 0
    else:
        logging.warning("No model deployed or deployed model not found in deployed model!")
        return 1


def check_result(connection, result):
    if result == 1:
        connection.close()
        exit(1)


if __name__ == "__main__":
    connection = connect_to_database()

    check_result(connection, choose_and_mark_to_deploy_best_model(connection))
    check_result(connection, save_deployed_model(connection))

    connection.close()
