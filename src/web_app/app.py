import os
import pickle

from flask import Flask, request, jsonify

from util.embeddings import calculate_embedding

model_path = os.path.join(os.path.dirname(__file__), 'deploy_model.pkl')

# Загрузка deployed модели
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Создание Flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из тела запроса
        data = request.get_json(force=True)
        comment = data['comment']

        # Векторизуем текст
        comment_embedding = calculate_embedding(comment)
        prediction = model.predict(comment_embedding)[0]

        # Маппинг результата
        result_mapping = {
            0: "негативный комментарий",
            1: "нейтральный комментарий",
            2: "положительный комментарий"
        }

        result = result_mapping.get(prediction, "неизвестный результат")

        # Формирование ответа в формате JSON
        response = {'result': result}

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
