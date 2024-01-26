FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get install libpq-dev
RUN pip install psycopg2-binary

RUN pip install -r requirements.txt

COPY src/web_app /app/src/web_app
COPY src/util /app/src/web_app/util
COPY deployed/deploy_model.pkl /app/src/web_app/deploy_model.pkl

EXPOSE 5000

CMD ["python", "src/web_app/app.py"]
