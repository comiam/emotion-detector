#!/bin/bash

set -e

DOCKER_IMAGE_NAME="comiam/nlp_web_app"
DOCKER_COMPOSE_PATH="~/docker-compose.yml"
REMOTE_SCRIPT_PATH="~/script.sh"

# Создание Docker Compose файла локально
echo "
version: '3'

services:
  flask-app:
    image: $DOCKER_IMAGE_NAME:latest
    ports:
      - \"5000:5000\"
    restart: always
" > docker-compose.yml

# Передача Docker Compose файла на удаленный сервер
scp docker-compose.yml admin@$DEPLOY_SERVER_IP:$DOCKER_COMPOSE_PATH

# Создание скрипта для выполнения на удаленном сервере
echo "
#!/bin/bash

docker compose -f $DOCKER_COMPOSE_PATH down
docker image prune -f

docker pull $DOCKER_IMAGE_NAME:latest
docker compose -f $DOCKER_COMPOSE_PATH up --build -d
" > script.sh

# Передача скрипта на удаленный сервер
scp script.sh admin@$DEPLOY_SERVER_IP:$REMOTE_SCRIPT_PATH

# Удаление локальных файлов
rm docker-compose.yml script.sh

# Выполнение скрипта на удаленном сервере
ssh admin@$DEPLOY_SERVER_IP "chmod +x $REMOTE_SCRIPT_PATH && $REMOTE_SCRIPT_PATH"
