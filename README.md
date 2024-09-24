# Self-Alignment with LLM for Russian Instructions

## Описание
Этот проект использует датасет  <a href='https://huggingface.co/datasets/OpenAssistant/oasst1'>OpenAssistant/oasst1</a> для обучения модели LLM, которая генерирует ответы на русскоязычные инструкции. Основная цель проекта — исследовать подходы к выравниванию модели (self-alignment) для достижения определенного стиля ответов.

## Структура проекта
- `data/`: Директория для хранения данных (не включены в репозиторий).
- `notebooks/`: Ноутбук с EDA и визуализацией данных.
- `src/`: Основной код для обучения, генерации ответов и оценки модели.
- `experiments/`: Настройки и результаты экспериментов.
- `models/`: Директория для сохранения весов обученных моделей.

## Установка и запуск
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/username/project_name.git
   cd project_name
   
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt

