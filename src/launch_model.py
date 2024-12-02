import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Логирование выводится в консоль
    ]
)
logger = logging.getLogger(__name__)

def launch_model(huggingface_token: str, query: str, model_directory: str, adapter_model_path: str, flag_cache: bool = False):
    if flag_cache:
        logger.info("Очистка кэша и вызов сборщика мусора...")
        gc.collect()
        torch.cuda.empty_cache()

    # Проверяем наличие GPU и автоматически распределяем модель по устройствам
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используемое устройство: {device}")

    # Загружаем токенизатор и базовую модель
    logger.info("Загрузка токенизатора и базовой модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_directory, token=huggingface_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        token=huggingface_token,
        low_cpu_mem_usage=True 
    ).to(device)

    # Загрузка адаптерной модели
    logger.info("Загрузка адаптерной модели...")
    model = PeftModel.from_pretrained(base_model, adapter_model_path).to(device)

    system_prompt = "Отвечай на русском языке. "
    input_text = system_prompt + query
    logger.info("Токенизация запроса...")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

    logger.info("Генерация текста...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Сниженное значение параметров инференса модели для экономии ресурсов, ты всегда можешь их изменить
        repetition_penalty=1.2,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Генерация завершена.")
    return generated_text


def main():
    logger.info("Запуск основного приложения.")
    flag_cache = input("Очистить кэш и вызвать сборщик мусора? Y/n: ").strip().lower() == 'y'
    query = input("Введите ваш запрос: ")
    huggingface_token = input("Введите ваш токен Hugging Face: ")  # здесь должен быть твой huggingface_token

    model_directory = "models/fine_tuned_gemma_model_dpo"
    adapter_model_path = "models/results"

    logger.info("Начало обработки запроса...")
    generated_text = launch_model(huggingface_token, query, model_directory, adapter_model_path, flag_cache)
    print("Сгенерированный ответ:\n", generated_text)
    logger.info("Работа завершена.")


if __name__ == "__main__":
    main()
