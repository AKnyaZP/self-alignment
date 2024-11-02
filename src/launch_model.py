from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc


def launch_model(huggingface_token: str, query: str, model_directory: str, adapter_model_path: str, flag_cache: bool = False):
    if flag_cache:
        print("Очистка кэша и вызов сборщика мусора...")
        gc.collect()
        torch.cuda.empty_cache()

    # Проверяем наличие GPU и автоматическое распределяем модель по устройствам
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # Загружаем токенизатор и базовую модель с токеном аутентификации и оптимизацией памяти
    print("Загрузка токенизатора и модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_directory, token=huggingface_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        token=huggingface_token,
        low_cpu_mem_usage=True  # Оптимизация использования CPU памяти. Можешь убрать, если потянешь
    ).to(device)

    # Загрузка адаптерной модели
    print("Загрузка адаптерной модели...")
    model = PeftModel.from_pretrained(base_model, adapter_model_path).to(device)

    # Подготовка системного промпта и токенизация
    system_prompt = "Отвечай на русском языке. "
    input_text = system_prompt + query
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Генерация текста с оптимизированными параметрами
    print("Генерация ответа...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Сниженное значение для экономии ресурсов (Ты можешь изменить его)
        repetition_penalty=1.2,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    flag_cache = input("Очистить кэш и вызвать сборщик мусора? Y/n: ").strip().lower() == 'y'
    query = input("Введите ваш запрос: ")
    huggingface_token = input("Введите ваш токен Hugging Face: ")  # здесь должен быть твой huggingface_token

    model_directory = "models/fine_tuned_gemma_model_dpo"
    adapter_model_path = "models/results"

    generated_text = launch_model(huggingface_token, query, model_directory, adapter_model_path, flag_cache)
    print("Сгенерированный ответ:\n", generated_text)


if __name__ == "__main__":
    main()
