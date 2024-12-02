import json
import gc
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from bert_score import score


device = "cuda" if torch.cuda.is_available() else "cpu"

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb
    logger.info(f"BitsAndBytes доступен: GPU поддержка: {bnb.has_cuda_support}")
except ImportError:
    logger.warning("BitsAndBytes не установлен. Попробуйте `pip install bitsandbytes`.")
except Exception as e:
    logger.error(f"Ошибка при инициализации BitsAndBytes: {e}")


class EstimateFineTuneModel:
    def __init__(self, huggingface_token, model_directory, adapter_model_path, benchmark_name="sberquad"):
        self.huggingface_token = huggingface_token
        self.model_directory = model_directory
        self.adapter_model_path = adapter_model_path
        self.benchmark_name = benchmark_name
        self.device = device

        logger.info(f"Используемое устройство: {self.device}")

    def load_model(self):
        """Загружает модель и токенизатор на устройство."""
        logger.info("Загрузка токенизатора и модели...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_directory, token=self.huggingface_token)
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_directory,
                token=self.huggingface_token,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        except Exception as e:
            logger.error(f"Ошибка при загрузке базовой модели: {e}")
            raise

        try:
            model = PeftModel.from_pretrained(base_model, self.adapter_model_path).to(self.device)
        except Exception as e:
            logger.error(f"Ошибка при загрузке адаптера: {e}")
            raise

        logger.info("Модель успешно загружена.")
        return model, tokenizer

    def generate_answer_batch(self, queries, model, tokenizer):
        """Генерирует ответы на список запросов."""
        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Ограничиваем длину токенов
        ).to(self.device)

        logger.info(f"Генерация ответов для {len(queries)} запросов...")
        with torch.no_grad():  # Отключаем градиенты для ускорения
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Сокращаем максимальное количество новых токенов
                repetition_penalty=1.2,
                do_sample=True,  # Включаем семплирование
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        logger.info("Генерация завершена.")
        return generated_texts

    def compute_length_metrics(self, answers):
        """Оценка краткости."""
        length_ratios = []
        for ans in answers:
            generated_len = len(ans["generated"].split())
            reference_len = len(ans["reference"].split())
            ratio = generated_len / reference_len
            length_ratios.append(ratio)
        avg_length_ratio = sum(length_ratios) / len(length_ratios)
        return avg_length_ratio

    def compute_usefulness_metrics(self, answers, model_for_scoring="bert-base-multilingual-cased"):
        """Оценка полезности."""
        references = [ans["reference"] for ans in answers]
        generated = [ans["generated"] for ans in answers]

        try:
            with torch.no_grad():
                P, R, F1 = score(
                    generated,
                    references,
                    model_type=model_for_scoring,
                    lang="ru",
                    device=self.device
                )
        except KeyError:
            logger.error(f"Модель {model_for_scoring} не поддерживается. Проверьте доступные модели.")
            raise

        bert_score = {
            "Precision": P.mean().item(),
            "Recall": R.mean().item(),
            "F1": F1.mean().item()
        }

        exact_matches = [1 if ref.strip() == gen.strip() else 0 for ref, gen in zip(references, generated)]
        em_score = sum(exact_matches) / len(exact_matches)

        return {"Exact Match": em_score, "BERTScore": bert_score}


    def evaluate_model_on_benchmark(self):
        """Оценка модели на бенчмарке."""
        logger.info(f"Загрузка бенчмарка {self.benchmark_name}...")
        dataset = load_dataset(self.benchmark_name, split="validation")

        dataset = dataset.select(range(16))  # Берем только первые 100 примеров

        model, tokenizer = self.load_model()

        answers = []
        queries = []

        logger.info(f"Пример данных из {self.benchmark_name}: {dataset[0]}")
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            context = sample["context"]
            reference = sample["answers"]["text"][0]  # Используем первый эталонный ответ

            query = f"Вопрос: {question}\nКонтекст: {context}"
            queries.append(query)

            # Для ускорения генерируем ответы батчами
            if len(queries) >= 16 or idx == len(dataset) - 1:
                generated_batch = self.generate_answer_batch(queries, model, tokenizer)
                for query, generated in zip(queries, generated_batch):
                    answers.append({
                        "query": query,
                        "reference": reference,  # Указываем эталонный ответ из текущего sample
                        "generated": generated
                    })
                queries = []  # Очищаем список для следующего батча

        # Оценка краткости
        logger.info("Оценка краткости...")
        length_ratio = self.compute_length_metrics(answers)
        logger.info(f"Среднее соотношение длины ответа: {length_ratio:.2f}")

        # Оценка полезности
        logger.info("Оценка полезности...")
        usefulness_metrics = self.compute_usefulness_metrics(answers)
        logger.info("Метрики полезности:")
        logger.info(f"  Exact Match: {usefulness_metrics['Exact Match']:.2f}")
        logger.info(f"  BERTScore Precision: {usefulness_metrics['BERTScore']['Precision']:.2f}")
        logger.info(f"  BERTScore Recall: {usefulness_metrics['BERTScore']['Recall']:.2f}")
        logger.info(f"  BERTScore F1: {usefulness_metrics['BERTScore']['F1']:.2f}")


if __name__ == "__main__":
    huggingface_token = input("Введите ваш токен Hugging Face: ")
    model_directory = "models/fine_tuned_gemma_model_dpo"
    adapter_model_path = "models/results"

    evaluator = EstimateFineTuneModel(
        huggingface_token=huggingface_token,
        model_directory=model_directory,
        adapter_model_path=adapter_model_path
    )
    evaluator.evaluate_model_on_benchmark()




#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