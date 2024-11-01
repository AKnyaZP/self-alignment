from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
# Эти библиотеки я использую для того, чтобы использовать hf_token, не показывая его в открытом доступе
import os
from dotenv import load_dotenv


model_directory = "models/fine_tuned_gemma_model_dpo"
adapter_model_path = "models/results"  # Path to the adapter folder

# Здесь должен быть твой huggingface access_token))) 
load_dotenv()
huggingface_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_directory, use_auth_token=huggingface_token)
# Базовая дообученная модель
base_model = AutoModelForCausalLM.from_pretrained(model_directory, use_auth_token=huggingface_token)

#Загрузка модель с peft
model = PeftModel.from_pretrained(base_model, adapter_model_path)

system_prompt = "Отвечай на русском языке" # Модель может начать отвечать на английском, если в запросе есть английские слова
input_text = system_prompt + "Привет Gemma! Расскажи мне как провести self-alignment модели google/gemma-2-2b-it?"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    repetition_penalty=1.2,  # Значение от 1.1 до 1.5 помогает избежать повторений
    temperature=0.7,         # Снижает случайность, делая текст более детерминированным
    top_k=50,                # Ограничивает выбор из топ-50 токенов
    top_p=0.9                # Ограничивает выбор токенов по кумулятивной вероятности
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

