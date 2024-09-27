from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
import torch
from transformers import BitsAndBytesConfig

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=access_token).to("cuda")


messages = [
    {"role": "user", "content": "Расскажи мне, что могут спросить на собеседовании на Python разработчика"},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))