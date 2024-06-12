import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("HODACHI/glm-4-9b-chat-FT-ja-v0.3", trust_remote_code=True)
print(tokenizer.default_chat_template)