import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel

base_model_id = "mistralai/Mistal-7B-v0.3"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype = torch.float16, 
                                                  device_map="auto")

adapter_path = "/results"
model = PeftModel.from_pretrained(base_model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

