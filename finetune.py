from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset 
import pandas as pd
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=3, 
    quantization_config=bnb_config, 
    device_map="auto"
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS 
)

# Convert labels to integers: EAP:0, HPL:1, MWS:2
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
df_train = pd.read_csv("train.csv")
df_train['label'] = df_train['author'].map(author_map)

dataset = Dataset.from_pandas(df_train[['text', 'label']])
def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_func, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1, 
    logging_steps=10,
    optim="paged_adamw_32bit",
    fp16=True,
    save_strategy="no", # Save manually at the end
    report_to="none"
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
)

trainer.train()

trainer.model.save_pretrained("./author_id_adapter")
print("Training complete! Adapter saved.")