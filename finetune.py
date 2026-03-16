from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device = "auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

def formatting_prompts_func(example):
    output_texts = []

    author_map = {
        "EAP": "Edgar Allan Poe",
        "HPL": "H.P. Lovecraft",
        "MWS": "Mary Shelley"
    }

    for i in range(len(example['text'])):
        instruction = (
            "Analyze the following gothic sentence and identify which author wrote it: "
            "Edgar Allan Poe, H.P. Lovecraft, or Mary Shelley."
        )
        input_text = example['text'][i]
        label = author_map.get(example['author'][i], example['author'][i])

        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{label}{tokenizer.eos_token}"
        )
        output_texts.append(text)
    return output_texts

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,
    optim="paged_adamw_32bit",      # The QLoRA-optimized optimizer
    fp16=True,
    save_strategy="steps",
    save_steps=50,
)