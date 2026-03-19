import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

base_model_id = "mistralai/Mistral-7B-v0.1"
adapter_path = "./author_id_adapter"

model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id, 
    num_labels=3, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

test_df = pd.read_csv("test.csv")
results = []

print("Starting inference...")
with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        inputs = tokenizer(row['text'], return_tensors="pt", truncation=True, padding=True).to("cuda")

        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        results.append({
            "id": row['id'],
            "EAP": round(float(probs[0]), 6),
            "HPL": round(float(probs[1]), 6),
            "MWS": round(float(probs[2]), 6)
        })


submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
print("Inference complete! submission.csv created.")