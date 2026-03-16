"""下载 Layer 1 text metrics 所需的模型到本地 models/ 目录"""
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. roberta-base-CoLA
print("=" * 50)
print("Downloading textattack/roberta-base-CoLA ...")
print("=" * 50)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_path = os.path.join(MODELS_DIR, "roberta-base-CoLA")
tok = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
tok.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Saved to {save_path}")

# 2. nltk punkt
print("\n" + "=" * 50)
print("Downloading nltk punkt tokenizer ...")
print("=" * 50)
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
print("nltk punkt done")

print("\n All models downloaded.")
