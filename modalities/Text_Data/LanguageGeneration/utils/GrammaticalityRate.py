from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def GrammaticalityRate(ds, device):
    model_path = "/mnt/petrelfs/wangjiedong/.cache/huggingface/hub/models--textattack--roberta-base-CoLA/snapshots/3ccf3a400f2fa75ff257eac171047603ffbe84f1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    def cola_acceptability(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)

        accept_prob = probs[0][1].item()   # label=1 is "acceptable"
        return accept_prob

    
    results = []
    for i in range(len(ds)):  # 示例：只跑 20 条，可改成 len(ds)
        text = ds[i]["dialogue"]  # 也可改为 "summary"
        score = cola_acceptability(text)
        # results.append((i, score, text[:80] + "..."))
        results.append(score)
        # print(f"ID{i}:{score}")
    res = np.mean(score)
    
    return res
    
    
    
if __name__ == '__main__':
    pass