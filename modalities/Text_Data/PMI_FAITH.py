import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

# -----------------------------
# 初始化模型
# -----------------------------
MODEL_NAME = "gpt2"  # 你可以换成其他自回归 LM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# 工具函数：计算 log P(x|context)
# -----------------------------
def log_prob_sequence(context: str, response: str) -> float:
    """
    给定 context (h 或 d+h) 和 response r，返回 log P(r | context)
    """
    # 拼接 context 和 response
    input_text = context + " " + response
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # loss 是负对数似然
        # multiply by token 数得到总 log likelihood
        total_log_prob = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return total_log_prob

# -----------------------------
# PMI-FAITH 计算函数
# -----------------------------
def compute_pmi_faith(samples: List[Dict], doc: str) -> List[Dict]:
    """
    samples: List[{'id': str, 'utterances': List[str]}]
    doc: 文档 d
    返回每个样本的 PMI-FAITH
    """
    results = []
    for sample in samples:
        h = " ".join(sample['utterances'][:-1])  # 对话历史 h
        r = sample['utterances'][-1]             # 最后一句作为生成响应 r
        
        # log P(r | h)
        log_p_r_given_h = log_prob_sequence(h, r)
        # log P(r | d, h)
        log_p_r_given_dh = log_prob_sequence(doc + " " + h, r)
        
        # PMI-FAITH
        pmi_faith = log_p_r_given_dh - log_p_r_given_h
        
        results.append({
            'id': sample['id'],
            'pmi_faith': pmi_faith
        })
    return results

# -----------------------------
# 示例使用
# -----------------------------
if __name__ == "__main__":
    samples = [
        {
            'id': 'example_0',
            'utterances': [
                "Say, Jim, how about going for a few beers after dinner?",
                "You know that is tempting but is really not good for our fitness.",
                "What do you mean? It will help us to relax.",
                "Do you really think so? I don't. It will just make us fat and act silly. Remember last time?",
                "I guess you are right.But what shall we do? I don't feel like sitting at home.",
                "I suggest a walk over to the gym where we can play singsong and meet some of our friends.",
                "That's a good idea. I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them.",
                "Sounds great to me! If they are willing, we could ask them to go dancing with us.That is excellent exercise and fun, too.",
                "Good.Let's go now.",
                "All right."
            ]
        }
    ]

    # 假设文档 d
    doc = "We can exercise and meet friends at the gym, play pingpong, or go dancing together."

    results = compute_pmi_faith(samples, doc)
    for res in results:
        print(res)
