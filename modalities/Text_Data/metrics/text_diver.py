import math
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize

# 确保 nltk 已下载 punkt
import nltk
nltk.download('punkt')

def type_token_ratio(text):
    tokens = word_tokenize(text.lower())
    if len(tokens) == 0:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)

def distinct_n(text, n=2):
    """
    Distinct-N: ratio of unique n-grams to total n-grams in the text
    """
    tokens = word_tokenize(text.lower())
    if len(tokens) < n:
        return 0.0
    n_grams = list(ngrams(tokens, n))
    unique_ngrams = set(n_grams)
    return len(unique_ngrams) / len(n_grams)

def ngram_entropy(text, n=2):
    """
    n-gram Response Entropy: -sum(p*log(p)) over n-grams
    """
    tokens = word_tokenize(text.lower())
    if len(tokens) < n:
        return 0.0
    n_grams = list(ngrams(tokens, n))
    counts = Counter(n_grams)
    total = sum(counts.values())
    entropy = -sum((count/total) * math.log(count/total, 2) for count in counts.values())
    return entropy

# 示例输入
data = [
    {"id": "1", "gen_text": "This is the first generated text."},
    {"id": "2", "gen_text": "Another example of generated output."},
    {"id": "3", "gen_text": "This generated text is also an example."}
]

# 计算指标
results = []
for item in data:
    text = item["gen_text"]
    ttr = type_token_ratio(text)
    distinct_1 = distinct_n(text, 1)
    distinct_2 = distinct_n(text, 2)
    entropy_2 = ngram_entropy(text, 2)
    
    results.append({
        "id": item["id"],
        "TTR": ttr,
        "Distinct-1": distinct_1,
        "Distinct-2": distinct_2,
        "2-gram Entropy": entropy_2
    })

# 打印结果
for r in results:
    print(r)
