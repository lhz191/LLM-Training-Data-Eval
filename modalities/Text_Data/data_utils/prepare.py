from datasets import load_dataset
import json


# ds = load_dataset("roskoN/dailydialog")



# with open('dailydialog_id_utterances.jsonl', 'w', encoding='utf-8') as f:
#     for item in ds['train']:
#         data = {
#             'id': item['id'],
#             'utterances': item['utterances']
#         }
#         f.write(json.dumps(data, ensure_ascii=False) + '\n')


def parse_triple(triple_str):
    parts = triple_str.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid triple: {triple_str}")
    subject = parts[0].strip()
    predicate = parts[1].strip()
    obj = parts[2].strip().strip('"')  # 去掉可能的引号
    return subject, predicate, obj



ds = load_dataset("GEM/web_nlg", "en")

with open('webnlg.jsonl', 'w', encoding='utf-8') as f:
    for item in ds['train']:
        
        s, p, o = parse_triple(item['input'][0])
        prompt = "Generate a natural sentence describing these facts: " + s + p + o
        data = {
            'id' : item['gem_id'],
            'ref_text' : prompt,
            'gen_text' : item['target']
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')