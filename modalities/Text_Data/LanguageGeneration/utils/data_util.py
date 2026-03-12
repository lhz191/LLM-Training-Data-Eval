import json
import os
from datasets import load_from_disk, load_dataset

# def get_dataset(file, instruction):
#     # with open(file, 'r', encoding="utf-8") as f:
#     #     data = []
#     #     for line in f:
#     #         data.append(json.loads(line))

#     #     for i in range(len(data)):
#     #         context = data[i]["dialogue"]
#     #         reference = data[i]["summary"]

#     #         ans = get_res_batch(context, instruction)

#     #         gen = {"input": context, "ground_truth": reference, "generation": ans}
#     #         dump_jsonl(gen, "generation/text-davinci-003.json")
#     return None

root_dir = "/mnt/petrelfs/wangjiedong/LLMDataBenchmark/Datasets/Text"

def get_xsum():
    file = os.path.join(root_dir, 'xsum')
    dataset = load_dataset('json', data_files = file)
    print(dataset)
    # dataset = dataset['train'].rename_columns({
    #     'dialogue' : 'text'
    #     }) # dialogue, summary
    return dataset['train']


def get_writingprompts():
    ds = load_dataset("euclaise/writingprompts")
    print(ds)
    dataset = ds['train'].rename_columns({
        'prompt': 'text',
        'story': 'summary'
    })
    return dataset

datasets = {
    "xsum" : get_xsum ,
    "writingprompts" : get_writingprompts
}

instruction = {
    "xsum" : "Please generate a one-sentence summary for the given document. "
}



def get_dataset(name:str):
    return datasets[name]()

if __name__  == '__main__':
    get_writingprompts()