import torch
import torch.nn as nn
import argparse
from utils.data_util import get_dataset
from utils.GrammaticalityRate import GrammaticalityRate
from utils.rouge import get_rouge_bleurt


"""
srun -p TDS --job-name=Sample --ntasks-per-node=1 python execute.py[]
"""




def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args.dataset)
    
    # res =  GrammaticalityRate(dataset, device)
    # print("Xsum-GR:", res)
    r1, r2 = get_rouge_bleurt(dataset)
    print(r1, r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default="xsum")
    
    args = parser.parse_args()
    main(args)
    
    
    
    
