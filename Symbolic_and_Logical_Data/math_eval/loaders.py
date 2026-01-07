#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Reasoning Data Evaluation - 数据集加载器

支持的数据集：
- MetaMathQA
- OpenMathInstruct-1
- GSM8K-Aug
- LILA (multi/iid/train.json)
- Nemotron-Math-Proofs (定理证明，暂不支持评估)
"""

import json
from pathlib import Path
from typing import Iterator, Optional, List
from data_types import MathSample


class BaseLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def load(self) -> List[MathSample]:
        """加载数据集，返回 MathSample 列表"""
        return list(self.iterate())
    
    def iterate(self) -> Iterator[MathSample]:
        """迭代返回 MathSample，子类需实现"""
        raise NotImplementedError


class MetaMathQALoader(BaseLoader):
    """
    MetaMathQA 数据集加载器
    
    数据格式:
    {
        "query": "数学问题",
        "response": "解答过程... The answer is: xxx",
        "type": "MATH_AnsAug" | "GSM_Rephrased" | ...,
        "original_question": "原始问题"
    }
    
    GT 提取: 从 response 末尾提取，格式为 "The answer is: xxx"
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        # 支持 .json 文件
        if self.data_path.is_dir():
            self.data_file = self.data_path / "MetaMathQA-395K.json"
        else:
            self.data_file = self.data_path
    
    def iterate(self) -> Iterator[MathSample]:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, record in enumerate(data):
            response = record.get('response', '')
            
            yield MathSample(
                question=record.get('query', ''),
                solution=response,
                ground_truth=self._extract_answer(response),  # 从 response 末尾提取 GT
                source_dataset=self._get_source(record.get('type', '')),
                question_type=record.get('type', ''),
                sample_id=f"metamath_{i}",
                metadata={
                    'original_question': record.get('original_question', '')
                }
            )
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        从 response 末尾提取答案
        
        支持的格式:
        - "The answer is: xxx"
        - "#### xxx"
        """
        import re
        
        # 按优先级尝试匹配
        patterns = [
            r'The answer is:?\s*(.+?)(?:\n|$)',  # The answer is: xxx (MetaMathQA 统一格式)
            r'####\s*(.+?)(?:\n|$)',              # #### xxx (GSM8K 风格)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _get_source(self, qtype: str) -> str:
        """从 type 字段推断原始数据集"""
        if qtype.startswith('MATH'):
            return 'math'
        elif qtype.startswith('GSM'):
            return 'gsm8k'
        return 'unknown'


class OpenMathInstructLoader(BaseLoader):
    """
    OpenMathInstruct-1 数据集加载器
    
    数据格式 (JSONL):
    {
        "question": "数学问题",
        "generated_solution": "包含代码的解答",
        "expected_answer": "标准答案",
        "predicted_answer": "预测答案",
        "is_correct": true/false,
        "generation_type": "...",
        "dataset": "gsm8k" | "math"
    }
    """
    
    def __init__(self, data_path: str, use_correct: bool = True):
        super().__init__(data_path)
        self.use_correct = use_correct
        
        # 确定数据文件路径
        if self.data_path.is_dir():
            subdir = "correct_solutions" if use_correct else "incorrect_solutions"
            self.data_file = self.data_path / subdir / "train.jsonl"
        else:
            self.data_file = self.data_path
    
    def iterate(self) -> Iterator[MathSample]:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                
                # 构建 metadata
                meta = {
                    'is_correct': record.get('is_correct')
                }
                # error_message 只在 incorrect_solutions 中存在
                if record.get('error_message'):
                    meta['error_message'] = record.get('error_message')
                
                yield MathSample(
                    question=record.get('question', ''),
                    solution=record.get('generated_solution', ''),
                    ground_truth=record.get('expected_answer', ''),  # OpenMathInstruct 有 GT
                    source_dataset=record.get('dataset', ''),
                    question_type=record.get('generation_type', ''),
                    sample_id=f"openmath_{i}",
                    metadata=meta
                )


class GSM8KAugLoader(BaseLoader):
    """
    GSM8K-Aug 数据集加载器
    
    数据来源: https://github.com/da03/Internalize_CoT_Step_by_Step
    论文: arxiv:2311.01460 "Implicit Chain of Thought Reasoning via Knowledge Distillation"
    
    数据格式 (JSON，按列存储):
    {
        "question": ["问题1", "问题2", ...],
        "cot": ["推理过程1", "推理过程2", ...],
        "answer": ["答案1", "答案2", ...]
    }
    
    两个版本:
    - GSM8K-Aug: 方程式版本 (gsm8k_aug_train.json)
    - GSM8K-Aug-NL: 自然语言版本 (gsm8k_aug_nl.json)
    """
    
    def __init__(self, data_path: str, use_nl: bool = False):
        """
        Args:
            data_path: 数据集目录路径
            use_nl: 是否使用自然语言版本 (GSM8K-Aug-NL)
        """
        super().__init__(data_path)
        self.use_nl = use_nl
        
        # 确定数据文件路径
        if self.data_path.is_dir():
            if use_nl:
                # 自然语言版本在 GSM8K-Aug-NL 目录
                nl_path = self.data_path.parent / "GSM8K-Aug-NL" / "gsm8k_aug_nl.json"
                if nl_path.exists():
                    self.data_file = nl_path
                else:
                    raise FileNotFoundError(f"GSM8K-Aug-NL not found: {nl_path}")
            else:
                self.data_file = self.data_path / "gsm8k_aug_train.json"
        else:
            self.data_file = self.data_path
    
    def iterate(self) -> Iterator[MathSample]:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 数据是按列存储的字典
        questions = data.get('question', [])
        cots = data.get('cot', [])
        answers = data.get('answer', [])
        
        for i in range(len(questions)):
            question = questions[i] if i < len(questions) else ''
            cot = cots[i] if i < len(cots) else ''
            answer = answers[i] if i < len(answers) else ''
            
            yield MathSample(
                question=question,
                solution=cot,  # cot 是推理过程
                ground_truth=str(answer),  # answer 是最终答案
                source_dataset='gsm8k',
                question_type='aug_nl' if self.use_nl else 'aug',
                sample_id=f"gsm8k_aug_{i}",
                metadata={}
            )


class LILALoader(BaseLoader):
    """
    LILA 数据集加载器
    
    数据来源: https://github.com/allenai/Lila
    论文: EMNLP 2022 "LĪLA: A Unified Benchmark for Mathematical Reasoning"
    
    数据格式 (JSONL):
    {
        "Input": "问题描述",
        "Output Program": ["Python 程序1", ...],
        "Output Answer": ["答案1", ...],
        "split": "train/dev/test",
        "dataset": "来源数据集名称",
        "dist": "iid/ood",
        "text": "格式化文本"
    }
    
    特点:
    - 有 Python 程序可执行验证
    - 有 GT 答案
    - 来自多个数据集的汇总
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 数据文件路径 (train_math_only.json)
        """
        super().__init__(data_path)
        self.data_file = self.data_path
    
    def iterate(self) -> Iterator[MathSample]:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                
                # 获取所有程序和答案
                programs = record.get('Output Program', [])
                answers = record.get('Output Answer', [])
                
                yield MathSample(
                    question=record.get('Input', ''),
                    solution=programs,  # 所有程序列表，评估时都执行验证
                    ground_truth=answers,  # 所有答案列表，合并后与执行结果比较
                    source_dataset=record.get('dataset', '').replace('.json', ''),
                    question_type=record.get('dist', ''),
                    sample_id=f"lila_{i}",
                    metadata={
                        'dist': record.get('dist', ''),
                        'split': record.get('split', ''),
                        'text': record.get('text', ''),
                    }
                )


# === 工厂函数 ===

def load_dataset(name: str, data_path: str, **kwargs) -> List[MathSample]:
    """
    统一的数据集加载接口
    
    Args:
        name: 数据集名称 ("metamathqa", "openmath", "gsm8k_aug")
        data_path: 数据路径
        **kwargs: 传递给具体 loader 的参数
    
    Returns:
        List[MathSample]
    
    Example:
        samples = load_dataset("metamathqa", "/path/to/MetaMathQA")
        samples = load_dataset("openmath", "/path/to/OpenMathInstruct-1", use_correct=True)
        samples = load_dataset("gsm8k_aug", "/path/to/GSM8K-Aug", use_nl=True)
    """
    loaders = {
        'metamathqa': MetaMathQALoader,
        'openmath': OpenMathInstructLoader,
        'openmathinstruct': OpenMathInstructLoader,
        'gsm8k_aug': GSM8KAugLoader,
        'lila': LILALoader,
    }
    
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower not in loaders:
        available = ', '.join(loaders.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    loader_cls = loaders[name_lower]
    loader = loader_cls(data_path, **kwargs)
    
    return loader.load()


def get_supported_datasets() -> List[str]:
    """返回支持的数据集列表"""
    return ['metamathqa', 'openmath', 'gsm8k_aug', 'lila']

