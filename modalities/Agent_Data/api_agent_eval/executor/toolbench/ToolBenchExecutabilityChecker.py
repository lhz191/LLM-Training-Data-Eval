#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToolBench 可执行性检查器（静态检查）
"""

import os
import sys
import re
import json
import glob
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import ExecutabilityChecker

from .constants import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    DERIVABILITY_PROMPT,
    RELEVANCE_PROMPT,
)

class ToolBenchExecutabilityChecker(ExecutabilityChecker):
    """
    ToolBench 数据集可执行性检查器
    
    复用自 evaluate_toolbench_basic.py.evaluate_executability
    
    检查项：
    1. System Constraint Check（系统约束检查）
       - API 调用是否在样本定义的工具列表中
       - Finish 函数跳过此检查
       
    2. Toolenv Definition Check（工具库定义检查）[可选]
       - API 是否在 toolenv 全局工具库中存在
       - 需要提供 toolenv_path
       
    3. Required Parameter Check（必需参数检查）
       - API 调用时是否提供了所有必需参数
       
    4. Train Derivability（训练数据可推导性）
       - final_answer 是否能从 API 响应中推导
       
    5. Query-Answer Relevance（查询-回答相关性）
       - final_answer 是否回答了用户的 query
    """
    
    def __init__(self, 
                 toolenv_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化 ToolBench 可执行性检查器
        
        Args:
            toolenv_path: toolenv 工具库路径，用于全局 API 存在性检查
                         如果为 None，则跳过 toolenv 检查
            cache_dir: API 映射缓存目录，默认为当前目录下的 cache/
        """
        self.toolenv_path = toolenv_path
        # 默认使用当前目录下的 cache 目录
        if cache_dir is None:
            import os
            self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        else:
            self.cache_dir = cache_dir
        
        # API 映射表：{标准化API名: {url, method, host, required_parameters, ...}}
        self.api_mapping: Dict[str, Dict[str, Any]] = {}
        self._mapping_loaded = False
    
    def _build_api_mapping(self, force_rebuild: bool = False) -> None:
        """
        从 toolenv 目录构建 API 映射表（带缓存）
        
        复用自 evaluate_toolbench_basic.py._build_api_mapping
        
        映射格式：
        {
            "{api_name}_for_{tool_name_normalized}": {
                "url": "https://...",
                "method": "GET/POST",
                "host": "xxx.p.rapidapi.com",
                "required_parameters": [...],
                "optional_parameters": [...],
                "tool_name": "原始工具名",
                "api_name": "原始API名"
            }
        }
        """
        if self._mapping_loaded and not force_rebuild:
            return
        
        if not self.toolenv_path:
            return
        
        # 缓存路径
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, 'api_mapping_cache.json')
        
        # 尝试从缓存加载
        if cache_path and not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.api_mapping = json.load(f)
                self._mapping_loaded = True
                return
            except Exception:
                pass
        
        if not os.path.exists(self.toolenv_path):
            return
        
        # 遍历所有类别目录下的 JSON 文件
        json_files = glob.glob(os.path.join(self.toolenv_path, '**', '*.json'), recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    tool_data = json.load(f)
                
                tool_name = tool_data.get('tool_name', '')
                host = tool_data.get('host', '')
                
                # 将工具名标准化为下划线格式
                tool_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', tool_name).lower().strip('_')
                
                for api in tool_data.get('api_list', []):
                    api_name = api.get('name', '')
                    api_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', api_name).lower().strip('_')
                    
                    # 构建完整的 API 键
                    full_api_key = f"{api_name_normalized}_for_{tool_name_normalized}"
                    
                    self.api_mapping[full_api_key] = {
                        'url': api.get('url', ''),
                        'method': api.get('method', 'GET'),
                        'host': host,
                        'required_parameters': api.get('required_parameters', []),
                        'optional_parameters': api.get('optional_parameters', []),
                        'tool_name': tool_name,
                        'api_name': api_name,
                        'code': api.get('code', ''),
                    }
            except Exception:
                pass
        
        self._mapping_loaded = True
        
        # 保存到缓存
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.api_mapping, f, ensure_ascii=False)
            except Exception:
                pass
    
    def check(self, sample: APIAgentSample) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """检查 ToolBench 样本的可执行性"""
        errors = []
        warnings = []
        stats = {
            'api_calls_checked': 0,
            'toolenv_check_enabled': bool(self.toolenv_path),
            'train_derivability': None,
            'query_relevance': None,
        }
        
        # 构建工具映射（如果提供了 toolenv_path）
        if self.toolenv_path:
            self._build_api_mapping()
        
        # 1. 构建样本内的工具名映射
        tool_names = [t.name for t in sample.tools]
        tool_map = {t.name: t for t in sample.tools}
        
        # 2. 检查每个 API 调用
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue
            
            stats['api_calls_checked'] += 1
            
            # Finish 函数跳过
            if call.name == 'Finish':
                continue
            
            # 2.1 System Constraint Check - API 是否在样本工具列表中
            if call.name not in tool_names:
                errors.append(f"API Call {i}: API '{call.name}' not defined in sample tools")
                continue
            
            # 2.2 Toolenv Definition Check - API 是否在全局工具库中
            if self.toolenv_path and self.api_mapping:
                api_key = call.name.lower()
                if api_key not in self.api_mapping:
                    errors.append(f"API Call {i}: API '{call.name}' not found in toolenv mapping")
            
            # 2.3 Required Parameter Check (Sample Tool) - 基于样本内工具定义的必需参数检查
            if call.name in tool_map:
                tool = tool_map[call.name]
                provided_args = set(call.arguments.keys()) if call.arguments else set()
                
                for param in (tool.parameters or []):
                    if not param.optional and param.name not in provided_args:
                        if param.default is None:
                            errors.append(
                                f"API Call {i} ({call.name}): missing required parameter '{param.name}' (sample tool)"
                            )
            
            # 2.4 Required Parameter Check (Toolenv) - 基于全局工具库的必需参数检查
            if self.toolenv_path and self.api_mapping:
                api_key = call.name.lower()
                if api_key in self.api_mapping:
                    api_info = self.api_mapping[api_key]
                    provided_args = set(call.arguments.keys()) if call.arguments else set()
                    
                    # 检查 toolenv 中定义的 required_parameters
                    for req_param in api_info.get('required_parameters', []):
                        param_name = req_param.get('name', '') if isinstance(req_param, dict) else str(req_param)
                        # 获取 default 值（toolenv 格式：{"name": "...", "default": "..."}）
                        param_default = req_param.get('default', '') if isinstance(req_param, dict) else ''
                        
                        # 只有当：1) 参数未提供 且 2) 没有 default 值时，才报错
                        if param_name and param_name not in provided_args:
                            if not param_default:  # default 为空字符串或 None 时才报错
                                errors.append(
                                    f"API Call {i} ({call.name}): missing required parameter '{param_name}' (toolenv)"
                                )
        
        # 3. Train Derivability - 检查 final_answer 是否能从 API 响应推导（LLM Judge）
        if sample.final_answer:
            # 收集所有 API 响应
            all_responses = []
            for call in sample.api_calls:
                if call.name != 'Finish' and call.response:
                    all_responses.append(f"[{call.name}]: {str(call.response)}")
            
            if all_responses:
                all_responses_text = "\n".join(all_responses)
                derivable, reason = self._check_derivability_llm(
                    sample.final_answer, all_responses_text
                )
                stats['train_derivability'] = {
                    'derivable': derivable,
                    'reason': reason,
                }
                if not derivable:
                    errors.append(f"Train Derivability: {reason}")
        
        # 4. Query-Answer Relevance - 检查 final_answer 是否回答了 query（LLM Judge）
        if sample.final_answer and sample.query:
            relevant, reason = self._check_relevance_llm(
                sample.query, sample.final_answer
            )
            stats['query_relevance'] = {
                'relevant': relevant,
                'reason': reason,
            }
            if not relevant:
                errors.append(f"Query-Answer Relevance: {reason}")
        
        return errors, warnings, stats
    
    def _check_keyword_derivability(self, source_text: str, target_text: str) -> Tuple[bool, str]:
        """
        关键词匹配检查答案可推导性
        
        复用自 evaluate_toolbench_basic.py._check_keyword_derivability
        
        从 source_text 中提取关键信息（数字、专有名词、关键词），
        检查这些信息是否出现在 target_text 中。
        
        Args:
            source_text: 源文本（如 final_answer 或 query）
            target_text: 目标文本（如 API 响应或 final_answer）
        
        Returns:
            (derivable: bool, info: str)
        """
        # 1. 提取数字（包括小数、百分比、金额等）
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|°[CF]?)?\b', source_text)
        
        # 2. 提取可能的专有名词（连续大写字母开头的词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_text)
        
        # 3. 提取引号中的内容
        quoted = re.findall(r'["\']([^"\']+)["\']', source_text)
        
        # 4. 提取特殊格式（日期等）
        dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', source_text)
        
        # 合并所有关键信息
        all_keywords = []
        all_keywords.extend([(n, 'number') for n in numbers if len(n) >= 2])
        all_keywords.extend([(n, 'proper_noun') for n in proper_nouns if len(n) >= 3])
        all_keywords.extend([(q, 'quoted') for q in quoted if len(q) >= 3])
        all_keywords.extend([(d, 'date') for d in dates])
        
        if not all_keywords:
            return True, "No extractable keywords (assumed derivable)"
        
        # 检查匹配
        matched = []
        not_matched = []
        
        target_text_lower = target_text.lower()
        
        for keyword, kw_type in all_keywords:
            if keyword.lower() in target_text_lower:
                matched.append((keyword, kw_type))
            else:
                not_matched.append((keyword, kw_type))
        
        # 计算匹配率
        match_rate = len(matched) / len(all_keywords) if all_keywords else 0
        
        # 判断是否可推导：至少 50% 的关键信息能匹配到
        derivable = match_rate >= 0.5
        
        info = f"Matched {len(matched)}/{len(all_keywords)} ({match_rate*100:.0f}%)"
        if matched:
            info += f" | Found: {[k for k, _ in matched[:5]]}"
        if not_matched:
            info += f" | Missing: {[k for k, _ in not_matched[:5]]}"
        
        return derivable, info
    
    def _check_derivability_llm(
        self, 
        final_answer: str, 
        api_responses: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 作为 Judge 检查答案可推导性
        
        Args:
            final_answer: 最终答案
            api_responses: API 响应文本
            max_retries: 最大重试次数
        
        Returns:
            (derivable: bool, reason: str)
        """
        try:
            from openai import OpenAI
        except ImportError:
            return False, "OpenAI library not installed"
        
        prompt = DERIVABILITY_PROMPT.format(
            api_responses=api_responses,
            final_answer=final_answer
        )
        
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                derivable = result.get('derivable', False)
                reason = result.get('reason', '')
                
                return derivable, reason
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False, "LLM call failed"
    
    def _check_relevance_llm(
        self, 
        query: str, 
        final_answer: str,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        使用 LLM 作为 Judge 检查查询-答案相关性
        
        Args:
            query: 用户查询
            final_answer: 最终答案
            max_retries: 最大重试次数
        
        Returns:
            (relevant: bool, reason: str)
        """
        try:
            from openai import OpenAI
        except ImportError:
            return False, "OpenAI library not installed"
        
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            final_answer=final_answer
        )
        
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # 提取 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                relevant = result.get('relevant', False)
                reason = result.get('reason', '')
                
                return relevant, reason
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False, "LLM call failed"



