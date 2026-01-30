#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToolBench 动态可执行性检查器（实际调用 RapidAPI）
"""

import os
import sys
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from api_executor import DynamicChecker

from .constants import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    DERIVABILITY_PROMPT,
)

class ToolBenchDynamicChecker(DynamicChecker):
    """
    ToolBench 数据集动态可执行性检查器
    
    通过实际调用 RapidAPI 来验证：
    1. API 是否真实存在且可访问
    2. 参数格式是否正确
    3. 是否能获得有效响应
    4. final_answer 是否能从真实 API 返回推导出来
    
    Pass Rate 计算规则（符合 ToolBench 论文定义）：
    - 所有 API 调用都成功返回（非错误响应）
    - 最终 Finish 的 return_type = "give_answer"
    
    注意：此检查器需要 RapidAPI Key 和网络访问权限。
    
    使用方式:
        checker = ToolBenchDynamicChecker(
            rapidapi_key='your_key',
            toolenv_path='/path/to/toolenv/tools',
            cache_dir='/path/to/cache'
        )
        
        results = checker.check_sample(sample)
        # 或批量检查
        results = checker.check_batch(samples, sample_size=100)
    """
    
    def __init__(self,
                 rapidapi_key: Optional[str] = None,
                 toolenv_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 timeout: int = 180):  # 默认 3 分钟超时
        """
        初始化动态检查器
        
        Args:
            rapidapi_key: RapidAPI Key，如果为 None 则尝试从环境变量读取
            toolenv_path: toolenv 工具库路径
            cache_dir: API 映射缓存目录
            timeout: 单个 API 调用的超时时间（秒），默认 180 秒（3 分钟）
        """
        # 获取 API Key
        self.rapidapi_key = rapidapi_key or os.environ.get('RAPIDAPI_KEY')
        
        self.toolenv_path = toolenv_path
        self.cache_dir = cache_dir
        self.timeout = timeout
        
        # API 映射表
        self.api_mapping: Dict[str, Dict[str, Any]] = {}
        self._mapping_loaded = False
    
    def _build_api_mapping(self, force_rebuild: bool = False) -> None:
        """
        从 toolenv 目录构建 API 映射表（带缓存）
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
                print(f"从缓存加载 API 映射: {cache_path}", flush=True)
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.api_mapping = json.load(f)
                print(f"已加载 {len(self.api_mapping):,} 个 API 定义", flush=True)
                self._mapping_loaded = True
                return
            except Exception as e:
                print(f"缓存加载失败: {e}，重新构建...", flush=True)
        
        if not os.path.exists(self.toolenv_path):
            print(f"toolenv 路径不存在: {self.toolenv_path}", flush=True)
            return
        
        # 遍历所有 JSON 文件
        print(f"从 toolenv 构建 API 映射: {self.toolenv_path}", flush=True)
        json_files = glob.glob(os.path.join(self.toolenv_path, '**', '*.json'), recursive=True)
        print(f"找到 {len(json_files):,} 个 JSON 文件", flush=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    tool_data = json.load(f)
                
                tool_name = tool_data.get('tool_name', '')
                host = tool_data.get('host', '')
                
                tool_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', tool_name).lower().strip('_')
                
                for api in tool_data.get('api_list', []):
                    api_name = api.get('name', '')
                    api_name_normalized = re.sub(r'[^a-zA-Z0-9]+', '_', api_name).lower().strip('_')
                    
                    full_api_key = f"{api_name_normalized}_for_{tool_name_normalized}"
                    
                    self.api_mapping[full_api_key] = {
                        'url': api.get('url', ''),
                        'method': api.get('method', 'GET'),
                        'host': host,
                        'required_parameters': api.get('required_parameters', []),
                        'optional_parameters': api.get('optional_parameters', []),
                        'tool_name': tool_name,
                        'api_name': api_name,
                    }
            except Exception:
                pass
        
        self._mapping_loaded = True
        print(f"已构建 {len(self.api_mapping):,} 个 API 定义", flush=True)
        
        # 保存到缓存
        if cache_path:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.api_mapping, f, ensure_ascii=False)
                print(f"已保存缓存到: {cache_path}", flush=True)
            except Exception as e:
                print(f"保存缓存失败: {e}", flush=True)
    
    def _call_rapidapi(self, api_info: Dict, params: Dict) -> Tuple[bool, Optional[str], Any]:
        """
        调用 RapidAPI
        
        Args:
            api_info: API 定义信息
            params: 调用参数
        
        Returns:
            (success, error_message, response_data)
        """
        import requests
        
        if not self.rapidapi_key:
            return False, "No RapidAPI key configured", None
        
        url = api_info.get('url', '')
        method = api_info.get('method', 'GET').upper()
        host = api_info.get('host', '')
        
        if not url or not host:
            return False, "Missing URL or host in API definition", None
        
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": host
        }
        
        # 代理配置
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        proxies = None
        if http_proxy or https_proxy:
            proxies = {
                'http': http_proxy,
                'https': https_proxy
            }
        
        try:
            # 处理 URL 中的路径参数
            for param_name, param_value in params.items():
                url = url.replace(f'{{{param_name}}}', str(param_value))
            
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, 
                                       proxies=proxies, timeout=self.timeout)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=params,
                                        proxies=proxies, timeout=self.timeout)
            else:
                return False, f"Unsupported HTTP method: {method}", None
            
            # 检查响应状态
            if response.status_code == 200:
                try:
                    return True, None, response.json()
                except:
                    return True, None, response.text
            elif response.status_code == 401:
                return False, "Unauthorized (invalid API key)", None
            elif response.status_code == 403:
                return False, "Forbidden (API access denied)", None
            elif response.status_code == 404:
                return False, "API endpoint not found", None
            elif response.status_code == 429:
                return False, "Rate limit exceeded", None
            elif response.status_code >= 500:
                return False, f"Server error ({response.status_code})", None
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}", None
                
        except Exception as e:
            if 'Timeout' in str(type(e).__name__):
                return False, "Request timeout", None
            elif 'ConnectionError' in str(type(e).__name__):
                return False, "Connection error", None
            else:
                return False, f"Request error: {str(e)}", None
    
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
    
    def check_sample(self, sample: APIAgentSample) -> Dict[str, Any]:
        """
        检查单个样本的动态可执行性
        
        Args:
            sample: API Agent 样本
            
        Returns:
            检查结果字典
        """
        if not self.rapidapi_key:
            return {
                'error': 'No RapidAPI key configured',
                'passed': False,
            }
        
        # 构建 API 映射
        if self.toolenv_path:
            self._build_api_mapping()
        
        result = {
            'sample_id': sample.sample_id,
            'passed': True,
            'api_calls': [],
            'finish_type': None,
            'final_answer': sample.final_answer,
            'derivability': None,
            'errors': [],
        }
        
        all_api_responses = []
        
        for i, call in enumerate(sample.api_calls):
            if not call.name:
                continue
            
            # 处理 Finish
            if call.name == 'Finish':
                args = call.arguments or {}
                return_type = args.get('return_type', '')
                
                if return_type == 'give_answer':
                    result['finish_type'] = 'give_answer'
                elif return_type == 'give_up_and_restart':
                    result['finish_type'] = 'give_up_and_restart'
                    result['passed'] = False
                    result['errors'].append("Finish: give_up_and_restart")
                else:
                    result['finish_type'] = 'invalid'
                continue
            
            # 查找 API 定义
            api_key = call.name.lower()
            api_info = self.api_mapping.get(api_key)
            
            if not api_info:
                # 模糊匹配
                for key in self.api_mapping:
                    if key.startswith(api_key) or api_key.startswith(key):
                        api_info = self.api_mapping[key]
                        break
            
            call_result = {
                'api_name': call.name,
                'arguments': call.arguments,
                'success': False,
                'error': None,
                'response': None,
            }
            
            if not api_info:
                call_result['error'] = 'API not found in toolenv mapping'
                result['passed'] = False
                result['errors'].append(f"API '{call.name}' not found in toolenv")
            else:
                # 调用 API
                success, error, response = self._call_rapidapi(
                    api_info,
                    call.arguments if isinstance(call.arguments, dict) else {}
                )
                
                call_result['success'] = success
                call_result['error'] = error
                call_result['response'] = response
                
                if success:
                    response_str = json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response)
                    all_api_responses.append({
                        'api_name': call.name,
                        'response': response_str  # 不截断，保留完整响应
                    })
                else:
                    result['passed'] = False
                    result['errors'].append(f"API '{call.name}': {error}")
            
            result['api_calls'].append(call_result)
        
        # 检查 finish_type
        if result['finish_type'] != 'give_answer':
            result['passed'] = False
        
        # 检查答案可推导性（LLM Judge）
        if result['passed'] and sample.final_answer and all_api_responses:
            all_responses_text = "\n".join([
                f"[{r['api_name']}]: {r['response']}" for r in all_api_responses
            ])
            
            derivable, reason = self._check_derivability_llm(
                sample.final_answer, all_responses_text
            )
            
            result['derivability'] = {
                'derivable': derivable,
                'reason': reason,
            }
        
        return result
    
    def check_batch(self, samples: List[APIAgentSample], 
                    sample_size: int = 0,
                    show_progress: bool = True) -> Dict[str, Any]:
        """
        批量检查样本的动态可执行性
        
        Args:
            samples: API Agent 样本列表
            sample_size: 抽样数量，0 表示全部
            show_progress: 是否显示进度
            
        Returns:
            批量检查结果
        """
        if not self.rapidapi_key:
            return {
                'error': 'No RapidAPI key configured',
                'total_samples': 0,
                'passed_samples': 0,
                'pass_rate': 0.0,
            }
        
        # 构建 API 映射
        if self.toolenv_path:
            self._build_api_mapping()
        
        # 选择样本
        if sample_size > 0:
            test_samples = samples[:min(sample_size, len(samples))]
        else:
            test_samples = samples
        
        results = {
            'total_samples': len(test_samples),
            'passed_samples': 0,
            'failed_samples': 0,
            'api_call_stats': {
                'total': 0,
                'success': 0,
                'failed': 0,
                'not_found': 0,
            },
            'finish_stats': {
                'give_answer': 0,
                'give_up_and_restart': 0,
                'invalid': 0,
            },
            'derivability': {
                'total_checked': 0,
                'derivable': 0,
                'not_derivable': 0,
            },
            'failed_apis': Counter(),
            'success_apis': Counter(),
            'errors': [],
            'sample_results': [],
        }
        
        iterator = test_samples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(test_samples, desc="Verifying API calls")
            except ImportError:
                pass
        
        for sample in iterator:
            sample_result = self.check_sample(sample)
            results['sample_results'].append(sample_result)
            
            if sample_result.get('passed'):
                results['passed_samples'] += 1
            else:
                results['failed_samples'] += 1
                if sample_result.get('errors'):
                    results['errors'].append({
                        'sample_id': sample_result['sample_id'],
                        'errors': sample_result['errors']
                    })
            
            # 统计 API 调用
            for call_result in sample_result.get('api_calls', []):
                results['api_call_stats']['total'] += 1
                if call_result['success']:
                    results['api_call_stats']['success'] += 1
                    results['success_apis'][call_result['api_name']] += 1
                elif call_result['error'] == 'API not found in toolenv mapping':
                    results['api_call_stats']['not_found'] += 1
                else:
                    results['api_call_stats']['failed'] += 1
                    results['failed_apis'][call_result['api_name']] += 1
            
            # 统计 finish
            finish_type = sample_result.get('finish_type')
            if finish_type in results['finish_stats']:
                results['finish_stats'][finish_type] += 1
            
            # 统计 derivability
            deriv = sample_result.get('derivability')
            if deriv:
                results['derivability']['total_checked'] += 1
                if deriv['derivable']:
                    results['derivability']['derivable'] += 1
                else:
                    results['derivability']['not_derivable'] += 1
        
        # 计算 pass rate
        results['pass_rate'] = (
            results['passed_samples'] / results['total_samples'] * 100
            if results['total_samples'] > 0 else 0.0
        )
        
        # 计算 derivability rate
        if results['derivability']['total_checked'] > 0:
            results['derivability']['rate'] = (
                results['derivability']['derivable'] / 
                results['derivability']['total_checked'] * 100
            )
        else:
            results['derivability']['rate'] = 0.0
        
        return results


