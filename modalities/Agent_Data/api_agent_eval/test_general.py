#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeneralLoader / GeneralFormatChecker / GeneralExecutabilityChecker 测试

测试用例设计覆盖 data_types.py 中所有 dataclass 字段：

  APIAgentSample
    ├── query: str                          → 空值检测
    ├── tools: List[ToolDefinition]         → 空列表 / 重复名 / 字段缺失
    │     ├── name: str                     → 空值
    │     ├── description: str              → 空值 (warning)
    │     └── parameters: List[Parameter]
    │           ├── name: str               → 空值 / 重复
    │           ├── type: str               → 空值 / 类型匹配
    │           ├── description: str
    │           ├── default: Any            → 有默认值时允许缺失
    │           ├── required: bool          → 必需参数缺失检测
    │           ├── optional: bool          → 互补性检测
    │           └── metadata: Dict
    ├── api_calls: List[APICall]            → 空列表 / 工具不存在 / 参数不匹配
    │     ├── name: str                     → 空值 / 不在 tools 中
    │     ├── arguments: Dict               → 类型检测 / 未知参数
    │     ├── response: Optional[str]
    │     └── metadata: Dict
    ├── final_answer: Optional[str]         → 空字符串 (warning)
    ├── sample_id: Optional[str]
    ├── source_dataset: Optional[str]
    └── metadata: Dict

运行:
    cd api_agent_eval
    python test_general.py
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter
from loaders import GeneralLoader
from executor.general import GeneralFormatChecker, GeneralExecutabilityChecker


# =============================================================================
# 测试数据
# =============================================================================

SAMPLES = [
    # ---- Case 1: 完全正常的样本 ----
    {
        "query": "What is the weather in Beijing?",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": [
                    {"name": "city", "type": "str", "description": "City name",
                     "required": True, "optional": False},
                    {"name": "unit", "type": "str", "description": "Temperature unit",
                     "default": "celsius", "required": False, "optional": True}
                ]
            }
        ],
        "api_calls": [
            {"name": "get_weather", "arguments": {"city": "Beijing"}}
        ],
        "final_answer": "The weather in Beijing is 25°C and sunny.",
        "sample_id": "case1_clean",
        "source_dataset": "test"
    },

    # ---- Case 2: 多工具 + 多调用，全部正确 ----
    {
        "query": "Find Italian restaurants near Central Park and book a table for 2",
        "tools": [
            {
                "name": "search_restaurants",
                "description": "Search for restaurants by location and cuisine",
                "parameters": [
                    {"name": "location", "type": "str", "required": True, "optional": False},
                    {"name": "cuisine", "type": "str", "required": False, "optional": True}
                ]
            },
            {
                "name": "book_table",
                "description": "Book a table at a restaurant",
                "parameters": [
                    {"name": "restaurant_id", "type": "str", "required": True, "optional": False},
                    {"name": "party_size", "type": "int", "required": True, "optional": False},
                    {"name": "notes", "type": "str", "required": False, "optional": True}
                ]
            }
        ],
        "api_calls": [
            {"name": "search_restaurants", "arguments": {"location": "Central Park", "cuisine": "Italian"}},
            {"name": "book_table", "arguments": {"restaurant_id": "rest_123", "party_size": 2}}
        ],
        "final_answer": "Booked a table for 2 at Trattoria Roma near Central Park.",
        "sample_id": "case2_multi_tool"
    },

    # ---- Case 3: FormatChecker warnings ----
    # - tool description 为空
    # - required=True AND optional=True (互补性违反)
    # - final_answer 是空字符串
    {
        "query": "Search something",
        "tools": [
            {
                "name": "search",
                "description": "",
                "parameters": [
                    {"name": "q", "type": "str", "required": True, "optional": True}
                ]
            }
        ],
        "api_calls": [
            {"name": "search", "arguments": {"q": "hello"}}
        ],
        "final_answer": "",
        "sample_id": "case3_format_warnings"
    },

    # ---- Case 4: FormatChecker errors ----
    # - api_calls 中调用不存在的工具
    {
        "query": "Translate hello to French",
        "tools": [
            {
                "name": "translate",
                "description": "Translate text",
                "parameters": [
                    {"name": "text", "type": "str", "required": True, "optional": False},
                    {"name": "target_lang", "type": "str", "required": True, "optional": False}
                ]
            }
        ],
        "api_calls": [
            {"name": "nonexistent_api", "arguments": {"text": "hello"}}
        ],
        "sample_id": "case4_format_errors"
    },

    # ---- Case 5: ExecutabilityChecker errors ----
    # - 缺必需参数 (destination, date)
    # - 类型不匹配 (passengers 期望 int 给了不可解析的 str)
    # - 工具不存在 (第一个 call)
    {
        "query": "Book a flight from Beijing to Shanghai",
        "tools": [
            {
                "name": "book_flight",
                "description": "Book a flight ticket",
                "parameters": [
                    {"name": "origin", "type": "str", "required": True, "optional": False},
                    {"name": "destination", "type": "str", "required": True, "optional": False},
                    {"name": "date", "type": "str", "required": True, "optional": False},
                    {"name": "passengers", "type": "int", "required": False, "optional": True}
                ]
            }
        ],
        "api_calls": [
            {"name": "check_availability", "arguments": {}},
            {"name": "book_flight", "arguments": {"origin": "Beijing", "passengers": "not_a_number"}}
        ],
        "sample_id": "case5_exec_errors"
    },

    # ---- Case 6: ExecutabilityChecker warnings ----
    # - 未知参数 (extra_field)
    # - 类型 warning: 字符串形式的数字 passengers="2" (可解析为 int)
    {
        "query": "Book a flight with extras",
        "tools": [
            {
                "name": "book_flight",
                "description": "Book a flight ticket",
                "parameters": [
                    {"name": "origin", "type": "str", "required": True, "optional": False},
                    {"name": "destination", "type": "str", "required": True, "optional": False},
                    {"name": "date", "type": "str", "required": True, "optional": False},
                    {"name": "passengers", "type": "int", "required": False, "optional": True}
                ]
            }
        ],
        "api_calls": [
            {
                "name": "book_flight",
                "arguments": {
                    "origin": "Beijing",
                    "destination": "Shanghai",
                    "date": "2026-04-01",
                    "passengers": "2",
                    "extra_field": "should_warn"
                }
            }
        ],
        "sample_id": "case6_exec_warnings"
    },

    # ---- Case 7: 有 default 的必需参数缺失不报错 ----
    {
        "query": "Get stock price",
        "tools": [
            {
                "name": "get_stock",
                "description": "Get stock price",
                "parameters": [
                    {"name": "symbol", "type": "str", "required": True, "optional": False},
                    {"name": "exchange", "type": "str", "required": True, "optional": False,
                     "default": "NYSE"}
                ]
            }
        ],
        "api_calls": [
            {"name": "get_stock", "arguments": {"symbol": "AAPL"}}
        ],
        "sample_id": "case7_default_param"
    },

    # ---- Case 8: 带 metadata 和 response 的完整样本 ----
    {
        "query": "Check user profile",
        "tools": [
            {
                "name": "get_user",
                "description": "Fetch user profile",
                "parameters": [
                    {"name": "user_id", "type": "int", "required": True, "optional": False,
                     "metadata": {"example_value": 42}}
                ]
            }
        ],
        "api_calls": [
            {
                "name": "get_user",
                "arguments": {"user_id": 12345},
                "response": "{\"name\": \"Alice\", \"age\": 30}",
                "metadata": {"thought": "I need to look up the user first"}
            }
        ],
        "final_answer": "User Alice, age 30.",
        "sample_id": "case8_full_fields",
        "source_dataset": "custom_dataset",
        "metadata": {"custom_key": "custom_value"}
    },
]


# =============================================================================
# 测试运行
# =============================================================================

def run_tests():
    passed = 0
    failed = 0

    def assert_eq(test_name, actual, expected, msg=""):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL {test_name}: expected {expected}, got {actual}. {msg}")

    def assert_true(test_name, condition, msg=""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL {test_name}: {msg}")

    # =================================================================
    # Part 1: GeneralLoader
    # =================================================================
    print("=" * 70)
    print("Part 1: GeneralLoader")
    print("=" * 70)

    tmpfile = tempfile.NamedTemporaryFile(
        mode='w', suffix='.jsonl', delete=False, dir='/tmp')
    for item in SAMPLES:
        tmpfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    tmpfile.close()

    loader = GeneralLoader(tmpfile.name)
    samples = loader.parse_all(show_progress=False)

    assert_eq("loader_count", len(samples), len(SAMPLES))

    # Case 1: 字段完整性
    s = samples[0]
    assert_eq("c1_query", s.query, "What is the weather in Beijing?")
    assert_eq("c1_tools_count", len(s.tools), 1)
    assert_eq("c1_tool_name", s.tools[0].name, "get_weather")
    assert_eq("c1_params_count", len(s.tools[0].parameters), 2)
    assert_eq("c1_param0_name", s.tools[0].parameters[0].name, "city")
    assert_eq("c1_param0_type", s.tools[0].parameters[0].type, "str")
    assert_eq("c1_param0_required", s.tools[0].parameters[0].required, True)
    assert_eq("c1_param0_optional", s.tools[0].parameters[0].optional, False)
    assert_eq("c1_param1_default", s.tools[0].parameters[1].default, "celsius")
    assert_eq("c1_calls_count", len(s.api_calls), 1)
    assert_eq("c1_call_name", s.api_calls[0].name, "get_weather")
    assert_eq("c1_call_args", s.api_calls[0].arguments, {"city": "Beijing"})
    assert_eq("c1_final", s.final_answer, "The weather in Beijing is 25°C and sunny.")
    assert_eq("c1_sid", s.sample_id, "case1_clean")
    assert_eq("c1_src", s.source_dataset, "test")

    # Case 2: 多工具多调用
    s = samples[1]
    assert_eq("c2_tools_count", len(s.tools), 2)
    assert_eq("c2_calls_count", len(s.api_calls), 2)

    # Case 7: default 字段保留
    s = samples[6]
    assert_eq("c7_default", s.tools[0].parameters[1].default, "NYSE")

    # Case 8: metadata / response
    s = samples[7]
    assert_eq("c8_response", s.api_calls[0].response, '{"name": "Alice", "age": 30}')
    assert_eq("c8_call_meta", s.api_calls[0].metadata.get("thought"),
              "I need to look up the user first")
    assert_eq("c8_param_meta", s.tools[0].parameters[0].metadata.get("example_value"), 42)
    assert_eq("c8_sample_meta", s.metadata.get("custom_key"), "custom_value")
    assert_eq("c8_src", s.source_dataset, "custom_dataset")

    # JSON 数组格式也能读
    tmpfile2 = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir='/tmp')
    json.dump(SAMPLES[:2], tmpfile2, ensure_ascii=False)
    tmpfile2.close()

    loader2 = GeneralLoader(tmpfile2.name)
    samples2 = loader2.parse_all(show_progress=False)
    assert_eq("json_array_count", len(samples2), 2)
    os.unlink(tmpfile2.name)

    print(f"  GeneralLoader: {passed} passed")

    # =================================================================
    # Part 2: GeneralFormatChecker
    # =================================================================
    print("\n" + "=" * 70)
    print("Part 2: GeneralFormatChecker")
    print("=" * 70)

    fmt = GeneralFormatChecker()
    prev_passed = passed

    # Case 1: clean
    e, w = fmt.check(samples[0])
    assert_eq("c1_fmt_errors", len(e), 0)
    assert_eq("c1_fmt_warnings", len(w), 0)

    # Case 2: multi-tool clean
    e, w = fmt.check(samples[1])
    assert_eq("c2_fmt_errors", len(e), 0)

    # Case 3: warnings (empty description, required+optional, empty final_answer)
    e, w = fmt.check(samples[2])
    assert_eq("c3_fmt_errors", len(e), 0)
    assert_true("c3_has_warnings", len(w) >= 2,
                f"Expected >=2 warnings, got {len(w)}: {w}")
    assert_true("c3_desc_warn", any("description is empty" in x for x in w), f"Missing desc warn: {w}")
    assert_true("c3_mutex_warn", any("required=True AND optional=True" in x for x in w),
                f"Missing mutex warn: {w}")
    assert_true("c3_final_warn", any("final_answer" in x for x in w), f"Missing final_answer warn: {w}")

    # Case 4: error (nonexistent_api)
    e, w = fmt.check(samples[3])
    assert_true("c4_has_errors", len(e) > 0, f"Expected errors, got 0")
    assert_true("c4_not_found", any("nonexistent_api" in x for x in e),
                f"Expected 'nonexistent_api' in errors: {e}")

    # Case 7: default param, should be clean
    e, w = fmt.check(samples[6])
    assert_eq("c7_fmt_errors", len(e), 0)

    # Case 8: full fields, should be clean
    e, w = fmt.check(samples[7])
    assert_eq("c8_fmt_errors", len(e), 0)

    print(f"  GeneralFormatChecker: {passed - prev_passed} passed")

    # =================================================================
    # Part 3: GeneralExecutabilityChecker
    # =================================================================
    print("\n" + "=" * 70)
    print("Part 3: GeneralExecutabilityChecker")
    print("=" * 70)

    exc = GeneralExecutabilityChecker()
    prev_passed = passed

    # Case 1: clean
    e, w, st = exc.check(samples[0])
    assert_eq("c1_exec_errors", len(e), 0)
    assert_eq("c1_exec_warnings", len(w), 0)
    assert_eq("c1_exec_checked", st["api_calls_checked"], 1)

    # Case 2: multi-tool clean
    e, w, st = exc.check(samples[1])
    assert_eq("c2_exec_errors", len(e), 0)
    assert_eq("c2_exec_checked", st["api_calls_checked"], 2)

    # Case 5: errors (tool_not_found + missing_required + type_mismatch)
    e, w, st = exc.check(samples[4])
    assert_true("c5_has_errors", len(e) >= 3, f"Expected >=3 errors, got {len(e)}: {e}")
    assert_eq("c5_tool_not_found", st["tool_not_found"], 1)
    assert_eq("c5_missing_req", st["missing_required"], 2)
    assert_eq("c5_type_mismatch", st["type_mismatches"], 1)
    assert_true("c5_destination_err", any("destination" in x for x in e), f"Missing destination: {e}")
    assert_true("c5_date_err", any("date" in x for x in e), f"Missing date: {e}")
    assert_true("c5_type_err", any("not parseable" in x for x in e), f"Missing type err: {e}")

    # Case 6: warnings (unknown_param + parseable str->int)
    e, w, st = exc.check(samples[5])
    assert_eq("c6_exec_errors", len(e), 0)
    assert_true("c6_has_warnings", len(w) >= 2, f"Expected >=2 warnings, got {len(w)}: {w}")
    assert_eq("c6_unknown", st["unknown_params"], 1)
    assert_true("c6_unknown_warn", any("extra_field" in x for x in w), f"Missing extra_field: {w}")
    assert_true("c6_parseable_warn", any("parseable" in x for x in w), f"Missing parseable warn: {w}")

    # Case 7: default param 允许缺失
    e, w, st = exc.check(samples[6])
    assert_eq("c7_exec_errors", len(e), 0, f"Got errors: {e}")

    # Case 8: full fields clean
    e, w, st = exc.check(samples[7])
    assert_eq("c8_exec_errors", len(e), 0)

    print(f"  GeneralExecutabilityChecker: {passed - prev_passed} passed")

    # =================================================================
    # Part 4: Registry
    # =================================================================
    print("\n" + "=" * 70)
    print("Part 4: Registry")
    print("=" * 70)
    prev_passed = passed

    from api_executor import (
        get_format_checker,
        get_executability_checker,
        list_format_checkers,
        list_executability_checkers,
    )

    assert_true("reg_fmt", "general" in list_format_checkers(),
                f"'general' not in {list_format_checkers()}")
    assert_true("reg_exec", "general" in list_executability_checkers(),
                f"'general' not in {list_executability_checkers()}")

    reg_fmt = get_format_checker("general")
    assert_eq("reg_fmt_type", type(reg_fmt).__name__, "GeneralFormatChecker")

    reg_exec = get_executability_checker("general")
    assert_eq("reg_exec_type", type(reg_exec).__name__, "GeneralExecutabilityChecker")

    e, w = reg_fmt.check(samples[0])
    assert_eq("reg_fmt_clean", len(e), 0)

    print(f"  Registry: {passed - prev_passed} passed")

    # =================================================================
    # Cleanup & Summary
    # =================================================================
    os.unlink(tmpfile.name)

    print("\n" + "=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{failed} FAILED / {total} total")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
