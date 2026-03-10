#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeneralLoader / GeneralFormatChecker 测试

测试用例设计覆盖 data_types.py 中所有 dataclass 字段：

  Record
    ├── actions: List[Action]               → 空列表检测
    │     ├── action_idx: int               → 负值 / 重复 / 连续性
    │     ├── action_type: str              → 空值 / 已知类型 / 未知类型
    │     ├── action_value: str             → 默认空
    │     ├── action_repr: str              → 空值 (warning)
    │     ├── cleaned_html: str             → 空值 (warning) / HTML 定位
    │     ├── raw_html: Optional[str]
    │     ├── screenshot: Optional[str]
    │     ├── target_element: Any           → None / str / dict / 在 candidates 中
    │     ├── candidates: List[Any]         → str 列表 / dict 列表
    │     └── metadata: Dict
    ├── sample_id: Optional[str]            → 缺失 (warning)
    ├── instruction: Optional[str]          → 空字符串 (warning)
    ├── website: Optional[str]
    └── metadata: Dict

运行:
    cd text_gui_agent_eval
    python test_general.py
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_types import Action, Record
from loaders import GeneralLoader
from executor.general import GeneralFormatChecker


# =============================================================================
# 测试数据
# =============================================================================

SAMPLES = [
    # ---- Case 1: 完全正常的样本 (click + type) ----
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "click",
                "action_value": "",
                "action_repr": "[tab] Flights -> CLICK",
                "cleaned_html": '<html><button backend_node_id="42">Flights</button></html>',
                "raw_html": '<html><body><button backend_node_id="42">Flights</button></body></html>',
                "target_element": {"tag": "button", "backend_node_id": "42", "text": "Flights"},
                "candidates": [
                    {"tag": "button", "backend_node_id": "42", "text": "Flights"},
                    {"tag": "a", "backend_node_id": "55", "text": "Hotels"},
                ],
                "metadata": {"action_uid": "abc-123"}
            },
            {
                "action_idx": 1,
                "action_type": "type",
                "action_value": "New York",
                "action_repr": "[input] Origin -> TYPE: New York",
                "cleaned_html": '<html><input id="origin" placeholder="From"/></html>',
                "target_element": {"tag": "input", "id": "origin"},
                "candidates": [
                    {"tag": "input", "id": "origin"},
                    {"tag": "input", "id": "destination"},
                ],
                "metadata": {}
            }
        ],
        "sample_id": "case1_clean",
        "instruction": "Search for flights from NYC to LA",
        "website": "google.com",
        "metadata": {"annotation_id": "test-001"}
    },

    # ---- Case 2: WebShop 风格 (search + click, str candidates) ----
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "search",
                "action_value": "black ottoman",
                "action_repr": "search[black ottoman]",
                "cleaned_html": "[button] Search [button_]",
                "target_element": "search[black ottoman]",
                "candidates": [],
                "metadata": {}
            },
            {
                "action_idx": 1,
                "action_type": "click",
                "action_value": "",
                "action_repr": "click[Buy Now]",
                "cleaned_html": "[button] Buy Now [button_] [button] Back [button_]",
                "target_element": "click[Buy Now]",
                "candidates": ["click[Buy Now]", "click[Back]"],
                "metadata": {}
            }
        ],
        "sample_id": "case2_webshop_style",
        "instruction": "Buy a black ottoman",
        "website": "webshop",
        "metadata": {}
    },

    # ---- Case 3: WebLINX 风格 (say + click, str target) ----
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "say",
                "action_value": "I'll help you book a flight.",
                "action_repr": "say(utterance=\"I'll help you book a flight.\")",
                "cleaned_html": '<html><div data-webtasks-id="uid-1">Search</div></html>',
                "target_element": None,
                "candidates": [],
                "metadata": {"turn": 1}
            },
            {
                "action_idx": 1,
                "action_type": "click",
                "action_value": "",
                "action_repr": "click(uid=\"uid-1\")",
                "cleaned_html": '<html><div data-webtasks-id="uid-1">Search</div></html>',
                "target_element": "uid-1",
                "candidates": [
                    {"uid": "uid-1", "tag": "div", "text": "Search"},
                    {"uid": "uid-2", "tag": "a", "text": "Home"},
                ],
                "metadata": {"turn": 2}
            }
        ],
        "sample_id": "case3_weblinx_style",
        "instruction": None,
        "website": "momondo.in",
        "metadata": {"demo_id": "test-demo"}
    },

    # ---- Case 4: FormatChecker warnings ----
    # - instruction 是空字符串
    # - sample_id 缺失
    # - action_repr 为空
    # - cleaned_html 为空
    # - action_idx 不连续
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "click",
                "action_value": "",
                "action_repr": "",
                "cleaned_html": "",
                "target_element": "some_target",
                "candidates": ["some_target", "other"],
                "metadata": {}
            },
            {
                "action_idx": 5,
                "action_type": "scroll",
                "action_value": "y=100",
                "action_repr": "scroll(y=100)",
                "cleaned_html": "",
                "metadata": {}
            }
        ],
        "instruction": "",
        "metadata": {}
    },

    # ---- Case 5: FormatChecker errors ----
    # - action_type 为空
    # - 重复 action_idx
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "",
                "cleaned_html": "<html></html>",
                "metadata": {}
            },
            {
                "action_idx": 0,
                "action_type": "click",
                "cleaned_html": "<html></html>",
                "target_element": "btn",
                "candidates": ["btn"],
                "metadata": {}
            }
        ],
        "sample_id": "case5_format_errors",
        "instruction": "Test",
        "metadata": {}
    },

    # ---- Case 6: StaticChecker errors ----
    # - click 无 target_element (error)
    # - click target 不在 candidates 中 (error)
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "click",
                "action_repr": "click missing target",
                "cleaned_html": "<html><button>OK</button></html>",
                "target_element": None,
                "candidates": ["click[OK]"],
                "metadata": {}
            },
            {
                "action_idx": 1,
                "action_type": "select",
                "action_repr": "select wrong target",
                "cleaned_html": "<html><select><option>A</option></select></html>",
                "target_element": "option_B",
                "candidates": ["option_A", "option_C"],
                "metadata": {}
            }
        ],
        "sample_id": "case6_static_errors",
        "instruction": "Test static errors",
        "metadata": {}
    },

    # ---- Case 7: Locator 测试 (各种 target 类型) ----
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "click",
                "action_repr": "click by backend_node_id",
                "cleaned_html": '<html><button backend_node_id="136">Submit</button></html>',
                "target_element": {"tag": "button", "backend_node_id": "136"},
                "candidates": [],
                "metadata": {}
            },
            {
                "action_idx": 1,
                "action_type": "click",
                "action_repr": "click by id attr",
                "cleaned_html": '<html><div id="main-nav">Nav</div></html>',
                "target_element": {"tag": "div", "id": "main-nav"},
                "candidates": [],
                "metadata": {}
            },
            {
                "action_idx": 2,
                "action_type": "click",
                "action_repr": "click by webtasks-id",
                "cleaned_html": '<html><span data-webtasks-id="wt-99">Link</span></html>',
                "target_element": {"uid": "wt-99"},
                "candidates": [],
                "metadata": {}
            },
            {
                "action_idx": 3,
                "action_type": "say",
                "action_repr": "say something",
                "cleaned_html": "<html></html>",
                "target_element": None,
                "candidates": [],
                "metadata": {}
            },
            {
                "action_idx": 4,
                "action_type": "click",
                "action_repr": "click unfindable",
                "cleaned_html": "<html><p>Nothing here</p></html>",
                "target_element": {"tag": "button", "backend_node_id": "999"},
                "candidates": [],
                "metadata": {}
            }
        ],
        "sample_id": "case7_locator",
        "instruction": "Test locator strategies",
        "metadata": {}
    },

    # ---- Case 8: 带完整 metadata 和 raw_html 的样本 ----
    {
        "actions": [
            {
                "action_idx": 0,
                "action_type": "hover",
                "action_value": "",
                "action_repr": "[menu] Options -> HOVER",
                "cleaned_html": '<nav><ul><li id="opts">Options</li></ul></nav>',
                "raw_html": '<html><body><nav class="main"><ul><li id="opts">Options</li></ul></nav></body></html>',
                "screenshot": "/path/to/screenshot.png",
                "target_element": {"tag": "li", "id": "opts", "text": "Options"},
                "candidates": [
                    {"tag": "li", "id": "opts", "text": "Options"},
                    {"tag": "li", "id": "home", "text": "Home"},
                ],
                "metadata": {"action_uid": "hover-001", "custom": "value"}
            }
        ],
        "sample_id": "case8_full_fields",
        "instruction": "Hover over Options menu",
        "website": "example.com",
        "metadata": {"custom_key": "custom_value", "source": "manual"}
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

    # JSONL 格式
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
    assert_eq("c1_sid", s.sample_id, "case1_clean")
    assert_eq("c1_instr", s.instruction, "Search for flights from NYC to LA")
    assert_eq("c1_website", s.website, "google.com")
    assert_eq("c1_actions_count", len(s.actions), 2)
    assert_eq("c1_a0_type", s.actions[0].action_type, "click")
    assert_eq("c1_a0_idx", s.actions[0].action_idx, 0)
    assert_eq("c1_a0_repr", s.actions[0].action_repr, "[tab] Flights -> CLICK")
    assert_true("c1_a0_html", "backend_node_id" in s.actions[0].cleaned_html)
    assert_true("c1_a0_raw", s.actions[0].raw_html is not None)
    assert_true("c1_a0_target", isinstance(s.actions[0].target_element, dict))
    assert_eq("c1_a0_cands", len(s.actions[0].candidates), 2)
    assert_eq("c1_a0_meta", s.actions[0].metadata.get("action_uid"), "abc-123")
    assert_eq("c1_a1_type", s.actions[1].action_type, "type")
    assert_eq("c1_a1_val", s.actions[1].action_value, "New York")
    assert_eq("c1_meta", s.metadata.get("annotation_id"), "test-001")

    # Case 2: str candidates
    s = samples[1]
    assert_eq("c2_a1_target", s.actions[1].target_element, "click[Buy Now]")
    assert_eq("c2_a1_cands", s.actions[1].candidates, ["click[Buy Now]", "click[Back]"])

    # Case 3: null instruction preserved
    s = samples[2]
    assert_eq("c3_instr", s.instruction, None)
    assert_eq("c3_a0_target", s.actions[0].target_element, None)
    assert_eq("c3_a1_target", s.actions[1].target_element, "uid-1")

    # Case 4: auto sample_id when missing
    s = samples[3]
    assert_eq("c4_sid", s.sample_id, "general_3")

    # Case 8: full fields
    s = samples[7]
    assert_eq("c8_screenshot", s.actions[0].screenshot, "/path/to/screenshot.png")
    assert_eq("c8_meta", s.metadata.get("source"), "manual")

    # JSON 数组格式
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
    assert_eq("c1_fmt_errors", len(e), 0, f"errors: {e}")
    assert_eq("c1_fmt_warnings", len(w), 0, f"warnings: {w}")

    # Case 2: clean
    e, w = fmt.check(samples[1])
    assert_eq("c2_fmt_errors", len(e), 0, f"errors: {e}")

    # Case 3: instruction is None → no warning (only empty str triggers)
    e, w = fmt.check(samples[2])
    assert_eq("c3_fmt_errors", len(e), 0, f"errors: {e}")

    # Case 4: warnings (empty instruction, missing sample_id, empty action_repr, empty cleaned_html, non-sequential idx)
    e, w = fmt.check(samples[3])
    assert_eq("c4_fmt_errors", len(e), 0, f"errors: {e}")
    assert_true("c4_has_warnings", len(w) >= 4, f"Expected >=4 warnings, got {len(w)}: {w}")
    assert_true("c4_instr_warn", any("instruction" in x for x in w), f"Missing instruction warn: {w}")
    assert_true("c4_html_warn", any("cleaned_html" in x for x in w), f"Missing cleaned_html warn: {w}")
    assert_true("c4_repr_warn", any("action_repr" in x for x in w), f"Missing action_repr warn: {w}")
    assert_true("c4_seq_warn", any("sequential" in x for x in w), f"Missing sequential warn: {w}")

    # Case 5: errors (empty action_type, duplicate action_idx)
    e, w = fmt.check(samples[4])
    assert_true("c5_has_errors", len(e) >= 2, f"Expected >=2 errors, got {len(e)}: {e}")
    assert_true("c5_type_err", any("action_type" in x for x in e), f"Missing type err: {e}")
    assert_true("c5_dup_err", any("duplicate" in x for x in e), f"Missing duplicate err: {e}")

    # Case 8: clean
    e, w = fmt.check(samples[7])
    assert_eq("c8_fmt_errors", len(e), 0, f"errors: {e}")

    print(f"  GeneralFormatChecker: {passed - prev_passed} passed")

    # =================================================================
    # Part 3: Registry
    # =================================================================
    print("\n" + "=" * 70)
    print("Part 3: Registry")
    print("=" * 70)
    prev_passed = passed

    from text_gui_executor import (
        get_format_checker,
        list_format_checkers,
    )

    assert_true("reg_fmt", "general" in list_format_checkers(),
                f"'general' not in {list_format_checkers()}")

    reg_fmt = get_format_checker("general")
    assert_eq("reg_fmt_type", type(reg_fmt).__name__, "GeneralFormatChecker")

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
