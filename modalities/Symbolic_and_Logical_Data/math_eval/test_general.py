#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeneralLoader / GeneralFormatChecker 测试

测试用例设计：
  c1 — 完整 CoT 样本，全字段，0 error 0 warning
  c2 — 完整 Code 样本（solution 为 List[str]），带 GT list
  c3 — 缺少 ground_truth / sample_id / source_dataset → 3 warnings
  c4 — question 为空 → error
  c5 — solution 为空字符串 → error
  c6 — solution 为空列表 → error
  c7 — solution 列表中含非 str 元素 → error
  c8 — solution 类型错误 (int) → error
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loaders import GeneralLoader
from executor.general import GeneralFormatChecker
from data_types import MathSample


SAMPLES = [
    {
        "question": "What is 2 + 3?",
        "solution": "We know 2 + 3 = 5. The answer is: 5",
        "ground_truth": "5",
        "sample_id": "c1_clean",
        "source_dataset": "test",
        "question_type": "arithmetic",
        "metadata": {"difficulty": "easy"}
    },
    {
        "question": "Solve x^2 = 9",
        "solution": ["import sympy\nx = sympy.Symbol('x')\nprint(sympy.solve(x**2 - 9, x))"],
        "ground_truth": [3, -3],
        "sample_id": "c2_code",
        "source_dataset": "test",
        "question_type": "algebra"
    },
    {
        "question": "What is 10 / 2?",
        "solution": "10 / 2 = 5"
    },
    {
        "question": "",
        "solution": "The answer is 42",
        "ground_truth": "42",
        "sample_id": "c4_empty_q",
        "source_dataset": "test"
    },
    {
        "question": "What is 1+1?",
        "solution": "",
        "ground_truth": "2",
        "sample_id": "c5_empty_sol",
        "source_dataset": "test"
    },
    {
        "question": "What is 1+1?",
        "solution": [],
        "ground_truth": "2",
        "sample_id": "c6_empty_list",
        "source_dataset": "test"
    },
    {
        "question": "What is 1+1?",
        "solution": ["valid code", 123],
        "ground_truth": "2",
        "sample_id": "c7_bad_elem",
        "source_dataset": "test"
    },
    {
        "question": "What is 1+1?",
        "solution": 42,
        "ground_truth": "2",
        "sample_id": "c8_bad_type",
        "source_dataset": "test"
    },
]


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_json(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def test_loader(fmt='jsonl'):
    print(f"\n{'='*60}")
    print(f"  GeneralLoader 测试 ({fmt})")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{fmt}', delete=False) as f:
        tmp_path = f.name

    try:
        if fmt == 'jsonl':
            write_jsonl(tmp_path, SAMPLES)
        else:
            write_json(tmp_path, SAMPLES)

        loader = GeneralLoader(tmp_path)
        samples = loader.load()

        print(f"  Loaded {len(samples)} samples")
        assert len(samples) == len(SAMPLES), f"Expected {len(SAMPLES)}, got {len(samples)}"

        s0 = samples[0]
        assert isinstance(s0, MathSample)
        assert s0.question == "What is 2 + 3?"
        assert s0.solution == "We know 2 + 3 = 5. The answer is: 5"
        assert s0.ground_truth == "5"
        assert s0.sample_id == "c1_clean"
        assert s0.source_dataset == "test"
        assert s0.metadata.get("difficulty") == "easy"
        print("  c1: fields correct ✓")

        s1 = samples[1]
        assert isinstance(s1.solution, list)
        assert len(s1.solution) == 1
        assert isinstance(s1.ground_truth, list)
        print("  c2: list solution + list GT correct ✓")

        s2 = samples[2]
        assert s2.ground_truth is None
        assert s2.sample_id == "general_2"
        print("  c3: missing optional fields, auto sample_id ✓")

        print(f"\n  GeneralLoader ({fmt}): ALL PASSED ✓")
    finally:
        os.unlink(tmp_path)


def test_format_checker():
    print(f"\n{'='*60}")
    print(f"  GeneralFormatChecker 测试")
    print(f"{'='*60}")

    checker = GeneralFormatChecker()

    s1 = MathSample(question="What is 2+3?", solution="2+3=5. The answer is: 5",
                     ground_truth="5", sample_id="c1", source_dataset="test")
    errs, warns = checker.check(s1)
    assert len(errs) == 0 and len(warns) == 0
    print("  c1 (clean):      0 errors, 0 warnings ✓")

    s2 = MathSample(question="Solve x^2=9",
                     solution=["import sympy\nprint(sympy.solve(x**2-9))"],
                     ground_truth=[3, -3], sample_id="c2", source_dataset="test")
    errs, warns = checker.check(s2)
    assert len(errs) == 0 and len(warns) == 0
    print("  c2 (code/list):  0 errors, 0 warnings ✓")

    s3 = MathSample(question="What is 10/2?", solution="10/2=5")
    errs, warns = checker.check(s3)
    assert len(errs) == 0 and len(warns) == 3
    print(f"  c3 (missing opt): 0 errors, 3 warnings ✓  {warns}")

    s4 = MathSample(question="", solution="answer is 42", ground_truth="42",
                     sample_id="c4", source_dataset="test")
    errs, warns = checker.check(s4)
    assert len(errs) == 1 and "question" in errs[0].lower()
    print(f"  c4 (empty q):    1 error ✓  {errs}")

    s5 = MathSample(question="What?", solution="", ground_truth="2",
                     sample_id="c5", source_dataset="test")
    errs, warns = checker.check(s5)
    assert len(errs) == 1 and "solution" in errs[0].lower()
    print(f"  c5 (empty sol):  1 error ✓  {errs}")

    s6 = MathSample(question="What?", solution=[], ground_truth="2",
                     sample_id="c6", source_dataset="test")
    errs, warns = checker.check(s6)
    assert len(errs) == 1
    print(f"  c6 (empty list): 1 error ✓  {errs}")

    s7 = MathSample(question="What?", solution=["valid", 123], ground_truth="2",
                     sample_id="c7", source_dataset="test")
    errs, warns = checker.check(s7)
    assert len(errs) == 1 and "solution[1]" in errs[0]
    print(f"  c7 (bad elem):   1 error ✓  {errs}")

    s8 = MathSample(question="What?", solution=42, ground_truth="2",
                     sample_id="c8", source_dataset="test")
    errs, warns = checker.check(s8)
    assert len(errs) == 1 and "unexpected type" in errs[0].lower()
    print(f"  c8 (bad type):   1 error ✓  {errs}")

    s9 = MathSample(question="Q?", solution="sol", ground_truth="  ",
                     sample_id="c9", source_dataset="test")
    errs, warns = checker.check(s9)
    assert len(errs) == 0 and any("ground_truth" in w for w in warns)
    print(f"  c9 (empty GT):   0 errors, warning ✓  {warns}")

    s10 = MathSample(question="Q?", solution="sol", ground_truth=[],
                      sample_id="c10", source_dataset="test")
    errs, warns = checker.check(s10)
    assert len(errs) == 0 and any("ground_truth" in w for w in warns)
    print(f"  c10 (empty GT[]): 0 errors, warning ✓  {warns}")

    print(f"\n  GeneralFormatChecker: ALL PASSED ✓")


def test_integration():
    print(f"\n{'='*60}")
    print(f"  集成测试: Loader → FormatChecker")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = f.name

    try:
        write_jsonl(tmp_path, SAMPLES)
        loader = GeneralLoader(tmp_path)
        samples = loader.load()
        checker = GeneralFormatChecker()

        total_errors = 0
        total_warnings = 0
        for s in samples:
            errs, warns = checker.check(s)
            total_errors += len(errs)
            total_warnings += len(warns)
            status = "ERROR" if errs else ("WARN" if warns else "OK")
            sid = s.sample_id or "(no id)"
            print(f"  [{status:>5}] {sid}: {len(errs)} errors, {len(warns)} warnings")
            if errs:
                for e in errs:
                    print(f"         ✗ {e}")
            if warns:
                for w in warns:
                    print(f"         ⚠ {w}")

        print(f"\n  Total: {len(samples)} samples, {total_errors} errors, {total_warnings} warnings")
        assert total_errors > 0
        assert total_warnings > 0
        print(f"  Integration: PASSED ✓")
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    test_loader('jsonl')
    test_loader('json')
    test_format_checker()
    test_integration()
    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'='*60}")
