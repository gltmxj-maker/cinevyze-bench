#!/usr/bin/env python3
"""27번 확장 코딩 과제의 고정 사례 실행 채점기."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = plan.get("tasks") or []
    if (plan.get("repetitions") != 3 or plan.get("models") != ["gemma3:4b", "qwen2.5-coder:7b"]
            or len(tasks) < 3 or len({task["id"] for task in tasks}) != len(tasks)
            or not all(task.get("prompt") and task.get("function") and task.get("cases") for task in tasks)):
        raise ValueError("고정 사례 계획 형식 불일치")
    for task in tasks:
        if len({case["id"] for case in task["cases"]}) != len(task["cases"]):
            raise ValueError(f"사례 ID 중복: {task['id']}")
    return plan


def extract_code(response: str) -> tuple[str | None, str]:
    blocks = re.findall(r"\x60{3}(?:python|py)?\s*\n(.*?)\x60{3}", response,
                        flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(blocks), "fenced_blocks"
    try:
        ast.parse(response)
        return response, "whole_response"
    except SyntaxError:
        return None, "unparseable"


def _execute(code: str, function: str, args: list) -> tuple[object, str | None, bool, bool, str | None]:
    driver = (
        "import copy, json\n" + code + "\n"
        "_args = " + repr(args) + "\n"
        "_before = copy.deepcopy(_args)\n"
        "try:\n"
        "    _value = " + function + "(*_args)\n"
        "    _same = any(_value is _arg for _arg in _args)\n"
        "    _data = {'actual': _value, 'actual_type': type(_value).__name__, "
        "'same_input_object': _same, 'mutated': _args != _before, 'error': None}\n"
        "except Exception as _exc:\n"
        "    _data = {'actual': None, 'actual_type': None, 'same_input_object': False, "
        "'mutated': _args != _before, "
        "'error': type(_exc).__name__ + ': ' + str(_exc)[:120]}\n"
        "print('CODE_EXPAND_RESULT=' + json.dumps(_data, ensure_ascii=False, default=repr))\n"
    )
    with tempfile.TemporaryDirectory() as temp:
        script = Path(temp) / "candidate.py"
        script.write_text(driver, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", "-S", str(script)], cwd=temp,
                                  text=True, capture_output=True, timeout=4)
        except subprocess.TimeoutExpired:
            return None, None, False, False, "timeout"
    if proc.returncode:
        return None, None, False, False, (proc.stderr.strip().splitlines()[-1][:160]
                             if proc.stderr.strip() else f"exit {proc.returncode}")
    markers = [line.removeprefix("CODE_EXPAND_RESULT=") for line in proc.stdout.splitlines()
               if line.startswith("CODE_EXPAND_RESULT=")]
    if not markers:
        return None, None, False, False, "result marker absent"
    try:
        value = json.loads(markers[-1])
    except json.JSONDecodeError:
        return None, None, False, False, "invalid result JSON"
    if not isinstance(value, dict):
        return None, None, False, False, "invalid result object"
    return (value.get("actual"), value.get("actual_type"), bool(value.get("same_input_object")),
            bool(value.get("mutated")), value.get("error"))


def score_response(task: dict, response: str) -> dict:
    code, extraction = extract_code(response)
    syntax_valid = False
    defines_function = False
    if code is not None:
        try:
            tree = ast.parse(code)
            syntax_valid = True
            defines_function = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                   and node.name == task["function"] for node in tree.body)
        except SyntaxError:
            pass
    results = []
    for case in task["cases"]:
        actual, actual_type, same_input_object, mutated, error = (None, None, False, False, "code unavailable")
        if syntax_valid and defines_function:
            actual, actual_type, same_input_object, mutated, error = _execute(code, task["function"], case["args"])
        elif syntax_valid:
            error = "required function absent"
        passed = (error is None and actual_type == type(case["expected"]).__name__
                  and actual == case["expected"]
                  and (not task["require_no_mutation"] or (not mutated and not same_input_object)))
        results.append({"case_id": case["id"], "expected": case["expected"],
                        "actual": actual, "actual_type": actual_type,
                        "same_input_object": same_input_object,
                        "mutated": mutated, "error": error, "passed": passed})
    example_shown = (not task.get("example_required")
                     or bool(re.search(r"\[\s*3\s*,\s*2\s*,\s*1\s*\]", response)))
    first_failure = next((case for case in results if not case["passed"]), None)
    return {"task": task["id"], "code_extractable": code is not None,
            "extraction_mode": extraction, "syntax_valid": syntax_valid,
            "defines_function": defines_function, "example_output_shown": example_shown,
            "cases": results, "cases_passed": sum(case["passed"] for case in results),
            "cases_total": len(results), "first_failure": first_failure,
            "contract_pass": bool(syntax_valid and defines_function and not first_failure
                                  and example_shown)}
