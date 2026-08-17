#!/usr/bin/env python3
"""Run the Korean large-number unit conversion benchmark against local Ollama.

Four conversions are asked of the same ten numbers: digits → Korean units, Korean units → digits,
digits → three-digit commas (the control task, since it has nothing to do with Korean's four-digit
grouping), and won → truncated 만 units (the one office workers actually type). The magnitude
ladder runs 6 → 14 digits, with carry boundaries and skipped-unit traps placed on purpose.

Every item has exactly one right answer, produced by the same functions the scorer grades with,
so nothing here depends on a hand-written gold column.

This script is the write-origin for run.yaml. It preserves every prompt, raw response, API
metadata, and infrastructure retry rather than asking a person to fill evidence in later.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from numunit_score import TASKS, expected_answer, score_run_dir, task_input

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("numunit_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-numunit-20260816"
# The output contract is identical across tasks so that "answer on one line" is never the variable
# under test; only the conversion rule and the example change.
TAIL = "답만 한 줄로 출력하세요. 계산 과정, 설명, 인사말은 쓰지 마세요."
TASK_SPEC: dict[str, dict[str, str]] = {
    "digit_to_ko": {
        "label": "숫자",
        "head": "다음 숫자를 한국어 단위 표기로 바꾸세요.",
        "rules": ("- 조, 억, 만 단위로 끊어 쓰고, 단위 사이는 공백 한 칸을 둡니다.\n"
                  "- 값이 0인 단위는 생략합니다.\n"
                  "- 단위 앞 숫자는 아라비아 숫자로 씁니다.\n"
                  "- 예: 87654321 → 8765만 4321"),
    },
    "ko_to_digit": {
        "label": "표기",
        "head": "다음 한국어 단위 표기를 아라비아 숫자로 바꾸세요.",
        "rules": ("- 쉼표 없이 숫자만 씁니다.\n"
                  "- 예: 8765만 4321 → 87654321"),
    },
    "comma": {
        "label": "숫자",
        "head": "다음 숫자에 세 자리마다 쉼표를 찍으세요.",
        "rules": ("- 숫자와 쉼표만 씁니다.\n"
                  "- 예: 87654321 → 87,654,321"),
    },
    "won_round": {
        "label": "금액",
        "head": "다음 금액을 만 원 단위로 버림해서 쓰세요.",
        "rules": ("- 만 원 미만은 버립니다.\n"
                  "- '□만' 형태로 쓰고 숫자는 아라비아 숫자로 씁니다.\n"
                  "- 예: 987654원 → 98만"),
    },
}


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or []
    values = data.get("values") or []
    if list(tasks) != list(TASKS):
        raise ValueError(f"tasks 계약 불일치: {tasks} != {list(TASKS)}")
    if len(values) != 10:
        raise ValueError(f"values {len(values)}건 != 10")
    ids: set[str] = set()
    for item in values:
        item_id = item.get("id")
        if not item_id or item_id in ids:
            raise ValueError(f"중복/빈 value id: {item_id}")
        ids.add(item_id)
        value = item.get("value")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{item_id} value는 양의 정수")
        if len(str(value)) != int(item.get("digits", 0)):
            raise ValueError(f"{item_id} digits={item.get('digits')}가 실제 자릿수와 불일치")
        # Fail here rather than mid-run if a value falls outside what the formatter can express.
        for task in TASKS:
            expected_answer(task, value)
    return data


def build_prompt(task: str, value: int) -> str:
    spec = TASK_SPEC[task]
    return (
        f"{spec['head']}\n"
        f"{spec['label']}: {task_input(task, value)}\n\n"
        f"규칙:\n{spec['rules']}\n\n"
        f"{TAIL}\n"
    )


def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("API 응답 최상위 값이 JSON 객체가 아님")
    return decoded


def model_metadata(base_url: str, model: str, timeout: int) -> dict[str, Any]:
    try:
        return api_json(base_url.rstrip("/") + "/api/show", {"model": model}, timeout)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"모델 메타데이터 조회 실패: {exc}") from exc


def _payload(model: str, prompt: str, num_ctx: int) -> dict[str, Any]:
    """Request body. `num_ctx` is explicit because a thinking model can spend the whole
    default window on reasoning and return an empty answer — a harness fault, not a result."""
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0}
    if num_ctx:
        body["options"] = {"num_ctx": num_ctx}
    return body


def generate(base_url: str, model: str, prompt: str, timeout: int, num_ctx: int = 0) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    payload = _payload(model, prompt, num_ctx)
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": None})
            return response, attempts, None
        except urllib.error.HTTPError as exc:
            error = f"HTTPError {exc.code}: {exc.reason}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            transient = exc.code in {408, 425, 429} or exc.code >= 500
            if not transient or attempt == 2:
                return None, attempts, error
            time.sleep(1)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            if attempt == 2:
                return None, attempts, error
            time.sleep(1)
    return None, attempts, attempts[-1]["error"]


def planned_runs(data: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    values = data["values"]
    if mode == "pilot":
        # One value per magnitude extreme across all four tasks — enough to prove the answer
        # extraction works on every output shape before spending the full grid.
        chosen = [values[0], values[-1]]
    else:
        chosen = values
    runs: list[dict[str, Any]] = []
    for item in chosen:
        for task in TASKS:
            runs.append({"key": f"{item['id']}/{task}", "item": item, "task": task})
    return runs


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys: set[str] = set()
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        keys.add(row["key"])
        maximum = max(maximum, int(row["run_id"]))
    return keys, maximum


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compare_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Task- and magnitude-wise accuracy read back from the scored aggregate, never hand-written."""
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for task, entry in (aggregate.get("by_task") or {}).items():
        rows.append({"metric": "accuracy_by_task", "arm": task, "value": entry["accuracy"],
                     "correct": entry["correct"], "runs": entry["runs"]})
    for digits, entry in (aggregate.get("by_digits") or {}).items():
        rows.append({"metric": "accuracy_by_digits", "arm": f"{digits}자리", "value": entry["accuracy"],
                     "correct": entry["correct"], "runs": entry["runs"]})
    return rows


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any], data: dict[str, Any],
                   num_ctx: int = 0) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "item_id": row["item_id"],
            "task": row["task"],
            "digits": row["digits"],
            "value": row["value"],
            "expected": row["expected"],
            "output_file": row["transcript_file"],
            "response_file": row["response_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_s"),
            "prompt_eval_count": (row.get("api_metrics") or {}).get("prompt_eval_count"),
            "eval_count": (row.get("api_metrics") or {}).get("eval_count"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": "NUMUNIT on N06a..N14b (tasks digit_to_ko/ko_to_digit/comma/won_round, 6~14자리)",
        "access": "local",
        "model": model,
        "generated_by": "run_numunit_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "num_ctx": num_ctx or None,
        "request_parallelism": 1,
        "axis": "korean_number_unit_conversion",
        "tasks": list(TASKS),
        "compare": _compare_rows(run_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "ollama_models": os.environ.get("OLLAMA_MODELS"),
        },
        "model_metadata": {
            "modified_at": metadata.get("modified_at"),
            "details": metadata.get("details"),
            "parameters": metadata.get("parameters"),
        },
        "runs": entries,
    }
    target_path = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target_path)


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    plan = planned_runs(data, args.mode)
    if not plan:
        raise ValueError("실행 계획이 비어 있음")
    if len({row["key"] for row in plan}) != len(plan):
        raise ValueError("실행 계획에 중복 키 존재")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "samples": [
            {"key": row["key"], "prompt": build_prompt(row["task"], row["item"]["value"]),
             "expected": expected_answer(row["task"], row["item"]["value"])}
            for row in plan[:4]
        ]}, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "full":
        decision_path = args.run_dir / "pilot_decision.json"
        if not decision_path.exists():
            print("파일럿 판정 없음 — 같은 --run-dir에서 --mode pilot을 먼저 완료하세요.")
            return 3
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not isinstance(decision, dict) or not decision.get("proceed"):
            reasons = decision.get("reasons") if isinstance(decision, dict) else ["invalid_pilot_decision"]
            print(f"파일럿 중단조건 발동: {reasons}")
            return 3

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    existing_keys, next_id = _existing(args.run_dir)
    pending = [row for row in plan if row["key"] not in existing_keys]
    pending_total = len(pending)
    if args.max_new > 0:
        pending = pending[:args.max_new]
    chunk_limited = len(pending) < pending_total
    print(f"mode={args.mode} planned={len(plan)} existing={len(plan) - pending_total} "
          f"pending_total={pending_total} running_now={len(pending)}")

    if pending:
        metadata = model_metadata(args.base_url, args.model, args.timeout)
    elif (args.run_dir / "run.yaml").exists():
        previous = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
        metadata = previous.get("model_metadata") or {}
    else:
        metadata = {}

    for index, row in enumerate(pending, 1):
        next_id += 1
        item = row["item"]
        prompt_text = build_prompt(row["task"], item["value"])
        response, attempts, infra_error = generate(args.base_url, args.model, prompt_text, args.timeout, args.num_ctx)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        stem = f"{next_id:03d}"
        response_rel = f"raw/{stem}-response.txt"
        transcript_rel = f"raw/{stem}-output.txt"
        invocation_rel = f"raw/{stem}-invocation.json"
        expected = expected_answer(row["task"], item["value"])
        (args.run_dir / response_rel).write_text(response_text, encoding="utf-8")
        transcript = (
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {args.model}\n"
            f"item_id: {item['id']}\ntask: {row['task']}\ndigits: {item['digits']}\n"
            f"value: {item['value']}\nexpected: {expected}\n\n"
            f"[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
            f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "item_id": item["id"],
            "task": row["task"],
            "digits": item["digits"],
            "value": item["value"],
            "note": item.get("note"),
            "expected": expected,
            "prompt": prompt_text,
            "payload": {k: v for k, v in _payload(args.model, prompt_text, args.num_ctx).items() if k != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item_row["elapsed_s"] for item_row in attempts), 3),
            "infra_error": infra_error,
            "response_file": response_rel,
            "transcript_file": transcript_rel,
            "api_metrics": {key: response.get(key) for key in (
                "done_reason", "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration"
            )} if response else None,
        }
        (args.run_dir / invocation_rel).write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_run_yaml(args.run_dir, args.model, metadata, data, args.num_ctx)
        print(f"[{index}/{len(pending)}] {row['key']} elapsed={invocation['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    write_run_yaml(args.run_dir, args.model, metadata, data, args.num_ctx)
    results, aggregate = score_run_dir(args.run_dir)
    # Scoring writes aggregate.json, which is where the compare rows come from — so the
    # manifest is rewritten once more to carry them.
    write_run_yaml(args.run_dir, args.model, metadata, data, args.num_ctx)
    infra_count = sum(bool(row.get("infra_error")) for row in results)
    limit = 0.10 if args.mode == "pilot" else 0.05
    status = {
        "mode": args.mode,
        "runs": len(results),
        "infra_errors": infra_count,
        "infra_rate": infra_count / len(results) if results else 1.0,
        "infra_limit": limit,
        "complete": len(results) == len(plan),
        "chunk_limited": chunk_limited,
        "pilot_decision": aggregate.get("pilot_decision"),
    }
    (args.run_dir / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["infra_rate"] > limit:
        return 4
    if args.mode == "pilot":
        if not status["complete"] or not isinstance(status["pilot_decision"], dict):
            return 4
        if not status["pilot_decision"].get("proceed"):
            return 3
        return 0
    if status["chunk_limited"] and not status["complete"]:
        return 0
    if not status["complete"]:
        return 4
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-new", type=int, default=0,
                        help="이번 호출에서 새로 실행할 최대 시행 수(0=전부, 긴 런의 안전한 청크용)")
    parser.add_argument("--num-ctx", type=int, default=0,
                        help="컨텍스트 창(0=ollama 기본값). 사고 출력이 긴 모델은 기본값에서 답이 잘린다")
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
