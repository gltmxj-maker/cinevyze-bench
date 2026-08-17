#!/usr/bin/env python3
"""Run the JSON/CSV/Markdown parsing benchmark against local Ollama.

The script is the write-origin for run.yaml. It preserves every prompt, raw response,
API metadata, and infrastructure retry rather than asking a person to fill evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from format_score import score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("format_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-format-20260729"
FORMAT_ROTATIONS = (
    ("json", "csv", "markdown"),
    ("csv", "markdown", "json"),
    ("markdown", "json", "csv"),
)


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or {}
    if not isinstance(tasks, dict) or set(tasks) != {"FMT-01", "FMT-02", "FMT-03"}:
        raise ValueError("cases에는 FMT-01/02/03 세 작업이 정확히 있어야 함")
    seen: set[str] = set()
    field_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for task_id, task in tasks.items():
        if not isinstance(task, dict) or not str(task.get("name", "")).strip():
            raise ValueError(f"{task_id} 작업 이름/객체 계약 오류")
        fields = task.get("fields") or []
        cases = task.get("cases") or []
        if len(cases) != 20:
            raise ValueError(f"{task_id} 사례 수 {len(cases)} != 20")
        if len(fields) != len(set(fields)) or "id" not in fields or not all(field_re.fullmatch(f) for f in fields):
            raise ValueError(f"{task_id} 필드 계약 오류")
        for case in cases:
            case_id = case.get("id")
            expected = case.get("expected")
            if not case_id or case_id in seen:
                raise ValueError(f"중복/빈 case id: {case_id}")
            seen.add(case_id)
            if not isinstance(expected, dict) or set(expected) != set(fields):
                raise ValueError(f"{case_id} expected 필드가 스키마와 다름")
            if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in expected.values()):
                raise ValueError(f"{case_id} expected 값은 문자열/정수 스칼라만 허용")
            if expected.get("id") != case_id or not str(case.get("input", "")).strip():
                raise ValueError(f"{case_id} id/input 계약 오류")
    return data


def _type_contract(expected: dict[str, Any], fields: list[str]) -> str:
    parts = []
    for field in fields:
        kind = "정수" if isinstance(expected[field], int) and not isinstance(expected[field], bool) else "문자열"
        parts.append(f"{field}({kind})")
    return ", ".join(parts)


def build_prompt(task_id: str, task_name: str, case: dict[str, Any], fields: list[str], output_format: str) -> str:
    expected = case["expected"]
    task_rules = {
        "FMT-01": (
            "- category 값은 다음 다섯 문자열 중 정확히 하나만 사용하세요: 환불, 배송, 교환, 상품문의, 기타.\n"
            "- '환불 요청'처럼 설명을 늘이거나 영어 라벨로 바꾸지 마세요.\n"
        ),
        "FMT-02": (
            "- vendor와 item은 입력의 고유 이름만 옮기고 수량·단가·조사·'업체' 같은 설명을 붙이지 마세요.\n"
            "- quantity와 unit_price는 단위·쉼표 없는 정수로 출력하세요.\n"
            "- due_date는 입력 표기와 무관하게 YYYY-MM-DD로 바꾸세요.\n"
        ),
        "FMT-03": (
            "- product에는 모델 코드까지 포함하되 색상·규격·재질 설명은 붙이지 마세요.\n"
            "- color와 material은 핵심 명칭만 쓰고 '색상', '소재', '재질' 같은 말을 붙이지 마세요.\n"
            "- in_stock은 재고가 있으면 Y, 없거나 품절이면 N만 출력하세요.\n"
        ),
    }
    shared = (
        "당신은 자동화 파이프라인의 구조화 데이터 변환기입니다.\n"
        f"작업: {task_name}\n"
        f"입력 ID: {case['id']}\n"
        f"입력: {case['input']}\n\n"
        "규칙:\n"
        "- 입력에 있는 정보만 사용하세요. 추측하거나 설명을 보태지 마세요.\n"
        f"- 필드는 정확히 다음 순서/이름을 사용하세요: {', '.join(fields)}\n"
        f"- 값 형식: {_type_contract(expected, fields)}\n"
        + task_rules[task_id]
        + "- 결과물 하나만 출력하세요. 머리말, 꼬리말, 설명 문장은 금지합니다.\n"
        "- 결과물 전체를 코드펜스 하나로 감싸는 것은 허용하지만 필수는 아닙니다.\n"
    )
    if output_format == "json":
        contract = (
            "- 유효한 JSON 객체 하나로 출력하세요. 배열은 금지합니다.\n"
            "- 큰따옴표를 쓰고 trailing comma를 넣지 마세요.\n"
            "- 추가 키를 만들지 마세요.\n"
        )
    elif output_format == "csv":
        contract = (
            "- CSV로 출력하세요. 정확히 헤더 1행과 데이터 1행만 출력하세요.\n"
            f"- 헤더는 정확히 다음과 같아야 합니다: {','.join(fields)}\n"
            "- 추가 행과 추가 열을 만들지 마세요.\n"
        )
    elif output_format == "markdown":
        header = "| " + " | ".join(fields) + " |"
        separator = "| " + " | ".join("---" for _ in fields) + " |"
        contract = (
            "- GFM 마크다운 표 하나로 출력하세요. 정확히 헤더·구분선·데이터 3행만 출력하세요.\n"
            f"- 첫 두 행은 정확히 다음 구조를 따르세요:\n{header}\n{separator}\n"
            "- 추가 행과 추가 열을 만들지 마세요.\n"
        )
    else:
        raise ValueError(output_format)
    return shared + contract


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


def generate(base_url: str, model: str, prompt: str, timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0}
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


def _selected_cases(data: dict[str, Any], mode: str) -> list[tuple[str, dict[str, Any], list[str], str]]:
    selected = []
    for task_id in sorted(data["tasks"]):
        task = data["tasks"][task_id]
        cases = task["cases"][:3] if mode == "pilot" else task["cases"]
        for case in cases:
            selected.append((task_id, case, task["fields"], task["name"]))
    return selected


def planned_runs(data: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index, (task_id, case, fields, task_name) in enumerate(_selected_cases(data, mode)):
        for output_format in FORMAT_ROTATIONS[index % len(FORMAT_ROTATIONS)]:
            runs.append({"key": f"{task_id}/{case['id']}/{output_format}", "task": task_id,
                         "task_name": task_name, "case": case, "fields": fields, "format": output_format})
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


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any]) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "task": row["task"],
            "case_id": row["case_id"],
            "format": row["format"],
            "output_file": row["transcript_file"],
            "response_file": row["response_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_s"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": "FMT-01,FMT-02,FMT-03",
        "access": "local",
        "model": model,
        "generated_by": "run_format_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
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
    # JSON is valid YAML; atomic replace prevents a killed process leaving a truncated manifest.
    target = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    plan = planned_runs(data, args.mode)
    if not plan:
        raise ValueError("실행 계획이 비어 있음")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": {
            "key": plan[0]["key"],
            "prompt": build_prompt(plan[0]["task"], plan[0]["task_name"], plan[0]["case"], plan[0]["fields"], plan[0]["format"]),
        }}, ensure_ascii=False, indent=2))
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

    for position, row in enumerate(pending, 1):
        next_id += 1
        prompt = build_prompt(row["task"], row["task_name"], row["case"], row["fields"], row["format"])
        response, attempts, infra_error = generate(args.base_url, args.model, prompt, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        stem = f"{next_id:03d}"
        response_rel = f"raw/{stem}-response.txt"
        transcript_rel = f"raw/{stem}-output.txt"
        invocation_rel = f"raw/{stem}-invocation.json"
        (args.run_dir / response_rel).write_text(response_text, encoding="utf-8")
        transcript = (
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {args.model}\n"
            f"task: {row['task']}\nformat: {row['format']}\n\n"
            f"[PROMPT]\n{prompt}\n\n[RAW RESPONSE]\n{response_text}\n"
            f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "task": row["task"],
            "case_id": row["case"]["id"],
            "format": row["format"],
            "fields": row["fields"],
            "expected": row["case"]["expected"],
            "prompt": prompt,
            "payload": {"model": args.model, "stream": False, "keep_alive": 0},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
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
        write_run_yaml(args.run_dir, args.model, metadata)
        print(f"[{position}/{len(pending)}] {row['key']} elapsed={invocation['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    write_run_yaml(args.run_dir, args.model, metadata)
    results, aggregate = score_run_dir(args.run_dir)
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
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-new", type=int, default=0,
                        help="이번 호출에서 새로 실행할 최대 시행 수(0=전부, 긴 런의 안전한 청크용)")
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
