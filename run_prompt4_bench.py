#!/usr/bin/env python3
"""Run the prompt-4-part (V vs F) benchmark against local Ollama — 2026-09-21 재제작 10번.

Design contract (집필 서브 설계 2026-09-21, b-seat-split 위임):
- 18 runs: 3 tasks x 2 arms x 3 repetitions, rotated so no task or arm always sits at the
  same position (order-effect defence). Every call is an independent single turn.
- Exact prompt reuse: the six inputs are the 2026-07-26 originals, byte-identical. No pilot
  generation, no warm-up generation — the title counts exactly 18 generation calls.
- Two arms have different denominators: V = minimal artifact (x/9 outputs), F = explicit
  requirements from the F prompt's [출력형식] only (y/42) plus fully-compliant outputs (z/9).
  The two rates are never merged into one improvement ratio.
- Markers: turn on task/arm switch, break on FIRST failure per arm (max 2), infra per event,
  agg once at the end with values read from the computed aggregate — never hand-typed.
- keep_alive=0 on every request (VRAM 상주 금지).
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
from collections import Counter
from pathlib import Path
from typing import Any

from live_mark import mark as live_mark
from prompt4_score import score_one, score_run_dir, score_legacy_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("prompt4_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-prompt4-20260921"
LEGACY_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3:4b-20260726"

# 회전: 각 과업이 초반·중반·후반에 한 번씩, 세 반복 중 한 번은 F가 먼저 온다.
PLAN: list[tuple[str, str, int]] = [
    ("PRM-01", "V", 1), ("PRM-01", "F", 1),
    ("PRM-02", "V", 1), ("PRM-02", "F", 1),
    ("PRM-03", "V", 1), ("PRM-03", "F", 1),
    ("PRM-02", "F", 2), ("PRM-02", "V", 2),
    ("PRM-03", "F", 2), ("PRM-03", "V", 2),
    ("PRM-01", "F", 2), ("PRM-01", "V", 2),
    ("PRM-03", "V", 3), ("PRM-03", "F", 3),
    ("PRM-01", "V", 3), ("PRM-01", "F", 3),
    ("PRM-02", "V", 3), ("PRM-02", "F", 3),
]


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or {}
    if set(tasks) != {"PRM-01", "PRM-02", "PRM-03"}:
        raise ValueError("tasks는 PRM-01/02/03 세 과업이 정확히 있어야 함")
    for task_id, task in tasks.items():
        arms = task.get("arms") or {}
        if set(arms) != {"V", "F"}:
            raise ValueError(f"{task_id} arms는 V/F 둘")
        for arm, prompt in arms.items():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{task_id}/{arm} 프롬프트 비어 있음")
    expected = Counter({(f"PRM-0{i}", a): 3 for i in (1, 2, 3) for a in ("V", "F")})
    if Counter((t, a) for t, a, _ in PLAN) != expected:
        raise ValueError("PLAN은 task-arm 셀마다 정확히 3회")
    return data


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


def _existing(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compare_rows(run_dir: Path) -> list[dict[str, Any]]:
    """compare는 코드가 aggregate에서 읽은 수치만 담는다(손 입력 금지)."""
    rows: list[dict[str, Any]] = []
    aggregate_path = run_dir / "aggregate.json"
    if aggregate_path.exists():
        agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
        rows.append({"metric": "v_artifact", "arm": "V", "value": agg["v_pass"], "total": agg["v_valid"]})
        rows.append({"metric": "f_requirements", "arm": "F", "value": agg["f_pass"], "total": agg["f_total"]})
        rows.append({"metric": "f_complete", "arm": "F", "value": agg["f_complete_n"], "total": agg["f_outputs"]})
        for task, entry in sorted((agg.get("by_task") or {}).items()):
            rows.append({"metric": "f_requirements_by_task", "arm": task,
                         "value": entry["pass"], "total": entry["total"]})
    legacy_path = run_dir / "aggregate_legacy0726.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        rows.append({"metric": "v_artifact", "arm": "V(07-26)", "value": legacy["v_pass"], "total": legacy["v_valid"]})
        rows.append({"metric": "f_requirements", "arm": "F(07-26)", "value": legacy["f_pass"], "total": legacy["f_total"]})
        rows.append({"metric": "f_complete", "arm": "F(07-26)", "value": legacy["f_complete_n"], "total": legacy["f_outputs"]})
    return rows


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any]) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "task": row["task"],
            "task_name": row["task_name"],
            "arm": row["arm"],
            "repetition": row["repetition"],
            "output_file": row["transcript_file"],
            "response_file": row["response_file"],
            "score_file": row["score_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_s"),
            "prompt_eval_count": (row.get("api_metrics") or {}).get("prompt_eval_count"),
            "eval_count": (row.get("api_metrics") or {}).get("eval_count"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": "PRM-4PART V vs F (3 tasks x 2 arms x 3 reps, rotated)",
        "access": "local",
        "model": model,
        "generated_by": "run_prompt4_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "axis": "prompt_structure_vagueness_vs_4part",
        "plan_order": [f"{t}{a}r{r}" for t, a, r in PLAN],
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


def _turn_marker(previous: tuple[str, str] | None, current: tuple[str, str], task_names: dict[str, str]) -> None:
    if previous == current:
        return
    task, arm = current
    if previous is None or previous[0] != task:
        if previous is not None and previous[1] != arm:
            live_mark("turn", f"과업·팔 전환 — {previous[0]}{previous[1]}→{task}{arm}")
        else:
            live_mark("turn", f"과업 전환 — {task} {task_names[task]} · 독립 단일 턴")
    else:
        direction = "V→F" if arm == "F" else "F→V"
        hint = "막연한 한 줄에서 4칸 지시로" if arm == "F" else "4칸 지시에서 막연한 한 줄로"
        live_mark("turn", f"지시 팔 전환 — {task} {direction} · {hint}")


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    tasks = data["tasks"]
    plan = [{"key": f"{t}{a}-r{r}", "task": t, "arm": a, "repetition": r} for t, a, r in PLAN]
    if args.dry_run:
        print(json.dumps({"runs": len(plan), "order": [row["key"] for row in plan]}, ensure_ascii=False, indent=2))
        return 0

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    existing = _existing(args.run_dir)
    pending = [row for row in plan if row["key"] not in existing]
    print(f"planned={len(plan)} existing={len(plan) - len(pending)} running_now={len(pending)}", flush=True)

    if pending:
        metadata = model_metadata(args.base_url, args.model, args.timeout)
        (args.run_dir / "model-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        previous = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
        metadata = previous.get("model_metadata") or {}

    seen_break = {"V": False, "F": False}
    previous_cell: tuple[str, str] | None = None
    next_id = len(existing)
    for position, row in enumerate(pending, 1):
        next_id += 1
        task, arm = row["task"], row["arm"]
        prompt_text = tasks[task]["arms"][arm]
        _turn_marker(previous_cell, (task, arm), {t: tasks[t]["name"] for t in tasks})
        previous_cell = (task, arm)

        response, attempts, infra_error = generate(args.base_url, args.model, prompt_text, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""

        stem = f"{next_id:03d}"
        response_rel = f"raw/{stem}-response.txt"
        transcript_rel = f"raw/{stem}-output.txt"
        invocation_rel = f"raw/{stem}-invocation.json"
        score_rel = f"raw/{stem}-score.json"
        (args.run_dir / response_rel).write_text(response_text, encoding="utf-8")
        transcript = (
            f"run_id: {row['key']}\nmodel: {args.model}\n"
            f"task: {task}({tasks[task]['name']}) · arm: {arm} · rep: {row['repetition']}\n\n"
            f"[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
            f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")

        score: dict[str, Any]
        if infra_error or not response_text.strip():
            reason = infra_error or "빈 응답"
            live_mark("infra", f"생성 실패/빈 응답 — {row['key']} · {reason}")
            score = {"run_id": row["key"], "task": task, "arm": arm, "valid": False,
                     "reason": reason, "requirements": [], "failed": [], "complete": False}
        else:
            try:
                score = score_one(task, arm, response_text)
                score["run_id"] = row["key"]
                score["valid"] = True
                if score["failed"] and not seen_break[arm]:
                    first_req = next(r for r in score["requirements"] if not r["pass"])
                    seen_break[arm] = True
                    label = "V 최소형태 첫 실패" if arm == "V" else "F 명시요구 첫 실패"
                    live_mark("break", f"{label} — {row['key']} · 요구={first_req['id']} · "
                                       f"관측={first_req['observed']}")
            except Exception as exc:                      # noqa: BLE001 - 채점기 예외는 infra
                live_mark("infra", f"즉시채점 실패 — {row['key']} · {type(exc).__name__}: {exc}")
                score = {"run_id": row["key"], "task": task, "arm": arm, "valid": False,
                         "reason": f"scorer_error: {type(exc).__name__}", "requirements": [],
                         "failed": [], "complete": False}
        (args.run_dir / score_rel).write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")

        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "task": task,
            "task_name": tasks[task]["name"],
            "arm": arm,
            "repetition": row["repetition"],
            "prompt": prompt_text,
            "payload": {"model": args.model, "stream": False, "keep_alive": 0},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
            "response_file": response_rel,
            "transcript_file": transcript_rel,
            "score_file": score_rel,
            "api_metrics": {key: response.get(key) for key in (
                "done_reason", "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration"
            )} if response else None,
        }
        (args.run_dir / invocation_rel).write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_yaml(args.run_dir, args.model, metadata)
        print(f"[{position}/{len(pending)}] {row['key']} elapsed={invocation['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    # 집계 — 디스크의 score JSON을 다시 읽는다.
    disk_aggregate = score_run_dir(args.run_dir)
    (args.run_dir / "aggregate.json").write_text(
        json.dumps(disk_aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy = score_legacy_dir(LEGACY_DIR)
    (args.run_dir / "aggregate_legacy0726.json").write_text(
        json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_yaml(args.run_dir, args.model, metadata)

    live_mark("agg", f"집계 완료 — V 최소형태 {disk_aggregate['v_pass']}/{disk_aggregate['v_valid']} · "
                     f"F 명시요구 {disk_aggregate['f_pass']}/{disk_aggregate['f_total']} · "
                     f"F 완전준수 {disk_aggregate['f_complete_n']}/{disk_aggregate['f_outputs']} · "
                     f"infra {disk_aggregate['infra']}건")

    status = {
        "runs": len(plan),
        "completed": len(existing) + len(pending),
        "infra_errors": disk_aggregate["infra"],
        "complete": len(existing) + len(pending) == len(plan),
    }
    (args.run_dir / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(disk_aggregate, ensure_ascii=False, indent=2))
    return 0 if status["complete"] else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
