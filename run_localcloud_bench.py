#!/usr/bin/env python3
"""로컬 gemma3:4b vs 클라우드 Gemini 같은 날 대조 벤치 — 2026-09-21 재제작 11번.

공개판 주기 — 내부 testset 의존을 localcloud_bench_cases.json(프롬프트 전문+SHA)으로 대체했다.

Design contract (집필 서브 Linnaeus 설계 2026-09-21, b-seat-split 위임):
- 18 runs = 2 providers x 3 tasks(TXT-02/TXT-01/TXT-05) x 3 reps. Rotated order so no
  model or task always sits at the same position (order-effect defence). Every call is
  an independent single turn. No generation pilot/warm-up — first generation counts.
- Inputs byte-identical to testset.build_prompt() (SHA-256 verified against the
  2026-06-26 originals before the run; mismatch = exit 2).
- Ollama: keep_alive=0 on every request (VRAM 상주 금지), elapsed includes cold load.
  Gemini: subscription CLI (agy adapter), same-day cloud arm, sequential only.
- Scoring is deterministic (localcloud_score.py). Two cross-task break axes only:
  task_contract and output_integrity, so the published-image budget stays within 3.
- Markers: turn on model/task switch, break max 2 (one per axis), infra per event,
  agg once at the end with values re-read from score JSONs on disk — never hand-typed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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

from live_mark import mark as live_mark
from localcloud_score import score_task

HARNESS_VERSION = "1.0"
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "localcloud-gemma3-vs-gemini-20260921"

TASKS = ("TXT-02", "TXT-01", "TXT-05")
TASK_NAMES = {"TXT-02": "코딩", "TXT-01": "요약", "TXT-05": "번역"}

# 회전: 로컬 선두/클라우드 선두 반복을 섞고 과업 위치도 초·중·후반으로 회전.
PLAN: list[tuple[str, str, int]] = [
    ("local", "TXT-02", 1), ("cloud", "TXT-02", 1),
    ("local", "TXT-01", 1), ("cloud", "TXT-01", 1),
    ("local", "TXT-05", 1), ("cloud", "TXT-05", 1),
    ("cloud", "TXT-01", 2), ("local", "TXT-01", 2),
    ("cloud", "TXT-05", 2), ("local", "TXT-05", 2),
    ("cloud", "TXT-02", 2), ("local", "TXT-02", 2),
    ("local", "TXT-05", 3), ("cloud", "TXT-05", 3),
    ("local", "TXT-02", 3), ("cloud", "TXT-02", 3),
    ("local", "TXT-01", 3), ("cloud", "TXT-01", 3),
]

LOCAL_MODEL = "gemma3:4b"
CLOUD_MODEL = "Gemini 3.1 Pro (High)"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

TASK_CONTRACT_KEY = {
    "TXT-02": "coding_contract_pass",
    "TXT-01": "summary_contract_pass",
    "TXT-05": "translation_contract_pass",
}

TASK_FAIL_FIELDS = {
    "TXT-02": ("code_extractable", "syntax_valid", "defines_dedup_sort",
               "functional_pass", "example_output_shown"),
    "TXT-01": ("sentence_count_pass", "core_coverage_pass", "grounding_rule_pass"),
    "TXT-05": ("term_coverage_pass", "term_consistency_pass",
               "clause_coverage_pass", "unsupported_numeric_pass"),
}

TASK_RULES = {
    "TXT-02": {
        "code_extractable": "python 코드 블록 또는 파이썬 소스로 실행 후보 확보",
        "syntax_valid": "ast.parse·compile 성공",
        "defines_dedup_sort": "호출 가능한 dedup_sort 정의",
        "functional_pass": "고정 5개 실행 케이스 전부 결과·타입 일치",
        "example_output_shown": "예시 입력 [3,1,2,3,1]의 결과 [3, 2, 1] 제시",
    },
    "TXT-01": {
        "sentence_count_pass": "정확히 5문장",
        "core_coverage_pass": "K1~K5 핵심 슬롯 전부 포함",
        "grounding_rule_pass": "허용 밖 숫자·모순·미매핑 문장 없음",
    },
    "TXT-05": {
        "term_coverage_pass": "9개 용어 전부 검출",
        "term_consistency_pass": "같은 개념의 한국어 표기 혼용 없음",
        "clause_coverage_pass": "P1~P7 조항 전부 번역",
        "unsupported_numeric_pass": "원문에 없는 숫자 없음",
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_inputs() -> tuple[dict[str, str], dict[str, str]]:
    """cases JSON 의 프롬프트가 기록된 SHA-256 과 일치하는지 실행 전 확인(공개판)."""
    cases_path = Path(__file__).with_name("localcloud_bench_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    prompts: dict[str, str] = cases["prompts"]
    recorded: dict[str, str] = cases["sha256"]
    hashes: dict[str, str] = {}
    for task_id in TASKS:
        if task_id not in prompts:
            raise SystemExit("[X] cases JSON 에 " + task_id + " 프롬프트가 없습니다. 실행 중단.")
        digest = _sha256(prompts[task_id])
        if task_id in recorded and digest != recorded[task_id]:
            raise SystemExit("[X] 입력 불일치 — " + task_id + " 가 기록된 SHA-256 과 다릅니다. 실행 중단.")
        hashes[task_id] = digest
    return hashes, prompts


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


def ollama_generate(base_url: str, model: str, prompt: str, timeout: int):
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0}
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3),
                             "error": None})
            return response, attempts, None
        except urllib.error.HTTPError as exc:
            error = "HTTPError " + str(exc.code) + ": " + str(exc.reason)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3),
                             "error": error})
            transient = exc.code in (408, 425, 429) or exc.code >= 500
            if not transient or attempt == 2:
                return None, attempts, error
            time.sleep(1)
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            error = type(exc).__name__ + ": " + str(exc)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3),
                             "error": error})
            if attempt == 2:
                return None, attempts, error
            time.sleep(1)
    return None, attempts, attempts[-1]["error"]


def gemini_generate(prompt: str, model: str, timeout: int):
    import agy_adapter

    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            text = agy_adapter.run_gemini(prompt, model=model, timeout=timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3),
                             "error": None})
            return text, attempts, None
        except Exception as exc:  # noqa: BLE001 - 어댑터 예외 전부 infra 후보
            error = type(exc).__name__ + ": " + str(exc)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3),
                             "error": error})
            if attempt == 2:
                return None, attempts, error
            time.sleep(2)
    return None, attempts, attempts[-1]["error"]


def model_metadata(base_url: str, model: str, timeout: int) -> dict[str, Any]:
    try:
        return api_json(base_url.rstrip("/") + "/api/show", {"model": model}, timeout)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("모델 메타데이터 조회 실패: " + str(exc)) from exc


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _existing(run_dir: Path) -> set:
    keys = set()
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys


def output_hygiene(raw_text: str) -> dict[str, Any]:
    ansi_count = len(ANSI_RE.findall(raw_text))
    clean = ANSI_RE.sub("", raw_text)
    control_count = len(CONTROL_RE.findall(clean))
    return {
        "ansi_escape_count": ansi_count,
        "control_char_count": control_count,
        "output_clean_pass": ansi_count == 0 and control_count == 0,
        "clean_text": clean,
    }


def _model_label(provider: str) -> str:
    return LOCAL_MODEL if provider == "local" else CLOUD_MODEL


def _turn_marker(previous, current) -> None:
    if previous == current:
        return
    provider, task, rep = current
    if previous is None:
        live_mark("turn", "측정 시작 — " + _model_label(provider) + "/" + task
                  + " · 반복 " + str(rep) + "/3 · 독립 요청 · keep_alive=0(로컬)")
        return
    prev_provider, prev_task, _ = previous
    if provider != prev_provider:
        live_mark("turn", "측정 조건 전환 — " + _model_label(prev_provider) + "/" + prev_task
                  + " → " + _model_label(provider) + "/" + task
                  + " · 반복 " + str(rep) + "/3 · 독립 요청")
    else:
        live_mark("turn", "과업 전환 — " + _model_label(provider) + "/" + prev_task
                  + " → " + task + " · 반복 " + str(rep) + "/3 · 독립 요청")


def write_run_yaml(run_dir: Path, metadata: dict[str, Any], input_hashes: dict[str, str]) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "task": row["task"],
            "provider": row["provider"],
            "model": row["model"],
            "repetition": row["repetition"],
            "output_file": row["transcript_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_total_s"),
            "score_file": row.get("score_file"),
            "infra_error": row.get("infra_error"),
        })
    compare: list[dict[str, Any]] = []
    aggregate_path = run_dir / "aggregate.json"
    if aggregate_path.exists():
        agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
        for side in ("local", "cloud"):
            stats = agg["by_provider"][side]
            compare.append({"metric": "task_contract_pass", "tool": _model_label(side),
                            "value": stats["contract_pass"], "total": stats["valid_outputs"]})
            compare.append({"metric": "median_elapsed_total_s", "tool": _model_label(side),
                            "value": stats["median_elapsed_total_s"], "total": stats["valid_outputs"]})
        for task_id, entry in sorted((agg.get("by_task") or {}).items()):
            for side in ("local", "cloud"):
                compare.append({"metric": "task_contract_by_task", "tool": _model_label(side),
                                "value": entry[side]["contract_pass"],
                                "total": entry[side]["valid_outputs"],
                                "task": task_id})
    payload = {
        "tool": "ollama+gemini-cli",
        "date": dt.date.today().isoformat(),
        "method": "TXT-02,TXT-01,TXT-05",
        "access": "local+cli",
        "model": LOCAL_MODEL + " vs " + CLOUD_MODEL,
        "generated_by": "run_localcloud_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동) · Antigravity 구독 CLI(공식 클라이언트·OAuth)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "axis": "local_free_4b_vs_subscription_cloud_same_day",
        "plan_order": [p + "-" + t + "-r" + str(r) for p, t, r in PLAN],
        "input_sha256": input_hashes,
        "compare": compare,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "ollama_models": os.environ.get("OLLAMA_MODELS"),
        },
        "model_metadata": {
            "modified_at": (metadata or {}).get("modified_at"),
            "parameters": (metadata or {}).get("parameters"),
        },
        "runs": entries,
    }
    tmp = run_dir / "run.yaml.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run.yaml")


def aggregate(run_dir: Path) -> dict[str, Any]:
    """디스크의 score JSON을 다시 읽어 집계한다(손 입력 금지)."""
    rows = []
    for path in sorted((run_dir / "raw").glob("*-score.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    by_provider: dict[str, Any] = {}
    for side in ("local", "cloud"):
        subset = [r for r in rows if r.get("provider") == side]
        valid = [r for r in subset if r.get("valid_response")]
        elapsed = sorted(r["elapsed_total_s"] for r in valid
                         if r.get("elapsed_total_s") is not None)
        med = elapsed[len(elapsed) // 2] if elapsed else None
        by_provider[side] = {
            "outputs": len(subset),
            "valid_outputs": len(valid),
            "contract_pass": sum(1 for r in valid if r.get("task_contract_pass")),
            "output_clean_pass": sum(1 for r in valid if r.get("output_clean_pass")),
            "median_elapsed_total_s": med,
            "elapsed_total_s": [r.get("elapsed_total_s") for r in subset],
            "invalid_cells": len(subset) - len(valid),
        }
    by_task: dict[str, Any] = {}
    for task_id in TASKS:
        entry = {}
        for side in ("local", "cloud"):
            subset = [r for r in rows if r.get("provider") == side and r.get("task") == task_id]
            valid = [r for r in subset if r.get("valid_response")]
            entry[side] = {
                "outputs": len(subset),
                "valid_outputs": len(valid),
                "contract_pass": sum(1 for r in valid if r.get("task_contract_pass")),
            }
        by_task[task_id] = entry
    infra_events = sum(1 for r in rows if r.get("infra_error"))
    return {
        "generated_by": "run_localcloud_bench.py:aggregate",
        "rows_read": len(rows),
        "by_provider": by_provider,
        "by_task": by_task,
        "infra_events": infra_events,
    }


def _observed_value(task_id: str, score: dict[str, Any]) -> str:
    if task_id == "TXT-02":
        if score.get("functional_cases_passed") is not None:
            return "실행케이스 " + str(score.get("functional_cases_passed")) + "/5"
        return str(score.get("extraction_mode", "N/A"))
    if task_id == "TXT-01":
        return (str(score.get("sentence_count", "N/A")) + "문장 · K "
                + str(score.get("core_coverage_count", "N/A")) + "/5")
    return ("P " + str(score.get("clause_coverage_count", "N/A")) + "/7 · 용어 "
            + str(score.get("term_coverage_count", 0)) + "/9")


def run_benchmark(args: argparse.Namespace) -> int:
    input_hashes, _ = verify_inputs()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    existing = _existing(args.run_dir)
    pending = [row for row in PLAN if (row[0] + "-" + row[1] + "-r" + str(row[2])) not in existing]
    print("planned=" + str(len(PLAN)) + " existing=" + str(len(PLAN) - len(pending))
          + " running_now=" + str(len(pending)), flush=True)

    metadata: dict[str, Any] = {}
    if pending:
        metadata = model_metadata(args.base_url, args.model, args.timeout)
        (args.run_dir / "model-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _, prompts = verify_inputs()
    seen_break = {"task_contract": False, "output_integrity": False}
    previous_cell = None
    next_id = len(existing)
    for position, (provider, task_id, rep) in enumerate(pending, 1):
        next_id += 1
        key = provider + "-" + task_id + "-r" + str(rep)
        _turn_marker(previous_cell, (provider, task_id, rep))
        previous_cell = (provider, task_id, rep)
        prompt_text = prompts[task_id]
        model = LOCAL_MODEL if provider == "local" else CLOUD_MODEL

        if provider == "local":
            response, attempts, infra_error = ollama_generate(
                args.base_url, args.model, prompt_text, args.timeout)
            response_text = (response or {}).get("response") if response else None
            api_metrics = ({k: response.get(k) for k in (
                "done_reason", "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration")} if response else None)
        else:
            response_text, attempts, infra_error = gemini_generate(
                prompt_text, args.cloud_model, args.timeout)
            api_metrics = None

        if response_text is not None and not isinstance(response_text, str):
            infra_error = infra_error or "invalid_response: response가 문자열이 아님"
            response_text = None
        if response_text is not None and not response_text.strip():
            infra_error = infra_error or "빈 응답"
            response_text = None

        elapsed_total = round(sum(a["elapsed_s"] for a in attempts), 3)
        stem = str(next_id).zfill(3)
        response_rel = "raw/" + stem + "-response.txt"
        transcript_rel = "raw/" + stem + "-output.txt"
        invocation_rel = "raw/" + stem + "-invocation.json"
        score_rel = "raw/" + stem + "-score.json"
        raw_text = response_text if response_text is not None else ""
        (args.run_dir / response_rel).write_text(raw_text, encoding="utf-8")

        if raw_text:
            hygiene = output_hygiene(raw_text)
        else:
            hygiene = {"ansi_escape_count": None, "control_char_count": None,
                       "output_clean_pass": None, "clean_text": ""}
        score = {
            "key": key,
            "provider": provider,
            "model": model,
            "task": task_id,
            "task_name": TASK_NAMES[task_id],
            "repetition": rep,
            "plan_position": position,
            "attempt_count": len(attempts),
            "elapsed_total_s": elapsed_total,
            "infra_error": infra_error,
            "valid_response": bool(raw_text.strip()) and not infra_error,
            "raw_sha256": _sha256(raw_text) if raw_text else None,
            "raw_bytes": len(raw_text.encode("utf-8")) if raw_text else 0,
            "ansi_escape_count": hygiene["ansi_escape_count"],
            "control_char_count": hygiene["control_char_count"],
            "output_clean_pass": hygiene["output_clean_pass"],
            "task_contract_pass": False,
        }
        if score["valid_response"]:
            try:
                task_score = score_task(task_id, hygiene["clean_text"])
                score.update(task_score)
                score["task_contract_pass"] = bool(task_score.get(TASK_CONTRACT_KEY[task_id]))
            except Exception as exc:  # noqa: BLE001 - 채점기 예외는 infra
                live_mark("infra", "측정 흔들림 — " + key + " · 채점/기록 실패=" + type(exc).__name__)
                score["valid_response"] = False
                score["infra_error"] = "scorer_error: " + type(exc).__name__ + ": " + str(exc)
        else:
            reason = infra_error or "빈 응답"
            live_mark("infra", "측정 흔들림 — " + model + "/" + task_id + "/r" + str(rep)
                      + " · " + reason + " · 시도 " + str(len(attempts)) + "/2")

        failed_fields = [k for k, v in score.items()
                         if k in TASK_FAIL_FIELDS.get(task_id, ()) and v is False]
        score["failed_fields"] = failed_fields

        if score["valid_response"] and not score["task_contract_pass"] and not seen_break["task_contract"]:
            seen_break["task_contract"] = True
            first = failed_fields[0] if failed_fields else "unknown"
            live_mark("break", "과업 계약 첫 실패 — " + model + "/" + task_id + "/r" + str(rep)
                      + " · 필드=" + first + " · 관측=" + _observed_value(task_id, score)
                      + " · 기준=" + TASK_RULES[task_id].get(first, "계약 충족"))
        if (score["valid_response"] and score["output_clean_pass"] is False
                and not seen_break["output_integrity"]):
            seen_break["output_integrity"] = True
            live_mark("break", "출력 위생 첫 실패 — " + model + "/" + task_id + "/r" + str(rep)
                      + " · ANSI=" + str(score["ansi_escape_count"])
                      + " · 제어문자=" + str(score["control_char_count"]))

        (args.run_dir / score_rel).write_text(json.dumps(score, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        transcript = ("run_id: " + key + "\nprovider: " + provider + " · model: " + model
                      + " · task: " + task_id + "(" + TASK_NAMES[task_id] + ") · rep: " + str(rep)
                      + "\n\n[PROMPT sha256=" + input_hashes[task_id] + "]\n" + prompt_text
                      + "\n\n[RAW RESPONSE]\n" + raw_text + "\n")
        if api_metrics:
            transcript += ("\n[API METADATA]\n"
                           + json.dumps(api_metrics, ensure_ascii=False, indent=2) + "\n")
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": key,
            "provider": provider,
            "model": model,
            "task": task_id,
            "repetition": rep,
            "prompt": prompt_text,
            "payload": ({"model": args.model, "stream": False, "keep_alive": 0}
                        if provider == "local" else {"cli": "agy", "model": args.cloud_model}),
            "attempts": attempts,
            "elapsed_total_s": elapsed_total,
            "infra_error": infra_error,
            "response_file": response_rel,
            "transcript_file": transcript_rel,
            "score_file": score_rel,
            "api_metrics": api_metrics,
        }
        (args.run_dir / invocation_rel).write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_yaml(args.run_dir, metadata, input_hashes)
        print("[" + str(position) + "/" + str(len(pending)) + "] " + key
              + " elapsed=" + str(elapsed_total) + "s"
              + (" ERROR=" + str(infra_error) if infra_error else ""), flush=True)

    agg = aggregate(args.run_dir)
    (args.run_dir / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_yaml(args.run_dir, metadata, input_hashes)

    local = agg["by_provider"]["local"]
    cloud = agg["by_provider"]["cloud"]
    invalid_total = local["invalid_cells"] + cloud["invalid_cells"]
    if invalid_total:
        live_mark("agg", "집계 불완전 — 유효 로컬 " + str(local["valid_outputs"]) + "/9·클라우드 "
                  + str(cloud["valid_outputs"]) + "/9 · 최종 무효 " + str(invalid_total)
                  + "셀 · infra " + str(agg["infra_events"]) + "건 · 모델 비교 판정 보류")
    else:
        bt = agg["by_task"]
        live_mark("agg", "같은 날 18회 집계 — 계약 로컬 " + str(local["contract_pass"]) + "/9·클라우드 "
                  + str(cloud["contract_pass"]) + "/9 · 코딩 "
                  + str(bt["TXT-02"]["local"]["contract_pass"]) + "/3:"
                  + str(bt["TXT-02"]["cloud"]["contract_pass"]) + "/3 · 요약 "
                  + str(bt["TXT-01"]["local"]["contract_pass"]) + "/3:"
                  + str(bt["TXT-01"]["cloud"]["contract_pass"]) + "/3 · 번역 "
                  + str(bt["TXT-05"]["local"]["contract_pass"]) + "/3:"
                  + str(bt["TXT-05"]["cloud"]["contract_pass"]) + "/3 · 중앙 대기 "
                  + str(local["median_elapsed_total_s"]) + "s:"
                  + str(cloud["median_elapsed_total_s"]) + "s · 출력위생 "
                  + str(local["output_clean_pass"]) + ":" + str(cloud["output_clean_pass"])
                  + " · infra " + str(agg["infra_events"]) + "건")
    print(json.dumps(agg, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-url",
                        default=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--cloud-model", default=CLOUD_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
