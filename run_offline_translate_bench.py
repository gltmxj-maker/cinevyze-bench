#!/usr/bin/env python3
"""오프라인 로컬 번역 지시 2x2 벤치 — 2026-09-22 재제작 13번(오프라인 번역 글 R2).

공개판 주기 — 입력 문서 3종을 corpus/ 파일 대신 offline_translate_bench_cases.json 에서 읽는다(SHA 검증 동일).

Design contract(집필 서브 Erdos 설계 2026-09-22, b-seat-split 위임):
- 36 runs = 4 arms(B 기본/G 용어 고정/S 문체 고정/GS 결합) x 3 inputs(T 기술/C 후기/E 메일)
  x 3 reps. 회전 순서 고정 — 팔·입력이 연속 같은 위치를 독점하지 않게(순서 효과 방어).
  모든 셀은 독립 단일턴 요청·직렬 실행. 워밍업 없음 — 첫 호출부터 측정에 포함.
- 입력은 6월 원문 3종을 바이트 그대로(SHA-256 실행 전 검증·불일치 exit 2).
- Ollama: temperature=0, seed=13, keep_alive=0(VRAM 상주 금지). elapsed 에 콜드 로드 포함.
- 채점은 결정론적(offline_translate_score.py). break 마커는 축별 1회(내용 보존·지시 준수)
  최대 2개 — 게재 이미지 예산 3장(break 전부+agg 1) 준수.
- This script is the write-origin for run.yaml. 매 시행의 프롬프트 전문·원응답·API 메타를
  raw/ 에 저장하고 그 기록만으로 run.yaml 을 만든다 — 사람이 수치를 적어 넣을 자리가 없다.
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
from offline_translate_score import ARMS, ARM_NAMES, TASKS, TASK_NAMES, score_response

HARNESS_VERSION = "1.0"
SCORER_VERSION = "1.0"
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "offline-translate-gemma3-instruction-20260922"
LOCAL_MODEL = "gemma3:4b"
TEMPERATURE = 0
SEED = 13

CASES_FILE = Path(__file__).with_name("offline_translate_bench_cases.json")
INPUT_FILES = None  # 공개판: cases JSON 에서 로드
INPUT_SHA256 = {
    "T": "54383481dd2051978e04c038396dcff3460fa37b65462d82a53425cd905d7333",
    "C": "b03fb4c25871746837a95540f1060059ac0074a077f94008b4c99da907076dbb",
    "E": "a838a7cdce50ae55652b9ed7333aae539cc704e171060f267e32cbbe398069c1",
}

# 회전(설계안 그대로): 팔·입력 위치가 반복마다 뒤섞이도록 고정.
PLAN: list[tuple[str, str, int]] = [
    ("B", "T", 1), ("G", "C", 1), ("S", "E", 1), ("GS", "T", 1),
    ("B", "C", 1), ("G", "E", 1), ("S", "T", 1), ("GS", "C", 1),
    ("B", "E", 1), ("G", "T", 1), ("S", "C", 1), ("GS", "E", 1),
    ("G", "E", 2), ("S", "T", 2), ("GS", "C", 2), ("B", "E", 2),
    ("G", "T", 2), ("S", "C", 2), ("GS", "E", 2), ("B", "T", 2),
    ("G", "C", 2), ("S", "E", 2), ("GS", "T", 2), ("B", "C", 2),
    ("S", "C", 3), ("GS", "E", 3), ("B", "T", 3), ("G", "C", 3),
    ("S", "E", 3), ("GS", "T", 3), ("B", "C", 3), ("G", "E", 3),
    ("S", "T", 3), ("GS", "C", 3), ("B", "E", 3), ("G", "T", 3),
]

GLOSSARY_LINES = {
    "T": ("large language model→대규모 언어 모델, latency→지연 시간, throughput→처리량, "
          "batching→배치, quantization→양자화, weights→가중치, inference→추론"),
    "C": ("setup instructions→설치 안내, battery→배터리, two weeks→2주, "
          "keeper→계속 쓸 만한 제품"),
    "E": ("address discrepancy→주소 불일치, regional facility→지역 시설, "
          "three business days→영업일 기준 3일, 10 percent credit→10% 크레딧"),
}
REGISTER_LINES = {
    "T": "문체는 설명문으로, 모든 문장을 '-다' 체로 끝내세요.",
    "C": "문체는 친근하게, 모든 문장을 '-요' 체로 끝내세요.",
    "E": "문체는 격식 있게, 모든 문장을 '-습니다' 체로 끝내세요.",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_inputs(base: Path) -> tuple[dict[str, str], dict[str, str]]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    inputs = cases["inputs"]
    recorded = cases["sha256"]
    hashes = {}
    for task, text in inputs.items():
        digest = _sha256_bytes(text.encode("utf-8"))
        if task in recorded and digest != recorded[task]:
            raise SystemExit(f"[X] 입력 불일치 — {task} SHA가 기록된 값과 다릅니다. 실행 중단.")
        hashes[task] = digest
    return hashes, inputs


def build_prompt(arm: str, task: str, source_text: str) -> str:
    lines = ["다음 영어 텍스트를 자연스러운 한국어로 번역해줘. 번역문만 출력하고 다른 설명·머리말은 붙이지 마."]
    if arm in ("G", "GS"):
        lines.append(f"용어는 다음 표기를 정확히 사용해야 해: {GLOSSARY_LINES[task]}")
    if arm in ("S", "GS"):
        lines.append(REGISTER_LINES[task])
    lines.append("")
    lines.append(source_text.rstrip("\n"))
    return "\n".join(lines)


def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
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
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
               "options": {"temperature": TEMPERATURE, "seed": SEED}}
    attempts = []
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


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        keys.add(row["key"])
        maximum = max(maximum, int(row["run_id"]))
    return keys, maximum


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def write_score_spec(run_dir: Path, input_hashes: dict[str, str]) -> None:
    spec = {
        "scorer": "offline_translate_score.py", "scorer_version": SCORER_VERSION,
        "axes": ["content_preservation", "instruction_compliance"],
        "arms": list(ARMS), "tasks": list(TASKS),
        "note": "채점 규칙은 offline_translate_score.py 와 동일값 — 결과를 보고 규칙을 고치지 않는다.",
        "input_sha256": input_hashes,
    }
    target = run_dir / "score-spec.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("input_sha256") != input_hashes:
            raise SystemExit("[X] score-spec 입력 해시 불일치 — 새 run 폴더를 쓰세요.")
        return
    target.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any], input_hashes: dict[str, str],
                   prompt_hashes: dict[str, str]) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        metrics = row.get("api_metrics") or {}
        entries.append({
            "key": row["key"], "plan_position": row["plan_position"],
            "arm": row["arm"], "task": row["task"], "repetition": row["repetition"],
            "source_file": row["source_file"], "source_sha256": row["source_sha256"],
            "prompt_file": row["prompt_file"], "prompt_sha256": row["prompt_sha256"],
            "output_file": row["response_file"], "invocation_file": str(path.relative_to(run_dir)),
            "log_file": str(path.relative_to(run_dir)),
            "score_file": row.get("score_file"),
            "attempt_count": row.get("attempt_count"),
            "wall_s": row.get("wall_s"), "elapsed_s": row.get("elapsed_s"),
            "eval_count": metrics.get("eval_count"), "eval_duration_ns": metrics.get("eval_duration"),
            "tok_per_s": row.get("tok_per_s"),
            "infra_error": row.get("infra_error"),
            "content_contract_pass": row.get("content_contract_pass"),
            "instruction_contract_pass": row.get("instruction_contract_pass"),
            "full_contract_pass": row.get("full_contract_pass"),
        })
    payload = {
        "tool": "ollama", "date": dt.date.today().isoformat(),
        "method": "offline-translation-instruction-2x2 — 6월 원문 3종(바이트 고정)을 4팔(기본/용어/문체/결합)로 번역. "
                  "품질은 사전 고정 계약(조항·용어·숫자·형식·문체)의 준수율로만 측정 — 자연스러움 전체는 UNKNOWN.",
        "access": "local-loopback", "model": model,
        "generated_by": "run_offline_translate_bench.py",
        "harness_version": HARNESS_VERSION, "scorer_version": SCORER_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0, "request_parallelism": 1,
        "temperature": TEMPERATURE, "seed": SEED,
        "planned_runs": len(PLAN), "valid_runs": sum(1 for e in entries if not e["infra_error"]),
        "axis": ["terminology_instruction", "register_instruction"],
        "arms": list(ARMS), "tasks": list(TASKS), "repetitions": 3,
        "plan_order": [f"{a}-{t}-r{r}" for a, t, r in PLAN],
        "input_sha256": input_hashes, "prompt_sha256": prompt_hashes,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "git_sha_at_run": _git_sha(), "ollama_host": "http://127.0.0.1:11434",
            "ollama_model_digest": (metadata.get("details") or {}).get("id"),
            "external_network_state": "UNKNOWN",
        },
        "model_metadata": {"modified_at": metadata.get("modified_at"),
                           "details": metadata.get("details"),
                           "parameters": metadata.get("parameters")},
        "compare": _compare_from_aggregate(run_dir),
        "runs": entries,
    }
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, run_dir / "run.yaml")


def _compare_from_aggregate(run_dir: Path) -> list[dict[str, Any]]:
    """집계 JSON 을 되읽어 비교 기록을 만든다 — 사람이 compare 를 손으로 쓸 자리가 없다."""
    path = run_dir / "aggregate.json"
    if not path.exists():
        return []
    agg = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for arm in ("B", "G", "S", "GS"):
        v = (agg.get("arms") or {}).get(arm) or {}
        rows.append({
            "arm": arm, "name": ARM_NAMES[arm],
            "full_contract_pass": f"{v.get('full_pass')}/{v.get('n')}",
            "content_contract_pass": f"{v.get('content_pass')}/{v.get('n')}",
            "instruction_contract_pass": f"{v.get('instruction_pass')}/{v.get('n')}",
            "tok_per_s_median": v.get("tok_per_s_median"),
            "wall_s_median": v.get("wall_s_median"),
        })
    rows.append({"valid_runs": agg.get("valid_rows"), "invalid_runs": agg.get("invalid_rows"),
                 "infra_errors": agg.get("infra_errors")})
    return rows


def build_aggregate(run_dir: Path) -> dict[str, Any]:
    """저장된 score.json 36개를 다시 읽어 집계한다 — 마커·run.yaml 이 되읽는 같은 원천."""
    rows = []
    for path in sorted((run_dir / "raw").glob("*-score.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    by_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r.get("arm") == arm and r.get("valid")]
        per_task = {}
        for task in TASKS:
            task_rows = [r for r in arm_rows if r.get("task") == task]
            per_task[task] = {
                "full": sum(1 for r in task_rows if r.get("full_contract_pass")),
                "content": sum(1 for r in task_rows if r.get("content_contract_pass")),
                "instruction": sum(1 for r in task_rows if r.get("instruction_contract_pass")),
                "n": len(task_rows),
            }
        tok_values = [r.get("tok_per_s") for r in arm_rows if r.get("tok_per_s")]
        wall_values = [r.get("wall_s") for r in arm_rows if r.get("wall_s")]
        by_arm[arm] = {
            "name": ARM_NAMES[arm], "n": len(arm_rows),
            "full_pass": sum(1 for r in arm_rows if r.get("full_contract_pass")),
            "content_pass": sum(1 for r in arm_rows if r.get("content_contract_pass")),
            "instruction_pass": sum(1 for r in arm_rows if r.get("instruction_contract_pass")),
            "tok_per_s_median": round(_median(tok_values), 1),
            "tok_per_s_min": round(min(tok_values), 1) if tok_values else None,
            "tok_per_s_max": round(max(tok_values), 1) if tok_values else None,
            "wall_s_median": round(_median(wall_values), 2),
            "per_task": per_task,
        }
    invalid = [r for r in rows if not r.get("valid")]
    infra = [r for r in rows if r.get("infra_error")]
    aggregate = {"arms": by_arm, "total_rows": len(rows), "valid_rows": len(rows) - len(invalid),
                 "invalid_rows": len(invalid), "infra_errors": len(infra)}
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def run_benchmark(args: argparse.Namespace) -> int:
    base = Path(__file__).resolve().parent
    input_hashes, sources = verify_inputs(base)

    prompts = {f"{arm}-{task}": build_prompt(arm, task, sources[task])
               for arm in ARMS for task in TASKS}
    prompt_hashes = {key: _sha256_bytes(text.encode("utf-8")) for key, text in prompts.items()}

    if args.dry_run:
        print(json.dumps({"planned": len(PLAN), "first_key": f"{PLAN[0][0]}-{PLAN[0][1]}-r{PLAN[0][2]}",
                          "first_prompt": prompts[f"{PLAN[0][0]}-{PLAN[0][1]}"][:400] + "…"},
                         ensure_ascii=False, indent=2))
        return 0

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    write_score_spec(args.run_dir, input_hashes)
    existing_keys, next_id = _existing(args.run_dir)
    pending = [row for row in PLAN if f"{row[0]}-{row[1]}-r{row[2]}" not in existing_keys]
    print(f"planned={len(PLAN)} existing={len(PLAN) - len(pending)} running_now={len(pending)}")

    metadata = model_metadata(args.base_url, args.model, args.timeout)
    inputs_dir = args.run_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for task, rel in INPUT_FILES.items():
        target = inputs_dir / Path(rel).name
        if not target.exists():
            target.write_bytes((base / rel).read_bytes())

    content_break_fired = False
    instruction_break_fired = False
    last_cell = None
    for position, (arm, task, rep) in enumerate(pending, 1):
        key = f"{arm}-{task}-r{rep}"
        if last_cell is None:
            live_mark("turn", f"측정 시작 — {args.model} · {ARM_NAMES[arm]}/{TASK_NAMES[task]} · "
                              f"반복 {rep}/3 · 독립 요청 · temperature=0 · keep_alive=0")
        elif (arm, task) != last_cell:
            live_mark("turn", f"측정 조건 전환 — {ARM_NAMES[last_cell[0]]}/{TASK_NAMES[last_cell[1]]} → "
                              f"{ARM_NAMES[arm]}/{TASK_NAMES[task]} · 반복 {rep}/3 · 독립 요청")
        last_cell = (arm, task)

        next_id += 1
        stem = f"{next_id:03d}"
        prompt = prompts[f"{arm}-{task}"]
        response, attempts, infra_error = generate(args.base_url, args.model, prompt, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        wall_s = round(sum(a["elapsed_s"] for a in attempts), 3)
        metrics = {k: response.get(k) for k in (
            "done_reason", "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration")} if response else None
        tok_per_s = None
        if metrics and metrics.get("eval_count") and metrics.get("eval_duration"):
            tok_per_s = round(metrics["eval_count"] / (metrics["eval_duration"] / 1e9), 1)

        prompt_rel = f"raw/{stem}-prompt.txt"
        response_rel = f"raw/{stem}-response.txt"
        invocation_rel = f"raw/{stem}-invocation.json"
        score_rel = f"raw/{stem}-score.json"
        (args.run_dir / prompt_rel).write_text(prompt, encoding="utf-8")
        (args.run_dir / response_rel).write_text(response_text, encoding="utf-8")

        scored = None
        if not infra_error:
            scored = score_response(arm, task, response_text)
            scored.update({"key": key, "tok_per_s": tok_per_s, "wall_s": wall_s,
                           "infra_error": None})
            (args.run_dir / score_rel).write_text(
                json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
            if not content_break_fired and not scored.get("content_contract_pass"):
                content_break_fired = True
                clause = scored.get("clause_coverage", {})
                term = scored.get("term_coverage", {})
                numeric_ok = scored.get("numeric_fidelity", {}).get("pass") and \
                    scored.get("unsupported_number", {}).get("pass")
                live_mark("break", f"내용 보존 첫 실패 — {key} · 누락 조항 "
                                   f"{clause.get('total', 0) - clause.get('hit', 0)}/{clause.get('total', 0)} · "
                                   f"핵심 용어 {term.get('matched', 0)}/{term.get('total', 0)} · "
                                   f"숫자·단위 {'통과' if numeric_ok else '실패'}")
            if not instruction_break_fired and not scored.get("instruction_contract_pass"):
                instruction_break_fired = True
                only = "통과" if scored.get("translation_only", {}).get("pass") else "실패"
                if arm in ("G", "GS"):
                    glos = "통과" if scored.get("term_coverage", {}).get("pass") else "실패"
                else:
                    glos = "적용 없음"
                if arm in ("S", "GS"):
                    reg = "통과" if scored.get("register_pass") else "실패"
                else:
                    reg = "적용 없음"
                live_mark("break", f"지시 준수 첫 실패 — {key} · 번역문만 {only} · 용어 고정 {glos} · 문체 {reg}")
        else:
            live_mark("infra", f"측정 흔들림 — {key} · {infra_error} · 시도 {len(attempts)}/2 · 해당 셀 무효")

        invocation = {
            "run_id": next_id, "key": key, "plan_position": PLAN.index((arm, task, rep)) + 1,
            "arm": arm, "task": task, "repetition": rep,
            "source_file": "offline_translate_bench_cases.json", "source_sha256": input_hashes[task],
            "prompt_file": prompt_rel, "prompt_sha256": prompt_hashes[f"{arm}-{task}"],
            "response_file": response_rel, "score_file": score_rel if scored else None,
            "payload": {"model": args.model, "stream": False, "keep_alive": 0,
                        "options": {"temperature": TEMPERATURE, "seed": SEED}},
            "attempts": attempts, "attempt_count": len(attempts),
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "wall_s": wall_s, "elapsed_s": wall_s,
            "tok_per_s": tok_per_s, "infra_error": infra_error,
            "response_sha256": _sha256_bytes(response_text.encode("utf-8")) if response_text else None,
            "api_metrics": metrics,
            "content_contract_pass": (scored or {}).get("content_contract_pass"),
            "instruction_contract_pass": (scored or {}).get("instruction_contract_pass"),
            "full_contract_pass": (scored or {}).get("full_contract_pass"),
            "register_pass": (scored or {}).get("register_pass"),
        }
        (args.run_dir / invocation_rel).write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_yaml(args.run_dir, args.model, metadata, input_hashes, prompt_hashes)
        status = "OK"
        if scored:
            status = ("full" if scored.get("full_contract_pass")
                      else "content" if scored.get("content_contract_pass")
                      else "instruction" if scored.get("instruction_contract_pass") else "FAIL")
        print(f"[{position}/{len(pending)}] {key} elapsed={wall_s}s tok/s={tok_per_s} {status}"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    write_run_yaml(args.run_dir, args.model, metadata, input_hashes, prompt_hashes)
    aggregate = build_aggregate(args.run_dir)
    if aggregate["valid_rows"] == len(PLAN) and aggregate["infra_errors"] == 0:
        arms = aggregate["arms"]
        live_mark("agg", f"36회 집계 완료 — 완전통과 기본 {arms['B']['full_pass']}/9·"
                         f"용어 {arms['G']['full_pass']}/9·문체 {arms['S']['full_pass']}/9·"
                         f"결합 {arms['GS']['full_pass']}/9 · 생성속도 중앙값 "
                         f"{arms['B']['tok_per_s_median']} tok/s · 전체 대기 중앙값 "
                         f"{arms['B']['wall_s_median']}s · infra 0건")
    else:
        live_mark("agg", f"집계 불완전 — 유효 {aggregate['valid_rows']}/36 · "
                         f"무효 {aggregate['invalid_rows']} · infra {aggregate['infra_errors']}건 · 팔 비교 판정 보류")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["valid_rows"] == len(PLAN) else 4


def main() -> int:
    parser = argparse.ArgumentParser(description="오프라인 번역 지시 2x2 벤치(재제작 13번)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
