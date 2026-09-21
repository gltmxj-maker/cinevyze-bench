#!/usr/bin/env python3
"""AI 문서요약 정확도 벤치 — 2026-09-22 재제작 14번(문서요약 글 R2).

공개판 주기 — 내부 판과 두 곳만 다르다:
  ① testset 교차검사 대신 입력 SHA-256 고정값 검증이 문서 무결성을 보장한다.
  ② inferhub 자격증명은 ~/.opencodex/config.json 대신 INFERHUB_API_KEY 환경변수로 받는다.

Design contract(집필 서브 Schrodinger 설계 2026-09-22, b-seat-split 위임):
- 72 runs = 2 models x 4 docs(D1~D4) x 3 arms(S3/S5/S8) x 3 reps.
  D1은 TXT-01 커피 원문을 바이트 그대로(D1xS5 프롬프트가 testset.build_prompt("TXT-01")와
  바이트 일치 — 실행 전 검증·불일치 exit 2). D2~D4는 자작 합성 문서(장애회고/요금정책/의사결정메모).
- 회전 순서 고정 — 문서/팔이 연속 같은 위치를 독점하지 않게(순서 효과 방어).
  셀 내 모델 순서는 (문서번호+팔번호+반복번호) 홀짝으로 교대 — 모든 셀이 양쪽 선두를 경험.
- 모든 셀은 독립 단일턴 요청·직렬 실행. 워밍업 없음 — 첫 호출부터 측정에 포함.
- Ollama: temperature=0, seed=14, keep_alive=0(VRAM 상주 금지). elapsed에 콜드 로드 포함.
- 채점은 결정론적(summary_score.py). break 마커는 축별 1회(핵심 보존·근거 안전) 최대 2개 —
  게재 이미지 예산 3장(break 전부+agg 1) 준수.
- This script is the write-origin for run.yaml. 매 시행의 프롬프트 전문·원응답·API 메타를
  raw/ 에 저장하고 그 기록만으로 run.yaml을 만든다 — 사람이 수치를 적어 넣을 자리가 없다.
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
from summary_score import TARGETS, score_response

HARNESS_VERSION = "1.0"
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "summary-gemma3-vs-gemini-20260922"
LOCAL_MODEL = "gemma3:4b"
CLOUD_MODEL = "Gemini 3.1 Pro (High)"
TEMPERATURE = 0
SEED = 14

CASES_FILE = Path(__file__).with_name("summary_bench_cases.json")
INPUT_FILES = None  # 공개판: cases JSON 에서 로드
INPUT_SHA256 = {
    "D1": "f30755f5044fb48695d658ac7ffa476cbf5d6bf995245f8fd10621394de20678",
    "D2": "e2b17115cdb203b23b79092e47b10cc8bbb58831ea4162e64793e3f49556b535",
    "D3": "d4d49b83ab7d79089b27cf388547f565b6ba77414049826d5a271a46b55327c6",
    "D4": "d7f251bd52bb5a7e4fb19a0c394edb0ff793877502298a2cfd66bcfedf0a69bf",
}
DOC_LENGTHS = {"D1": (900, 1000), "D2": (1600, 2000), "D3": (3200, 4000), "D4": (6500, 8000)}

ARM_INSTRUCTIONS = {
    "S3": "다음 글을 핵심 3문장으로 요약해줘. 원문에 없는 내용은 추가하지 마.",
    "S5": "다음 글을 핵심 5문장으로 요약해줘. 원문에 없는 내용은 추가하지 마.",
    "S8": "다음 글을 핵심 8문장으로 요약해줘. 원문에 없는 내용은 추가하지 마.",
}
ARM_NUM = {"S3": 0, "S5": 1, "S8": 2}

# 회전(설계안 그대로): 반복마다 12개 셀(문서/팔) 순서가 뒤섞이도록 고정.
CELL_ORDER = {
    1: [("D1","S3"),("D2","S5"),("D3","S8"),("D4","S3"),("D1","S5"),("D2","S8"),
        ("D3","S3"),("D4","S5"),("D1","S8"),("D2","S3"),("D3","S5"),("D4","S8")],
    2: [("D4","S5"),("D3","S3"),("D2","S8"),("D1","S5"),("D4","S8"),("D3","S5"),
        ("D2","S3"),("D1","S8"),("D4","S3"),("D3","S8"),("D2","S5"),("D1","S3")],
    3: [("D2","S8"),("D4","S5"),("D1","S3"),("D3","S8"),("D2","S5"),("D4","S3"),
        ("D1","S8"),("D3","S5"),("D2","S3"),("D4","S8"),("D1","S5"),("D3","S3")],
}

def build_plan() -> list[tuple[str, str, str, int]]:
    """(provider, doc, arm, rep) 72행 — 셀 내 모델 순서는 홀짝 교대."""
    plan = []
    for rep in (1, 2, 3):
        for doc, arm in CELL_ORDER[rep]:
            local_first = ((int(doc[1]) + ARM_NUM[arm] + rep) % 2) == 0
            pair = [("local", doc, arm, rep), ("cloud", doc, arm, rep)]
            plan.extend(pair if local_first else pair[::-1])
    return plan

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def verify_inputs(base: Path) -> tuple[dict[str, str], dict[str, str]]:
    hashes = {}
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    docs = cases["docs"]
    for doc, text in docs.items():
        digest = _sha256_bytes(text.encode("utf-8"))
        if doc in INPUT_SHA256 and digest != INPUT_SHA256[doc]:
            raise SystemExit("[X] 입력 불일치 — " + doc + " SHA가 고정값과 다릅니다. 실행 중단.")
        lo, hi = DOC_LENGTHS[doc]
        if not (lo <= len(text) <= hi):
            raise SystemExit("[X] 길이 이탈 — " + doc + " 가 목표 범위(" + str(lo) + "~" + str(hi) + "자)를 벗어났습니다.")
        hashes[doc] = digest
    # 공개판: 위 SHA-256 고정값 검증이 문서 무결성을 보장한다(내부 testset 교차검사 대체).
    return hashes, docs

def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("API 응답 최상위 값이 JSON 객체가 아님")
    return decoded

def ollama_generate(base_url: str, model: str, prompt: str, timeout: int):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "keep_alive": 0, "options": {"temperature": TEMPERATURE, "seed": SEED}}
    attempts = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": None})
            return response, attempts, None
        except urllib.error.HTTPError as exc:
            error = "HTTPError " + str(exc.code) + ": " + str(exc.reason)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            transient = exc.code in (408, 425, 429) or exc.code >= 500
            if not transient or attempt == 2:
                return None, attempts, error
            time.sleep(1)
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            error = type(exc).__name__ + ": " + str(exc)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            if attempt == 2:
                return None, attempts, error
            time.sleep(1)
    return None, attempts, attempts[-1]["error"]

def gemini_generate(prompt: str, model: str, timeout: int):
    import agy_adapter
    attempts = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            text = agy_adapter.run_gemini(prompt, model=model, timeout=timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": None})
            return text, attempts, None
        except Exception as exc:  # noqa: BLE001 — 어댑터 예외 전부 infra 후보
            error = type(exc).__name__ + ": " + str(exc)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            if attempt == 2:
                return None, attempts, error
            time.sleep(2)
    return None, attempts, attempts[-1]["error"]

INFERHUB_MODEL = "gemini-3.8-flash"
INFERHUB_LABEL = "Gemini 3.8 Flash (High)"

def inferhub_generate(prompt: str, timeout: int):
    """inferhub ag/gemini-3.8-flash-high 호출 — agy 구독 한도 보완 경로(2026-09-22).

    모델은 응답 model 필드로 ag/gemini-3.8-flash-high 로 확인됨(프로브 2026-09-22 10.7s).
    reasoning_effort 는 프로바이더가 강제 매핑한다고 문서화된 값이 없으므로 지정하지 않고
    UNKNOWN 으로 run.yaml 에 기록한다.
    """
    api_key = os.environ.get("INFERHUB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("INFERHUB_API_KEY 환경변수가 필요합니다(공개판: 키는 각자 준비)")
    url = os.environ.get("INFERHUB_BASE_URL", "https://api.inferhub.dev/v1").rstrip("/") + "/chat/completions"
    payload = {"model": INFERHUB_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 4096}
    attempts = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + api_key},
                method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            text = decoded["choices"][0]["message"].get("content", "")
            attempts.append({"attempt": attempt,
                             "elapsed_s": round(time.monotonic() - started, 3),
                             "error": None})
            return text, attempts, None
        except Exception as exc:  # noqa: BLE001 — 어댑터 예외 전부 infra 후보
            error = type(exc).__name__ + ": " + str(exc)
            attempts.append({"attempt": attempt,
                             "elapsed_s": round(time.monotonic() - started, 3),
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

def output_hygiene(raw_text: str) -> dict[str, Any]:
    ansi_count = len(ANSI_RE.findall(raw_text))
    clean = ANSI_RE.sub("", raw_text)
    control_count = len(CONTROL_RE.findall(clean))
    return {"ansi_escape_count": ansi_count, "control_char_count": control_count,
            "output_clean_pass": ansi_count == 0 and control_count == 0, "clean_text": clean}

def _model_label(provider: str, inferhub: bool = False) -> str:
    if provider == "local":
        return LOCAL_MODEL
    return INFERHUB_LABEL if inferhub else CLOUD_MODEL

def _existing(run_dir: Path) -> set:
    keys = set()
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys

def _seen_breaks_from_disk(run_dir: Path) -> dict[str, bool]:
    """재개 시 이미 발사된 break 축을 디스크 score에서 복원한다."""
    seen = {"core_coverage": False, "grounding": False}
    for path in (run_dir / "raw").glob("*-score.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("valid_response"):
            continue
        if row.get("core_coverage_pass") is False:
            seen["core_coverage"] = True
        if row.get("grounding_rule_pass") is False:
            seen["grounding"] = True
    return seen

def write_run_yaml(run_dir: Path, metadata: dict[str, Any], input_hashes: dict[str, str], cloud_inferhub: bool = False) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "task": row["task"], "provider": row["provider"], "model": row["model"],
            "doc": row["doc"], "arm": row["arm"], "repetition": row["repetition"],
            "output_file": row["transcript_file"], "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_total_s"), "score_file": row.get("score_file"),
            "infra_error": row.get("infra_error"),
        })
    compare = []
    agg_path = run_dir / "aggregate.json"
    if agg_path.exists():
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        for side in ("local", "cloud"):
            stats = agg["by_provider"][side]
            for metric in ("summary_contract_pass", "core_coverage_pass",
                           "grounding_rule_pass", "sentence_count_pass"):
                compare.append({"metric": metric, "tool": _model_label(side, cloud_inferhub),
                                "value": stats[metric], "total": stats["valid_outputs"]})
            compare.append({"metric": "median_elapsed_total_s", "tool": _model_label(side, cloud_inferhub),
                            "value": stats["median_elapsed_total_s"], "total": stats["valid_outputs"]})
        for doc_id, entry in sorted((agg.get("by_doc") or {}).items()):
            for side in ("local", "cloud"):
                compare.append({"metric": "contract_by_doc", "tool": _model_label(side, cloud_inferhub),
                                "value": entry[side]["contract_pass"],
                                "total": entry[side]["valid_outputs"], "doc": doc_id})
        for arm_id, entry in sorted((agg.get("by_arm") or {}).items()):
            for side in ("local", "cloud"):
                compare.append({"metric": "contract_by_arm", "tool": _model_label(side, cloud_inferhub),
                                "value": entry[side]["contract_pass"],
                                "total": entry[side]["valid_outputs"], "arm": arm_id})
    payload = {
        "tool": "ollama+gemini-cli",
        "date": dt.date.today().isoformat(),
        "method": "TXT-01(D1)+D2-INCIDENT+D3-PRICING+D4-DECISION x S3/S5/S8",
        "access": "local+cli",
        "model": (LOCAL_MODEL + " vs " + INFERHUB_LABEL + " (inferhub ag/gemini-3.8-flash-high)"
                     if cloud_inferhub else
                     LOCAL_MODEL + " vs " + CLOUD_MODEL),
        "generated_by": "run_summary_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동) · Antigravity 구독 CLI(공식 클라이언트·OAuth)",
        "keep_alive": 0,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "cloud_decoding": "UNKNOWN",
        "request_parallelism": 1,
        "axis": "summary_budget_3_5_8_sentences_local_vs_cloud_same_day",
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
    rows = []
    for path in sorted((run_dir / "raw").glob("*-score.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    by_provider = {}
    for side in ("local", "cloud"):
        subset = [r for r in rows if r.get("provider") == side]
        valid = [r for r in subset if r.get("valid_response")]
        elapsed = sorted(r["elapsed_total_s"] for r in valid if r.get("elapsed_total_s") is not None)
        med = elapsed[len(elapsed) // 2] if elapsed else None
        by_provider[side] = {
            "outputs": len(subset), "valid_outputs": len(valid),
            "summary_contract_pass": sum(1 for r in valid if r.get("summary_contract_pass")),
            "core_coverage_pass": sum(1 for r in valid if r.get("core_coverage_pass")),
            "grounding_rule_pass": sum(1 for r in valid if r.get("grounding_rule_pass")),
            "sentence_count_pass": sum(1 for r in valid if r.get("sentence_count_pass")),
            "output_clean_pass": sum(1 for r in valid if r.get("output_clean_pass")),
            "median_elapsed_total_s": med,
            "elapsed_total_s": [r.get("elapsed_total_s") for r in subset],
            "invalid_cells": len(subset) - len(valid),
        }
    by_doc, by_arm = {}, {}
    for doc_id in INPUT_FILES:
        entry = {}
        for side in ("local", "cloud"):
            subset = [r for r in rows if r.get("provider") == side and r.get("doc") == doc_id]
            valid = [r for r in subset if r.get("valid_response")]
            entry[side] = {"outputs": len(subset), "valid_outputs": len(valid),
                           "contract_pass": sum(1 for r in valid if r.get("summary_contract_pass"))}
        by_doc[doc_id] = entry
    for arm_id in TARGETS:
        entry = {}
        for side in ("local", "cloud"):
            subset = [r for r in rows if r.get("provider") == side and r.get("arm") == arm_id]
            valid = [r for r in subset if r.get("valid_response")]
            entry[side] = {"outputs": len(subset), "valid_outputs": len(valid),
                           "contract_pass": sum(1 for r in valid if r.get("summary_contract_pass"))}
        by_arm[arm_id] = entry
    infra_events = sum(1 for r in rows if r.get("infra_error"))
    return {"generated_by": "run_summary_bench.py:aggregate", "rows_read": len(rows),
            "by_provider": by_provider, "by_doc": by_doc, "by_arm": by_arm,
            "infra_events": infra_events}

def run_benchmark(args: argparse.Namespace) -> int:
    base = Path(__file__).resolve().parent
    input_hashes, doc_texts = verify_inputs(base)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    existing = _existing(args.run_dir)
    plan = build_plan()
    pending = [row for row in plan if (row[0] + "-" + row[1] + "-" + row[2] + "-r" + str(row[3])) not in existing]
    print("planned=" + str(len(plan)) + " existing=" + str(len(plan) - len(pending))
          + " running_now=" + str(len(pending)), flush=True)

    metadata = {}
    if any(p[0] == "local" for p in pending):
        metadata = model_metadata(args.base_url, args.model, args.timeout)
        (args.run_dir / "model-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


    prompts = {(doc, arm): ARM_INSTRUCTIONS[arm] + "\n\n" + doc_texts[doc]
               for doc in INPUT_FILES for arm in TARGETS}
    prompt_hashes = {doc + "-" + arm: _sha256_bytes(p.encode("utf-8"))
                     for (doc, arm), p in prompts.items()}

    seen_break = _seen_breaks_from_disk(args.run_dir)
    previous_cell = None
    first_turn_done = bool(existing)
    # 스템 충돌 방지: 개수가 아니라 기존 최대 스템 + 1 (2026-09-22 복원 런에서 070/071을 덮어쓴 사고 수리)
    existing_stems = [int(path.name.split("-")[0]) for path in (args.run_dir / "raw").glob("*-invocation.json")]
    next_id = max(existing_stems) if existing_stems else 0
    for position, (provider, doc, arm, rep) in enumerate(pending, 1):
        next_id += 1
        key = provider + "-" + doc + "-" + arm + "-r" + str(rep)
        model = LOCAL_MODEL if provider == "local" else (
            INFERHUB_LABEL if args.cloud_inferhub else CLOUD_MODEL)
        target = TARGETS[arm]

        if previous_cell != (provider, doc, arm, rep):
            if not first_turn_done and previous_cell is None:
                live_mark("turn", "측정 시작 — " + model + "/" + doc + "/" + arm
                          + " · 반복 " + str(rep) + "/3 · 독립 요청 · 목표 " + str(target) + "문장"
                          + (" · temperature=0 · seed=14 · keep_alive=0" if provider == "local" else ""))
                first_turn_done = True
            else:
                prev = previous_cell or ("", "", "", 0)
                live_mark("turn", "측정 조건 전환 — " + _model_label(prev[0], args.cloud_inferhub) + "/" + prev[1] + "/" + prev[2]
                          + " → " + model + "/" + doc + "/" + arm
                          + " · 반복 " + str(rep) + "/3 · 독립 요청")
            previous_cell = (provider, doc, arm, rep)

        prompt_text = prompts[(doc, arm)]
        if provider == "local":
            response, attempts, infra_error = ollama_generate(
                args.base_url, args.model, prompt_text, args.timeout)
            response_text = (response or {}).get("response") if response else None
            api_metrics = ({k: response.get(k) for k in (
                "done_reason", "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration")} if response else None)
        else:
            if args.cloud_inferhub:
                response_text, attempts, infra_error = inferhub_generate(
                    prompt_text, args.cloud_timeout)
            else:
                response_text, attempts, infra_error = gemini_generate(
                    prompt_text, args.cloud_model, args.cloud_timeout)
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
            "key": key, "provider": provider, "model": model,
            "task": "SUMMARY-" + doc, "doc": doc, "arm": arm,
            "target_sentence_count": target, "repetition": rep,
            "plan_position": position, "attempt_count": len(attempts),
            "elapsed_total_s": elapsed_total, "infra_error": infra_error,
            "valid_response": bool(raw_text.strip()) and not infra_error,
            "raw_sha256": _sha256_bytes(raw_text.encode("utf-8")) if raw_text else None,
            "raw_bytes": len(raw_text.encode("utf-8")) if raw_text else 0,
            "ansi_escape_count": hygiene["ansi_escape_count"],
            "control_char_count": hygiene["control_char_count"],
            "output_clean_pass": hygiene["output_clean_pass"],
            "summary_contract_pass": False,
        }
        if score["valid_response"]:
            try:
                task_score = score_response(doc, arm, hygiene["clean_text"])
                score.update(task_score)
            except Exception as exc:  # noqa: BLE001 — 채점기 예외는 infra
                live_mark("infra", "측정 흔들림 — " + key + " · 채점/기록 실패=" + type(exc).__name__)
                score["valid_response"] = False
                score["infra_error"] = "scorer_error: " + type(exc).__name__ + ": " + str(exc)
        else:
            reason = infra_error or "빈 응답"
            kind = "rate-limit(exit 53)" if "53" in str(infra_error) else (type(None).__name__ if infra_error is None else infra_error.split(":")[0])
            if len(attempts) >= 2:
                live_mark("infra", "측정 흔들림 — " + model + "/" + doc + "/" + arm + "/r" + str(rep)
                          + " · " + reason + " · 2회 실패 · 해당 셀 무효")
            else:
                live_mark("infra", "측정 흔들림 — " + model + "/" + doc + "/" + arm + "/r" + str(rep)
                          + " · " + reason + " · 재시도 1/1")

        if score["valid_response"] and not score.get("core_coverage_pass") and not seen_break["core_coverage"]:
            seen_break["core_coverage"] = True
            live_mark("break", "핵심 보존 첫 실패 — " + model + "/" + doc + "/" + arm + "/r" + str(rep)
                      + " · 핵심 슬롯 " + str(score.get("core_coverage_count")) + "/5 · 누락 "
                      + ",".join(score.get("missing_slot_ids", [])))
        if score["valid_response"] and not score.get("grounding_rule_pass") and not seen_break["grounding"]:
            seen_break["grounding"] = True
            live_mark("break", "근거 안전 첫 실패 — " + model + "/" + doc + "/" + arm + "/r" + str(rep)
                      + " · 허용 밖 숫자 " + str(score.get("unsupported_numeric_count", 0)) + "개 · 모순·폐기 "
                      + str(score.get("contradiction_retired_count", 0)) + "개 · 미매핑 문장 "
                      + str(score.get("unmapped_count", 0)) + "개")

        (args.run_dir / score_rel).write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
        transcript = ("run_id: " + key + "\nprovider: " + provider + " · model: " + model
                      + " · doc: " + doc + " · arm: " + arm + "(목표 " + str(target) + "문장) · rep: " + str(rep)
                      + "\n\n[PROMPT sha256=" + prompt_hashes[doc + "-" + arm] + "]\n" + prompt_text
                      + "\n\n[RAW RESPONSE]\n" + raw_text + "\n")
        if api_metrics:
            transcript += ("\n[API METADATA]\n" + json.dumps(api_metrics, ensure_ascii=False, indent=2) + "\n")
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id, "key": key, "provider": provider, "model": model,
            "task": "SUMMARY-" + doc, "doc": doc, "arm": arm, "repetition": rep,
            "prompt": prompt_text,
            "payload": ({"model": args.model, "stream": False, "keep_alive": 0,
                         "options": {"temperature": TEMPERATURE, "seed": SEED}}
                        if provider == "local" else
                        ({"cli": "inferhub", "model": INFERHUB_MODEL,
                          "resolved_model": "ag/gemini-3.8-flash-high"}
                         if args.cloud_inferhub else
                         {"cli": "agy", "model": args.cloud_model})),
            "attempts": attempts, "elapsed_total_s": elapsed_total,
            "infra_error": infra_error, "response_file": response_rel,
            "transcript_file": transcript_rel, "score_file": score_rel,
            "api_metrics": api_metrics,
        }
        (args.run_dir / invocation_rel).write_text(json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_yaml(args.run_dir, metadata, input_hashes, args.cloud_inferhub)
        print("[" + str(position) + "/" + str(len(pending)) + "] " + key
              + " elapsed=" + str(elapsed_total) + "s"
              + (" ERROR=" + str(infra_error) if infra_error else ""), flush=True)

    agg = aggregate(args.run_dir)
    (args.run_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_yaml(args.run_dir, metadata, input_hashes, args.cloud_inferhub)

    local = agg["by_provider"]["local"]
    cloud = agg["by_provider"]["cloud"]
    invalid_total = local["invalid_cells"] + cloud["invalid_cells"]
    if invalid_total:
        live_mark("agg", "집계 불완전 — 유효 로컬 " + str(local["valid_outputs"]) + "/36·클라우드 "
                  + str(cloud["valid_outputs"]) + "/36 · 최종 무효 " + str(invalid_total)
                  + "셀 · infra " + str(agg["infra_events"]) + "건 · 모델 비교 판정 보류")
    else:
        live_mark("agg", "같은 날 72회 집계 — 완전통과 로컬 " + str(local["summary_contract_pass"])
                  + "/36·클라우드 " + str(cloud["summary_contract_pass"]) + "/36 · 핵심보존 "
                  + str(local["core_coverage_pass"]) + ":" + str(cloud["core_coverage_pass"])
                  + " · 근거안전 " + str(local["grounding_rule_pass"]) + ":" + str(cloud["grounding_rule_pass"])
                  + " · 문장수 " + str(local["sentence_count_pass"]) + ":" + str(cloud["sentence_count_pass"])
                  + " · infra " + str(agg["infra_events"]) + "건")
    print(json.dumps(agg, ensure_ascii=False, indent=2), flush=True)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--cloud-model", default=CLOUD_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cloud-timeout", type=int, default=300)
    parser.add_argument("--cloud-inferhub", action="store_true",
                        help="클라우드 팔을 agy 대신 inferhub gemini-3.8-flash 로 호출(한도 보완·2026-09-22 시네 지시)")
    args = parser.parse_args()
    return run_benchmark(args)

if __name__ == "__main__":
    raise SystemExit(main())

