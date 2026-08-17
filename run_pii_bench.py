#!/usr/bin/env python3
"""PII-masking benchmark against local Ollama.

한국형 개인정보를 심은 문서를 주고 "지워 달라"고 시킨 뒤, 놓친 것(leak)과 지우면 안 될 것을
지운 것(over_mask)을 함께 잰다. 축 = 지시 방식 3팔(막연히 / 유형 열거 / 열거+표기형식) x 문서 6종.

세 팔 모두 "처리한 문서 전체를 출력해 주세요"라는 같은 꼬리말을 붙인다. 그러지 않으면 팔 간 차이가
'무엇을 지우라 했는가'가 아니라 '문서를 돌려줬는가'로 오염된다.

★문서·이름·번호는 전부 자체 작성한 가상값이다. 주민등록번호·카드번호는 형식만 맞춘 무효값,
전자우편은 문서 전용 예약 도메인(.example)을 쓴다.

This script is the write-origin for run.yaml: 매 시행의 프롬프트 전문·원응답·API 메타데이터를
raw/에 저장하고 그 기록에서만 run.yaml을 만든다.
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

from pii_score import PII_TYPES, score_run_dir, validate_cases

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("pii_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-pii-20260806"
ARMS = ("simple", "enumerated", "format")
PILOT_DOCS = 2


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data["prompt"]["arms"]) != set(ARMS):
        raise ValueError(f"arms는 {ARMS} 셋")
    validate_cases(data)
    return data


def build_prompt(data: dict[str, Any], arm: str, document: dict[str, Any]) -> str:
    prompt = data["prompt"]
    return (f"{prompt['preamble']}\n\n{prompt['arms'][arm]}\n{prompt['tail']}\n\n"
            f"---\n{document['text']}\n---")


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


def generate(base_url: str, model: str, prompt: str, num_ctx: int,
             timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None, dict[str, Any]]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
               "options": {"num_ctx": num_ctx}}
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": None})
            return response, attempts, None, payload
        except urllib.error.HTTPError as exc:
            error = f"HTTPError {exc.code}: {exc.reason}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            transient = exc.code in {408, 425, 429} or exc.code >= 500
            if not transient or attempt == 2:
                return None, attempts, error, payload
            time.sleep(1)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            if attempt == 2:
                return None, attempts, error, payload
            time.sleep(1)
    return None, attempts, attempts[-1]["error"], payload


def _existing(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys


def _next_id(run_dir: Path) -> int:
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        maximum = max(maximum, int(row.get("run_id", 0)))
    return maximum


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compare_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Read back from the scored aggregate — never hand-written."""
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for arm, entry in (aggregate.get("by_arm") or {}).items():
        rows.append({"metric": "leak_rate_by_arm", "arm": arm, "value": entry["leak_rate"],
                     "leaked": entry["pii_leaked"], "total": entry["pii_instances"],
                     "over_mask_rate": entry["over_mask_rate"], "clean_rate": entry["clean_rate"],
                     "refused": entry["refused"], "unchanged": entry["unchanged"]})
    for name, entry in (aggregate.get("by_type") or {}).items():
        rows.append({"metric": "leak_rate_by_type", "arm": name, "value": entry["leak_rate"],
                     "leaked": entry["leaked"], "total": entry["instances"], "partial": entry["partial"]})
    for model, entry in (aggregate.get("by_model") or {}).items():
        rows.append({"metric": "leak_rate_by_model", "arm": model, "value": entry["leak_rate"],
                     "leaked": entry["pii_leaked"], "total": entry["pii_instances"],
                     "over_mask_rate": entry["over_mask_rate"]})
    for token, entry in (aggregate.get("must_keep_detail") or {}).items():
        rows.append({"metric": "must_keep_drop_rate", "arm": token, "value": entry["drop_rate"],
                     "dropped": entry["dropped"], "total": entry["n"]})
    return rows


def write_run_yaml(run_dir: Path, metadata: dict[str, Any], data: dict[str, Any], num_ctx: int) -> None:
    entries = []
    models: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        models.add(row["model"])
        entries.append({
            "key": row["key"],
            "arm": row["arm"],
            "model": row["model"],
            "doc_id": row["doc_id"],
            "repeat": row["repeat"],
            "doc_chars": row["doc_chars"],
            "output_file": row["transcript_file"],
            "response_file": row["response_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_s"),
            "num_ctx": row.get("num_ctx"),
            "prompt_eval_count": (row.get("api_metrics") or {}).get("prompt_eval_count"),
            "eval_count": (row.get("api_metrics") or {}).get("eval_count"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": ("SGR-PIIMASK — 한국형 개인정보 7종을 심은 자체 작성 문서 6종에 마스킹을 시키고, "
                   "원본 문자열 잔존(leak)과 보존 대상 토큰 소실(over_mask)을 결정론 판정한다. "
                   "축 = 지시 방식 3팔(막연히 / 유형 열거 / 열거+표기형식) x 문서 6종 x 반복. "
                   "작업 거부 회차는 유출이 없었으므로 leak 분모에서 뺀다."),
        "access": "local",
        "model": sorted(models)[0] if models else None,
        "models": sorted(models),
        "generated_by": "run_pii_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "num_ctx": num_ctx,
        "axis": "instruction_arm_x_document",
        "arms": list(ARMS),
        "pii_types": list(PII_TYPES),
        "documents": [document["id"] for document in data["documents"]],
        "synthetic_data": "문서·이름·번호 전부 자체 작성 가상값(주민등록번호·카드번호는 형식만 맞춘 무효값)",
        "compare": _compare_rows(run_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "ollama_models": os.environ.get("OLLAMA_MODELS"),
            "ollama_max_loaded_models": os.environ.get("OLLAMA_MAX_LOADED_MODELS"),
            "ollama_num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL"),
        },
        "model_metadata": metadata,
        "runs": entries,
    }
    target_path = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target_path)


def _write_records(run_dir: Path, run_id: int, record: dict[str, Any], prompt_text: str,
                   response: dict[str, Any] | None, response_text: str) -> dict[str, Any]:
    stem = f"{run_id:03d}"
    response_rel = f"raw/{stem}-response.txt"
    transcript_rel = f"raw/{stem}-output.txt"
    (run_dir / response_rel).write_text(response_text, encoding="utf-8")
    header = "\n".join(f"{key}: {value}" for key, value in record.items()
                       if key in ("key", "arm", "model", "doc_id", "repeat", "num_ctx"))
    (run_dir / transcript_rel).write_text(
        f"run_id: {run_id}\n{header}\n\n[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
        f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8")
    record.update({
        "run_id": run_id,
        "response_file": response_rel,
        "transcript_file": transcript_rel,
        "api_metrics": {key: (response or {}).get(key) for key in (
            "done_reason", "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration"
        )} if response else None,
    })
    (run_dir / f"raw/{stem}-gen.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    if args.dry_run:
        document = data["documents"][0]
        print(json.dumps({
            "documents": [d["id"] for d in data["documents"]],
            "pii_instances_per_pass": sum(len(d["pii"]) for d in data["documents"]),
            "must_keep_per_pass": sum(len(d["must_keep"]) for d in data["documents"]),
            "planned_calls_per_model": len(ARMS) * len(data["documents"]) * args.repeats,
            "sample_prompt": build_prompt(data, "format", document),
        }, ensure_ascii=False, indent=2))
        return 0

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    # 채점기는 이 스냅샷을 읽는다 — 케이스 파일이 나중에 바뀌어도 채점 기준은 실행 시점에 고정된다.
    (run_dir / "cases_snapshot.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    documents = data["documents"][:PILOT_DOCS] if args.mode == "pilot" else data["documents"]
    repeats = 1 if args.mode == "pilot" else args.repeats

    if args.mode == "pilot":
        (run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot", "documents": len(documents)}, ensure_ascii=False), encoding="utf-8")
    else:
        decision_path = run_dir / "pilot_decision.json"
        if not decision_path.exists():
            print("파일럿 판정 없음 — 같은 --run-dir에서 --mode pilot을 먼저 완료하세요.")
            return 3
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("proceed"):
            print(f"파일럿 중단조건 발동: {decision.get('reasons')}")
            return 3

    metadata_raw = model_metadata(args.base_url, args.model, args.timeout)
    (run_dir / "model_metadata").mkdir(exist_ok=True)
    (run_dir / "model_metadata" / f"{args.model.replace(':', '_')}.json").write_text(
        json.dumps({"modified_at": metadata_raw.get("modified_at"), "details": metadata_raw.get("details"),
                    "parameters": metadata_raw.get("parameters")}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    existing = _existing(run_dir)
    run_id = _next_id(run_dir)
    plan = []
    for arm in ARMS:
        for document in documents:
            for repeat in range(1, repeats + 1):
                key = f"{arm}/{args.model}/{document['id']}/r{repeat}"
                if key not in existing:
                    plan.append((key, arm, document, repeat))
    print(f"model={args.model} mode={args.mode} num_ctx={args.num_ctx} pending={len(plan)}", flush=True)

    for index, (key, arm, document, repeat) in enumerate(plan, 1):
        run_id += 1
        prompt_text = build_prompt(data, arm, document)
        response, attempts, infra_error, payload = generate(
            args.base_url, args.model, prompt_text, args.num_ctx, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        record = {
            "key": key,
            "arm": arm,
            "model": args.model,
            "doc_id": document["id"],
            "doc_kind": document["kind"],
            "doc_chars": len(document["text"]),
            "repeat": repeat,
            "pii_planted": len(document["pii"]),
            "must_keep_planted": len(document["must_keep"]),
            "num_ctx": args.num_ctx,
            "payload": {k: v for k, v in payload.items() if k != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(run_dir, run_id, record, prompt_text, response, response_text)
        print(f"[{index}/{len(plan)}] {key} {record['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    rows, aggregate = score_run_dir(run_dir)
    metadata = {}
    metadata_dir = run_dir / "model_metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.glob("*.json")):
            metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    write_run_yaml(run_dir, metadata, data, args.num_ctx)
    print(json.dumps({key: aggregate[key] for key in ("overall", "by_arm", "by_type", "infra_errors")},
                     ensure_ascii=False, indent=2))
    print(f"rows_total={len(rows)}")
    if args.mode == "pilot" and (run_dir / "pilot_decision.json").exists():
        decision = json.loads((run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
        if not decision["proceed"]:
            print(f"파일럿 중단조건: {decision['reasons']}")
            return 3
    return 0 if aggregate["infra_errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
