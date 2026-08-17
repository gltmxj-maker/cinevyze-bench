#!/usr/bin/env python3
"""Run the length-instruction compliance benchmark against local Ollama.

The axis is the requested character count. Topic, role, tone, and the "no preamble" rule are
held fixed; only the number in the length sentence changes. A second, smaller axis varies how
the length is phrased (plain / strict / range) so that a single unlucky wording cannot be
mistaken for the model's general behaviour.

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

from length_score import score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("length_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-length-20260801"
# Target order is rotated per prompt so warm-up or drift cannot masquerade as a length effect.
TARGET_ROTATION_STEP = 1


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = data.get("length_targets") or []
    prompts = data.get("prompts") or []
    phrasings = data.get("phrasings") or {}
    if len(targets) != 4 or any(not isinstance(t, int) or t <= 0 for t in targets):
        raise ValueError("length_targets는 양의 정수 4개")
    if len(targets) != len(set(targets)):
        raise ValueError("length_targets 중복")
    if len(prompts) != 10:
        raise ValueError(f"prompts {len(prompts)}건 != 10")
    if set(phrasings) != {"plain", "strict", "range"}:
        raise ValueError("phrasings는 plain/strict/range 셋")
    for phrasing, template in phrasings.items():
        if "N" not in template:
            raise ValueError(f"{phrasing} 템플릿에 N 자리표시자 없음")
    seen: set[str] = set()
    for prompt in prompts:
        pid = prompt.get("id")
        if not pid or pid in seen:
            raise ValueError(f"중복/빈 prompt id: {pid}")
        seen.add(pid)
        if not str(prompt.get("topic", "")).strip() or not str(prompt.get("genre", "")).strip():
            raise ValueError(f"{pid} topic/genre 계약 오류")
    return data


# The length sentence is the ONLY part that varies between runs of the same prompt. Everything
# else — role, topic, tone, the ban on preamble and markdown — is byte-identical, otherwise the
# arms would be different writing tasks rather than different length instructions.
def build_prompt(prompt: dict[str, Any], target: int, phrasing_template: str) -> str:
    length_sentence = phrasing_template.replace("N", str(target))
    return (
        "당신은 한국어 글을 쓰는 작가입니다.\n"
        f"주제: {prompt['topic']}\n"
        f"길이: {length_sentence}\n\n"
        "규칙:\n"
        "- 한국어로 쓰세요.\n"
        "- 길이는 공백을 포함한 글자 수 기준입니다.\n"
        "- 본문만 출력하세요. 인사말, 머리말, '알겠습니다' 같은 대답, 글자 수 표기는 금지합니다.\n"
        "- 제목, 소제목, 목록 기호, 굵게 표시 같은 마크다운 장식을 쓰지 마세요.\n"
        "- 줄글로 쓰세요.\n"
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


def planned_runs(data: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    targets = data["length_targets"]
    prompts = data["prompts"]
    phrasings = data["phrasings"]
    runs: list[dict[str, Any]] = []
    if mode == "pilot":
        # Smallest plan that still exercises every target once: 3 prompts x 4 targets, plain only.
        for index, prompt in enumerate(prompts[:3]):
            for offset in range(len(targets)):
                target = targets[(index * TARGET_ROTATION_STEP + offset) % len(targets)]
                runs.append({"key": f"{prompt['id']}/{target}/plain", "prompt": prompt,
                             "target": target, "phrasing": "plain", "template": phrasings["plain"]})
        return runs
    # Full plan: every prompt x every target on the plain wording (the main grid), plus the
    # two alternative wordings on a fixed 3-prompt subset to check phrasing sensitivity.
    for index, prompt in enumerate(prompts):
        for offset in range(len(targets)):
            target = targets[(index * TARGET_ROTATION_STEP + offset) % len(targets)]
            runs.append({"key": f"{prompt['id']}/{target}/plain", "prompt": prompt,
                         "target": target, "phrasing": "plain", "template": phrasings["plain"]})
    for phrasing in ("strict", "range"):
        for index, prompt in enumerate(prompts[:3]):
            for offset in range(len(targets)):
                target = targets[(index * TARGET_ROTATION_STEP + offset) % len(targets)]
                runs.append({"key": f"{prompt['id']}/{target}/{phrasing}", "prompt": prompt,
                             "target": target, "phrasing": phrasing, "template": phrasings[phrasing]})
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
    """Target- and phrasing-wise comparison read back from the scored aggregate, never hand-written.

    The gate that reads `compare` exists to stop an article claiming a comparison the raw data
    does not contain, so the rows are derived from aggregate.json produced by length_score —
    if scoring has not run yet the list is simply empty.
    """
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for target, entry in (aggregate.get("by_target") or {}).items():
        rows.append({"metric": "median_chars_vs_target", "arm": f"{target}자",
                     "value": entry["median_chars"], "ratio": entry["median_ratio"],
                     "within_10": entry["within_10"], "total": entry["total"]})
    for phrasing, entry in (aggregate.get("by_phrasing") or {}).items():
        rows.append({"metric": "within_10_rate_by_phrasing", "arm": phrasing,
                     "value": round(entry["within_10_rate"], 4), "ratio": entry["median_ratio"],
                     "within_10": entry["within_10"], "total": entry["total"]})
    return rows


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any], targets: list[int]) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "prompt_id": row["prompt_id"],
            "genre": row["genre"],
            "target": row["target"],
            "phrasing": row["phrasing"],
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
        "method": "LEN-COMPLY on L01..L10 (targets 100/300/500/800자, phrasings plain/strict/range)",
        "access": "local",
        "model": model,
        "generated_by": "run_length_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "axis": "length_instruction",
        "targets": targets,
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
    # JSON is valid YAML; atomic replace prevents a killed process leaving a truncated manifest.
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
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": {
            "key": plan[0]["key"],
            "prompt": build_prompt(plan[0]["prompt"], plan[0]["target"], plan[0]["template"]),
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
        prompt_text = build_prompt(row["prompt"], row["target"], row["template"])
        response, attempts, infra_error = generate(args.base_url, args.model, prompt_text, args.timeout)
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
            f"prompt_id: {row['prompt']['id']}\ntarget: {row['target']}\nphrasing: {row['phrasing']}\n\n"
            f"[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
            f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "prompt_id": row["prompt"]["id"],
            "genre": row["prompt"]["genre"],
            "topic": row["prompt"]["topic"],
            "target": row["target"],
            "phrasing": row["phrasing"],
            "prompt": prompt_text,
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
        write_run_yaml(args.run_dir, args.model, metadata, data["length_targets"])
        print(f"[{position}/{len(pending)}] {row['key']} elapsed={invocation['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    write_run_yaml(args.run_dir, args.model, metadata, data["length_targets"])
    results, aggregate = score_run_dir(args.run_dir)
    # Scoring writes aggregate.json, which is where the compare rows come from — so the
    # manifest is rewritten once more to carry them.
    write_run_yaml(args.run_dir, args.model, metadata, data["length_targets"])
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
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
