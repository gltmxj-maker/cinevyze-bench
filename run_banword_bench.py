#!/usr/bin/env python3
"""Run the banned-word (negative instruction) compliance benchmark against local Ollama.

The axis is the ban itself: the writing task, the length request, the tone rules and the topic are
byte-identical across arms, and only the "do not write these words" sentence changes — how many
words it lists (1/3/5) and where it sits in the prompt (before the task or after the rules). A
control arm carries no ban sentence at all, which is what makes the numbers interpretable: a word
that was never going to be written is not evidence that the instruction worked.

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

from banword_score import score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("banword_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-banword-20260816"
LENGTH_REQUEST = "200자 내외로 써 주세요."
# Held fixed across every arm, control included, so the ban sentence is the only difference.
FIXED_RULES = (
    "- 한국어로 쓰세요.",
    "- 본문만 출력하세요. 인사말, 머리말, '알겠습니다' 같은 대답은 금지합니다.",
    "- 제목, 소제목, 목록 기호, 굵게 표시 같은 마크다운 장식을 쓰지 마세요.",
    "- 줄글로 쓰세요.",
)


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") or []
    counts = data.get("ban_counts") or []
    positions = data.get("positions") or []
    subset = data.get("position_subset") or []
    sentence = data.get("ban_sentence") or ""
    if len(prompts) != 8:
        raise ValueError(f"prompts {len(prompts)}건 != 8")
    if counts != sorted(set(counts)) or not counts:
        raise ValueError("ban_counts는 오름차순 중복 없는 목록")
    if positions != ["front", "back"]:
        raise ValueError("positions는 front/back 둘")
    if "{WORDS}" not in sentence:
        raise ValueError("ban_sentence에 {WORDS} 자리표시자 없음")
    if int(data.get("control_reps", 0)) < 1:
        raise ValueError("control_reps는 1 이상 — 통제군 없이는 준수율 해석 불가")
    ids: set[str] = set()
    for prompt in prompts:
        pid = prompt.get("id")
        if not pid or pid in ids:
            raise ValueError(f"중복/빈 prompt id: {pid}")
        ids.add(pid)
        banned = prompt.get("banned") or []
        if len(banned) != max(counts) or len(set(banned)) != len(banned):
            raise ValueError(f"{pid} banned 목록은 중복 없는 {max(counts)}개")
        topic = str(prompt.get("topic", ""))
        if not topic.strip() or not str(prompt.get("genre", "")).strip():
            raise ValueError(f"{pid} topic/genre 계약 오류")
        # A banned stem that already sits in the task text would be an impossible instruction,
        # and impossibility is a different experiment from disobedience.
        for word in banned:
            if word in topic:
                raise ValueError(f"{pid} 금지어 '{word}'가 주제문에 이미 있음")
    for pid in subset:
        if pid not in ids:
            raise ValueError(f"position_subset의 {pid}가 prompts에 없음")
    return data


def build_prompt(prompt: dict[str, Any], banned_shown: list[str], position: str, sentence: str) -> str:
    ban_line = ""
    if banned_shown:
        words = ", ".join(f'"{word}"' for word in banned_shown)
        ban_line = sentence.replace("{WORDS}", words)
    lines = ["당신은 한국어 글을 쓰는 작가입니다.", f"주제: {prompt['topic']}"]
    if ban_line and position == "front":
        lines.append(ban_line)
    lines.append(f"길이: {LENGTH_REQUEST}")
    lines.extend(["", "규칙:", *FIXED_RULES])
    if ban_line and position == "back":
        lines.extend(["", ban_line])
    return "\n".join(lines) + "\n"


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


def planned_runs(data: dict[str, Any], mode: str, control_reps: int = 0) -> list[dict[str, Any]]:
    prompts = data["prompts"]
    counts = data["ban_counts"]
    subset = set(data["position_subset"])
    # A slow model can be given fewer control repeats than the cases file asks for; the arm still
    # has to exist (the base rate is what makes the ban rate readable), only its n shrinks.
    reps = int(control_reps or data["control_reps"])
    runs: list[dict[str, Any]] = []

    def row(prompt: dict[str, Any], arm: str, position: str, shown: list[str], key: str) -> dict[str, Any]:
        return {"key": key, "prompt": prompt, "arm": arm, "position": position, "shown": shown}

    if mode == "pilot":
        # Smallest plan that still exercises the control arm, the shortest ban and the longest one.
        for prompt in prompts[:3]:
            runs.append(row(prompt, "control", "none", [], f"{prompt['id']}/control/1"))
            for count in (counts[0], counts[-1]):
                runs.append(row(prompt, f"ban{count}", "front", prompt["banned"][:count],
                                f"{prompt['id']}/ban{count}/front"))
        return runs

    for prompt in prompts:
        for rep in range(1, reps + 1):
            runs.append(row(prompt, "control", "none", [], f"{prompt['id']}/control/{rep}"))
    for prompt in prompts:
        for count in counts:
            runs.append(row(prompt, f"ban{count}", "front", prompt["banned"][:count],
                            f"{prompt['id']}/ban{count}/front"))
    # The position axis runs on a fixed subset — enough to see whether burying the ban at the end
    # changes anything, without doubling the whole grid.
    for prompt in prompts:
        if prompt["id"] not in subset:
            continue
        for count in counts:
            runs.append(row(prompt, f"ban{count}", "back", prompt["banned"][:count],
                            f"{prompt['id']}/ban{count}/back"))
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
    """Arm- and position-wise comparison read back from the scored aggregate, never hand-written."""
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for arm, entry in (aggregate.get("by_arm") or {}).items():
        rows.append({"metric": "violated_rate_by_arm", "arm": arm, "value": entry["violated_rate"],
                     "excl_meta": entry["violated_rate_excl_meta"], "runs": entry["runs"]})
    for position, entry in (aggregate.get("by_position") or {}).items():
        rows.append({"metric": "violated_rate_by_position", "arm": position, "value": entry["violated_rate"],
                     "excl_meta": entry["violated_rate_excl_meta"], "runs": entry["runs"]})
    for state, entry in (aggregate.get("state_totals") or {}).items():
        rows.append({"metric": "word_presence_rate_by_state", "arm": state, "value": entry["rate"],
                     "excl_meta": None, "runs": entry["runs"]})
    return rows


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any], data: dict[str, Any],
                   num_ctx: int = 0) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "prompt_id": row["prompt_id"],
            "genre": row["genre"],
            "arm": row["arm"],
            "position": row["position"],
            "banned_shown": row["banned_shown"],
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
        "method": "BAN-COMPLY on B01..B08 (arms control/ban1/ban3/ban5, positions front/back)",
        "access": "local",
        "model": model,
        "generated_by": "run_banword_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "num_ctx": num_ctx or None,
        "request_parallelism": 1,
        "axis": "banned_word_instruction",
        "ban_counts": data["ban_counts"],
        "control_reps": data["control_reps"],
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
    plan = planned_runs(data, args.mode, args.control_reps)
    if not plan:
        raise ValueError("실행 계획이 비어 있음")
    if len({row["key"] for row in plan}) != len(plan):
        raise ValueError("실행 계획에 중복 키 존재")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": {
            "key": plan[0]["key"],
            "prompt": build_prompt(plan[0]["prompt"], plan[0]["shown"], plan[0]["position"], data["ban_sentence"]),
        }, "sample_ban": {
            "key": plan[-1]["key"],
            "prompt": build_prompt(plan[-1]["prompt"], plan[-1]["shown"], plan[-1]["position"], data["ban_sentence"]),
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

    for index, row in enumerate(pending, 1):
        next_id += 1
        prompt_text = build_prompt(row["prompt"], row["shown"], row["position"], data["ban_sentence"])
        response, attempts, infra_error = generate(args.base_url, args.model, prompt_text, args.timeout, args.num_ctx)
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
            f"prompt_id: {row['prompt']['id']}\narm: {row['arm']}\nposition: {row['position']}\n"
            f"banned_shown: {row['shown']}\n\n"
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
            "arm": row["arm"],
            "position": row["position"],
            "banned_pool": row["prompt"]["banned"],
            "banned_shown": row["shown"],
            "prompt": prompt_text,
            "payload": {k: v for k, v in _payload(args.model, prompt_text, args.num_ctx).items() if k != "prompt"},
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
    parser.add_argument("--control-reps", type=int, default=0,
                        help="통제군 반복 수 override(0=케이스 파일 값). 느린 모델용")
    parser.add_argument("--num-ctx", type=int, default=0,
                        help="컨텍스트 창(0=ollama 기본값). 사고 출력이 긴 모델은 기본값에서 답이 잘린다")
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
