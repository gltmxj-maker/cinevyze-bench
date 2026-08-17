#!/usr/bin/env python3
"""Multi-turn instruction-decay benchmark against local Ollama.

대화 첫 턴에 형식 규칙 3개를 주고, 규칙과 무관한 질문을 21턴 이어가며 매 턴 준수 여부를 잰다.
축이 두 개다. ①규칙 부여 후 경과 턴 ②규칙 유지 방식 3팔(첫 턴에만 / system 고정 / 매 턴 재삽입).

★num_ctx를 명시적으로 지정한다. 누적 대화가 컨텍스트 창을 넘으면 모델이 규칙을 '잊은' 게 아니라
프롬프트에서 규칙이 '잘려 나간' 것이다. 매 턴 prompt_eval_count를 기록해 창을 넘지 않았음을
숫자로 증명하고, 넘긴 턴이 하나라도 있으면 채점기가 파일럿을 세운다.

This script is the write-origin for run.yaml: 매 시행의 메시지 전문·원응답·API 메타데이터를
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

from multiturn_score import RULES, score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("multiturn_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-multiturn-20260806"
ARMS = ("user_once", "system", "remind")
PILOT_TURNS = 6


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    conversations = data.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError("대화가 2본 미만이면 스크립트 하나의 특성을 결과로 착각하게 된다")
    if set(data.get("arms") or {}) != set(ARMS):
        raise ValueError(f"arms는 {ARMS} 셋")
    rule_ids = {item["id"] for item in data["rules"]["items"]}
    if rule_ids != set(RULES):
        raise ValueError(f"규칙 id가 채점기와 불일치: {rule_ids} vs {set(RULES)}")
    seen: set[str] = set()
    lengths: set[int] = set()
    for conversation in conversations:
        cid = conversation.get("id")
        if not cid or cid in seen:
            raise ValueError(f"중복/빈 conversation id: {cid}")
        seen.add(cid)
        questions = conversation.get("questions") or []
        if len(questions) < PILOT_TURNS:
            raise ValueError(f"{cid} 질문이 {PILOT_TURNS}개 미만")
        lengths.add(len(questions))
    if len(lengths) != 1:
        raise ValueError(f"대화별 턴 수가 다르면 구간 비교가 깨진다: {sorted(lengths)}")
    return data


def build_user_message(data: dict[str, Any], arm: str, turn: int, question: str) -> str:
    rules = data["rules"]
    if turn == 1 and arm in ("user_once", "remind"):
        return f"{rules['block']}\n\n{question}"
    if arm == "remind":
        return f"{question}\n{rules['remind']}"
    return question


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


def chat(base_url: str, model: str, messages: list[dict[str, str]], num_ctx: int,
         timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None, dict[str, Any]]:
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": 0,
               "options": {"num_ctx": num_ctx}}
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/chat", payload, timeout)
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


def _existing(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "raw").glob("*-turn.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["key"]] = row
    return rows


def _next_id(run_dir: Path) -> int:
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-turn.json")):
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
        for rule in RULES + ("all_rules",):
            rows.append({"metric": "keep_rate_by_arm", "arm": arm, "rule": rule,
                         "value": entry[rule]["rate"], "kept": entry[rule]["kept"], "total": entry[rule]["n"]})
    for arm, buckets in (aggregate.get("by_arm_turn_bucket") or {}).items():
        for label, entry in buckets.items():
            rows.append({"metric": "all_rules_by_turn_bucket", "arm": arm, "bucket": label,
                         "value": entry["all_rules"]["rate"], "kept": entry["all_rules"]["kept"],
                         "total": entry["all_rules"]["n"]})
    for model, entry in (aggregate.get("by_model") or {}).items():
        rows.append({"metric": "all_rules_by_model", "arm": model, "value": entry["all_rules"]["rate"],
                     "kept": entry["all_rules"]["kept"], "total": entry["all_rules"]["n"]})
    for arm, entry in (aggregate.get("survival_by_arm") or {}).items():
        for rule in RULES:
            rows.append({"metric": "clean_conversations_by_arm", "arm": arm, "rule": rule,
                         "value": entry[rule]["clean"], "total": entry[rule]["conversations"],
                         "median_first_violation": entry[rule]["median_first_violation"]})
    context = aggregate.get("context") or {}
    rows.append({"metric": "context_headroom", "arm": "all", "value": context.get("max_prompt_tokens"),
                 "num_ctx": context.get("num_ctx"), "overflow_turns": context.get("overflow_turns")})
    return rows


def write_run_yaml(run_dir: Path, metadata: dict[str, Any], data: dict[str, Any], num_ctx: int) -> None:
    entries = []
    models: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-turn.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        models.add(row["model"])
        entries.append({
            "key": row["key"],
            "arm": row["arm"],
            "model": row["model"],
            "conv_id": row["conv_id"],
            "turn": row["turn"],
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
        "method": ("SGR-MULTITURN — 대화 첫 턴에 형식 규칙 3개(접두 '요약:' · 영문 알파벳 금지 · 말미 서명)를 "
                   "주고 규칙과 무관한 질문을 21턴 이어가며 매 턴 준수 여부를 결정론 판정한다. "
                   "축 = 경과 턴 x 규칙 유지 방식 3팔(첫 턴에만 / system 고정 / 매 턴 재삽입). "
                   "num_ctx를 명시하고 매 턴 prompt_eval_count를 기록해 잘림과 망각을 분리한다."),
        "access": "local",
        "model": sorted(models)[0] if models else None,
        "models": sorted(models),
        "generated_by": "run_multiturn_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "num_ctx": num_ctx,
        "axis": "turn_distance_x_rule_persistence_arm",
        "arms": list(ARMS),
        "rules": [item["id"] for item in data["rules"]["items"]],
        "conversations": [conversation["id"] for conversation in data["conversations"]],
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


def _write_records(run_dir: Path, run_id: int, record: dict[str, Any], messages: list[dict[str, str]],
                   response: dict[str, Any] | None, response_text: str) -> dict[str, Any]:
    stem = f"{run_id:03d}"
    response_rel = f"raw/{stem}-response.txt"
    transcript_rel = f"raw/{stem}-output.txt"
    (run_dir / response_rel).write_text(response_text, encoding="utf-8")
    header = "\n".join(f"{key}: {value}" for key, value in record.items()
                       if key in ("key", "arm", "model", "conv_id", "turn", "num_ctx"))
    rendered = "\n\n".join(f"[{message['role'].upper()}]\n{message['content']}" for message in messages)
    (run_dir / transcript_rel).write_text(
        f"run_id: {run_id}\n{header}\n\n[MESSAGES]\n{rendered}\n\n[RAW RESPONSE]\n{response_text}\n"
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
    (run_dir / f"raw/{stem}-turn.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_conversation(args: argparse.Namespace, data: dict[str, Any], arm: str,
                     conversation: dict[str, Any], existing: dict[str, dict[str, Any]],
                     run_dir: Path, max_turns: int, run_id: int) -> tuple[int, int]:
    """한 대화를 끝까지 굴린다. 이미 기록된 턴은 저장된 응답으로 history만 복원하고 건너뛴다."""
    messages: list[dict[str, str]] = []
    if arm == "system":
        messages.append({"role": "system", "content": data["rules"]["block"]})
    performed = 0
    for turn, question in enumerate(conversation["questions"][:max_turns], 1):
        user_content = build_user_message(data, arm, turn, question)
        key = f"{arm}/{args.model}/{conversation['id']}/t{turn:02d}"
        messages.append({"role": "user", "content": user_content})
        if key in existing:
            stored = existing[key]
            stored_path = run_dir / stored["response_file"]
            messages.append({"role": "assistant",
                             "content": stored_path.read_text(encoding="utf-8") if stored_path.exists() else ""})
            continue

        run_id += 1
        response, attempts, infra_error, payload = chat(
            args.base_url, args.model, messages, args.num_ctx, args.timeout)
        message_value = (response or {}).get("message") if response is not None else None
        response_value = message_value.get("content") if isinstance(message_value, dict) else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: message.content가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        record = {
            "key": key,
            "arm": arm,
            "model": args.model,
            "conv_id": conversation["id"],
            "turn": turn,
            "question": question,
            "user_content": user_content,
            "num_ctx": args.num_ctx,
            "payload": {k: v for k, v in payload.items() if k != "messages"},
            "message_count": len(messages),
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(run_dir, run_id, record, messages, response, response_text)
        performed += 1
        prompt_tokens = ((record.get("api_metrics") or {}).get("prompt_eval_count")
                         if record.get("api_metrics") else None)
        print(f"  [{key}] {record['elapsed_s']}s ptok={prompt_tokens}"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)
        if infra_error:
            # 응답이 없는 채로 history를 이어 붙이면 이후 턴 전부가 오염된다 — 이 대화는 여기서 끊는다.
            print(f"  ! {conversation['id']} 중단 — infra 오류", flush=True)
            return run_id, performed
        messages.append({"role": "assistant", "content": response_text})
    return run_id, performed


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    if args.dry_run:
        conversation = data["conversations"][0]
        print(json.dumps({
            "conversations": [c["id"] for c in data["conversations"]],
            "turns_per_conversation": len(conversation["questions"]),
            "planned_calls_per_model": len(ARMS) * len(data["conversations"]) * len(conversation["questions"]),
            "sample": {arm: [build_user_message(data, arm, turn, conversation["questions"][turn - 1])
                             for turn in (1, 2)] for arm in ARMS},
        }, ensure_ascii=False, indent=2))
        return 0

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    max_turns = PILOT_TURNS if args.mode == "pilot" else len(data["conversations"][0]["questions"])
    conversations = data["conversations"][:1] if args.mode == "pilot" else data["conversations"]

    if args.mode == "pilot":
        (run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot", "turns": max_turns}, ensure_ascii=False), encoding="utf-8")
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
    arms = getattr(args, "arms", ARMS)
    planned = len(arms) * len(conversations) * max_turns
    done = sum(1 for key in existing if key.split("/")[1] == args.model)
    print(f"model={args.model} mode={args.mode} num_ctx={args.num_ctx} "
          f"planned={planned} already={done}", flush=True)

    performed_total = 0
    for arm in arms:
        for conversation in conversations:
            print(f"[{arm}] {conversation['id']}", flush=True)
            run_id, performed = run_conversation(
                args, data, arm, conversation, existing, run_dir, max_turns, run_id)
            performed_total += performed

    rows, aggregate = score_run_dir(run_dir)
    metadata = {}
    metadata_dir = run_dir / "model_metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.glob("*.json")):
            metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    write_run_yaml(run_dir, metadata, data, args.num_ctx)
    print(json.dumps({key: aggregate[key] for key in ("overall", "by_arm", "context", "infra_errors")},
                     ensure_ascii=False, indent=2))
    print(f"performed_this_run={performed_total} rows_total={len(rows)}")
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
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="★명시 필수 — 누적 대화가 창을 넘으면 망각이 아니라 잘림을 재게 된다")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="돌릴 팔(쉼표 구분). 위치 교차 대조군처럼 한 팔만 필요할 때 좁힌다")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        parser.error(f"알 수 없는 팔: {sorted(unknown)} (가능: {ARMS})")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
