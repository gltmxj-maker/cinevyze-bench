#!/usr/bin/env python3
"""Positional-recall ("needle in a haystack") benchmark against local Ollama.

한 문서에 유일한 사실 한 줄을 심고 **문서 길이 × 심은 위치**만 바꾼다. 질문·지시문·채우기 텍스트는
바이트 단위로 동일하게 유지하므로, 정답률 차이는 "어디에 있었나"에서만 나온다.

★num_ctx를 명시적으로 지정한다. Ollama 기본 컨텍스트가 문서보다 짧으면 앞부분이 조용히 잘려
"위치 효과"가 아니라 "잘림"을 측정하게 된다. 매 시행의 prompt_eval_count를 기록해 실제로 다
들어갔는지 확인한다.

This script is the write-origin for run.yaml: 매 시행의 프롬프트 전문·원응답·API 메타데이터를
raw/에 저장하고 그 기록에서만 run.yaml을 만든다. 사람이 수치를 적어 넣을 자리는 없다.
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

from needle_score import score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("needle_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-gemma3-4b-needle-20260803b"
LENGTH_ORDER = ("L1-2k", "L2-8k", "L3-24k", "L4-48k")


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lengths = data.get("lengths") or {}
    positions = data.get("positions") or []
    needles = data.get("needles") or []
    filler = data.get("filler_clauses") or []
    if set(lengths) != set(LENGTH_ORDER):
        raise ValueError(f"lengths는 {LENGTH_ORDER} 넷이어야 함")
    ordered = [lengths[key] for key in LENGTH_ORDER]
    if ordered != sorted(ordered) or len(set(ordered)) != len(LENGTH_ORDER):
        raise ValueError("lengths 값이 오름차순 서로 다른 4개가 아님")
    if len(positions) != 5 or positions != sorted(positions):
        raise ValueError("positions는 오름차순 5개")
    if positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError("positions는 0.0에서 1.0까지 덮어야 함")
    if len(needles) != 4:
        raise ValueError(f"needles {len(needles)}건 != 4")
    if len(filler) < 20:
        raise ValueError("filler_clauses가 20개 미만이면 문서가 같은 문장 반복이 된다")
    seen: set[str] = set()
    for needle in needles:
        nid = needle.get("id")
        if not nid or nid in seen:
            raise ValueError(f"중복/빈 needle id: {nid}")
        seen.add(nid)
        if not needle.get("answer_aliases"):
            raise ValueError(f"{nid} answer_aliases 없음")
        # 심은 문장 자체가 정답을 담고 있지 않으면 애초에 회상을 잴 수 없다.
        if not any(alias in needle["clause_body"] for alias in needle["answer_aliases"]):
            raise ValueError(f"{nid} clause_body에 정답 별칭이 없음")
    for needle in needles:
        for clause in filler:
            for alias in needle["answer_aliases"]:
                if alias in clause["body"] or alias in clause["title"]:
                    raise ValueError(f"채우기 문장이 정답 '{alias}'를 이미 포함 — 문항 오염")
    distractor_positions = data.get("distractor_positions") or []
    if len(distractor_positions) != 3:
        raise ValueError("distractor_positions는 3개")
    for needle in needles:
        distractors = needle.get("distractors") or []
        if len(distractors) != 3:
            raise ValueError(f"{needle['id']} distractors 3개 아님")
        for distractor in distractors:
            # 방해 조항이 정답과 같은 값을 담고 있으면 오답을 정답으로 세게 된다.
            if any(alias in distractor["clause_body"] for alias in needle["answer_aliases"]):
                raise ValueError(f"{needle['id']} 방해 조항이 정답 값을 포함")
            if distractor["value"] not in distractor["clause_body"]:
                raise ValueError(f"{needle['id']} 방해 조항 본문에 value 없음")
    return data


def _clause(index: int, title: str, body: str) -> str:
    return f"제{index}조 ({title}) {body}"


def build_document(data: dict[str, Any], length_key: str, needle: dict[str, Any],
                   position: float, with_distractors: bool = False) -> dict[str, Any]:
    """Filler clauses are cycled with unique article numbers until the target size is reached."""
    target_chars = data["lengths"][length_key]
    filler = data["filler_clauses"]
    clauses: list[str] = []
    index = 1
    size = 0
    while size < target_chars:
        source = filler[(index - 1) % len(filler)]
        text = _clause(index, source["title"], source["body"])
        clauses.append(text)
        size += len(text) + 1
        index += 1
    # 방해 조항을 먼저 흩뿌린 뒤 정답 조항을 넣는다. 순서를 반대로 하면 정답 위치가 밀린다.
    distractor_values: list[str] = []
    if with_distractors:
        placements = sorted(zip(data["distractor_positions"], needle["distractors"]), reverse=True)
        for ratio, distractor in placements:
            slot = max(0, min(round(ratio * len(clauses)), len(clauses)))
            clauses.insert(slot, _clause(0, distractor["clause_title"], distractor["clause_body"]))
            distractor_values.append(distractor["value"])
    # 심은 문장은 채우기 조항과 구조가 같아야 한다(형식이 다르면 눈에 띄어 위치 효과가 흐려진다).
    slot = round(position * len(clauses))
    slot = max(0, min(slot, len(clauses)))
    clauses.insert(slot, _clause(0, needle["clause_title"], needle["clause_body"]))
    numbered = []
    for order, text in enumerate(clauses, 1):
        head, _, rest = text.partition(") ")
        title = head.split("(", 1)[1] if "(" in head else ""
        numbered.append(_clause(order, title, rest))
    document = "\n".join(numbered)
    return {
        "document": document,
        "doc_chars": len(document),
        "clause_count": len(numbered),
        "needle_clause_index": slot + 1,
        "distractor_values": distractor_values,
    }


def build_prompt(data: dict[str, Any], document: str, needle: dict[str, Any]) -> str:
    return (
        f"다음은 「{data['doc_title']}」 전문입니다. 문서를 읽고 아래 질문에 답하세요.\n\n"
        "규칙:\n"
        "- 문서에 적힌 내용만 근거로 답하세요.\n"
        "- 문서에 없으면 '문서에 없음'이라고만 답하세요.\n"
        "- 설명 없이 답만 한 줄로 쓰세요.\n\n"
        "[문서 시작]\n"
        f"{document}\n"
        "[문서 끝]\n\n"
        f"질문: {needle['question']}\n"
        "답:"
    )


def build_control_prompt(data: dict[str, Any], needle: dict[str, Any]) -> str:
    """Ceiling check — the same question with the planted sentence as the whole document."""
    return build_prompt(data, _clause(1, needle["clause_title"], needle["clause_body"]), needle)


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


def generate(base_url: str, model: str, prompt: str, num_ctx: int | None,
             timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
    }
    # num_ctx=None은 "아무 설정도 안 한 기본값" 팔이다 — 대부분의 사용자가 실제로 쓰는 조건.
    if num_ctx is not None:
        payload["options"] = {"num_ctx": num_ctx}
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


def planned_runs(data: dict[str, Any], mode: str, repeats: int) -> list[dict[str, Any]]:
    needles = data["needles"]
    positions = data["positions"]
    runs: list[dict[str, Any]] = []
    if mode == "pilot":
        # 배관 점검용 최소 계획: 대조군 전 문항 1회 + 가장 긴 문서의 양 끝 위치(방해 조항 두 축 모두).
        for needle in needles:
            runs.append({"key": f"control/{needle['id']}/1", "arm": "control",
                         "length": "control", "position": "control", "needle": needle,
                         "distractors": False, "ctx_mode": "explicit", "repeat": 1})
        for needle in needles[:2]:
            for distractors in (False, True):
                tag = "d1" if distractors else "d0"
                for position in (positions[0], positions[-1]):
                    runs.append({"key": f"{LENGTH_ORDER[-1]}/{tag}/{position}/{needle['id']}/1",
                                 "arm": "haystack", "length": LENGTH_ORDER[-1], "position": position,
                                 "needle": needle, "distractors": distractors,
                                 "ctx_mode": "explicit", "repeat": 1})
        for position in (positions[0], positions[-1]):
            runs.append({"key": f"{LENGTH_ORDER[-1]}/ctxdefault/{position}/{needles[0]['id']}/1",
                         "arm": "haystack", "length": LENGTH_ORDER[-1], "position": position,
                         "needle": needles[0], "distractors": False,
                         "ctx_mode": "default", "repeat": 1})
        return runs
    for length_key in LENGTH_ORDER:
        for distractors in (False, True):
            tag = "d1" if distractors else "d0"
            for position in positions:
                for needle in needles:
                    for repeat in range(1, repeats + 1):
                        runs.append({"key": f"{length_key}/{tag}/{position}/{needle['id']}/{repeat}",
                                     "arm": "haystack", "length": length_key, "position": position,
                                     "needle": needle, "distractors": distractors,
                                     "ctx_mode": "explicit", "repeat": repeat})
    # 컨텍스트 창을 아무것도 지정하지 않은 팔 — 문서가 창보다 길면 조용히 잘린다.
    for length_key in LENGTH_ORDER:
        for position in positions:
            for needle in needles:
                for repeat in range(1, repeats + 1):
                    runs.append({"key": f"{length_key}/ctxdefault/{position}/{needle['id']}/{repeat}",
                                 "arm": "haystack", "length": length_key, "position": position,
                                 "needle": needle, "distractors": False,
                                 "ctx_mode": "default", "repeat": repeat})
    for needle in needles:
        for repeat in range(1, repeats + 1):
            runs.append({"key": f"control/{needle['id']}/{repeat}", "arm": "control",
                         "length": "control", "position": "control", "needle": needle,
                         "distractors": False, "ctx_mode": "explicit", "repeat": repeat})
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
    """Read back from the scored aggregate — never hand-written.

    The gate that reads `compare` exists to stop an article claiming a comparison the raw data
    does not contain, so these rows are derived from aggregate.json produced by needle_score.
    """
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for position, entry in (aggregate.get("by_position") or {}).items():
        rows.append({"metric": "hit_rate_by_position", "arm": f"위치 {position}",
                     "value": entry["hit_rate"], "hit": entry["hit"], "total": entry["scored"]})
    for length_key, entry in (aggregate.get("by_length") or {}).items():
        rows.append({"metric": "hit_rate_by_length", "arm": length_key,
                     "value": entry["hit_rate"], "hit": entry["hit"], "total": entry["scored"]})
    for label, entry in (aggregate.get("by_distractor") or {}).items():
        rows.append({"metric": "hit_rate_by_distractor", "arm": f"방해 조항 {label}",
                     "value": entry["hit_rate"], "hit": entry["hit"], "total": entry["scored"],
                     "distractor_pull": entry.get("distractor_rate")})
    control = aggregate.get("control") or {}
    if control.get("scored"):
        rows.append({"metric": "hit_rate_control", "arm": "대조군(문장 1줄)",
                     "value": control["hit_rate"], "hit": control["hit"], "total": control["scored"]})
    return rows


def write_run_yaml(run_dir: Path, model: str, metadata: dict[str, Any], data: dict[str, Any],
                   num_ctx: int) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "arm": row["arm"],
            "length": row["length"],
            "position": row["position"],
            "distractors": row.get("distractors"),
            "needle_id": row["needle_id"],
            "doc_chars": row.get("doc_chars"),
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
        "method": ("NDL-RECALL — 자체 작성 한국어 사내 규정 문서(2000/8000/24000/48000자)에 유일한 사실 1줄을 "
                   "0/25/50/75/100% 위치에 심고 회상 정답률 측정. 축 = 문서 길이 x 위치 x 방해 조항 유무(같은 형식·"
                   "다른 값 3개). 문항 4종. 대조군 = 그 문장만 준 경우."),
        "access": "local",
        "model": model,
        "generated_by": "run_needle_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "axis": "needle_position_x_document_length_x_distractors",
        "num_ctx": num_ctx,
        "lengths": data["lengths"],
        "positions": data["positions"],
        "distractor_positions": data["distractor_positions"],
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
    plan = planned_runs(data, args.mode, args.repeats)
    if not plan:
        raise ValueError("실행 계획이 비어 있음")
    if len({row["key"] for row in plan}) != len(plan):
        raise ValueError("실행 계획에 중복 키 존재")

    if args.dry_run:
        first = plan[0]
        if first["arm"] == "control":
            prompt = build_control_prompt(data, first["needle"])
        else:
            built = build_document(data, first["length"], first["needle"], first["position"],
                                   first["distractors"])
            prompt = build_prompt(data, built["document"], first["needle"])
        sizes = {}
        for length_key in LENGTH_ORDER:
            sample = build_document(data, length_key, data["needles"][0], 0.5, True)
            sizes[length_key] = {"doc_chars": sample["doc_chars"], "clauses": sample["clause_count"],
                                 "distractors": sample["distractor_values"]}
        print(json.dumps({"mode": args.mode, "runs": len(plan), "doc_sizes": sizes,
                          "first_key": first["key"], "first_prompt_chars": len(prompt),
                          "first_prompt_head": prompt[:400]}, ensure_ascii=False, indent=2))
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
    if args.mode == "pilot":
        (args.run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot", "planned": len(plan)}, ensure_ascii=False), encoding="utf-8")

    existing_keys, next_id = _existing(args.run_dir)
    pending = [row for row in plan if row["key"] not in existing_keys]
    pending_total = len(pending)
    if args.max_new > 0:
        pending = pending[:args.max_new]
    print(f"mode={args.mode} planned={len(plan)} existing={len(plan) - pending_total} "
          f"pending_total={pending_total} running_now={len(pending)} num_ctx={args.num_ctx}")

    if pending:
        metadata = model_metadata(args.base_url, args.model, args.timeout)
    elif (args.run_dir / "run.yaml").exists():
        previous = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
        metadata = previous.get("model_metadata") or {}
    else:
        metadata = {}

    for position_index, row in enumerate(pending, 1):
        next_id += 1
        needle = row["needle"]
        if row["arm"] == "control":
            prompt_text = build_control_prompt(data, needle)
            built = {"doc_chars": len(prompt_text), "clause_count": 1, "needle_clause_index": 1,
                     "distractor_values": []}
        else:
            built = build_document(data, row["length"], needle, row["position"], row["distractors"])
            prompt_text = build_prompt(data, built["document"], needle)

        effective_ctx = args.num_ctx if row["ctx_mode"] == "explicit" else None
        response, attempts, infra_error, payload = generate(
            args.base_url, args.model, prompt_text, effective_ctx, args.timeout)
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
            f"arm: {row['arm']}\nlength: {row['length']}\nposition: {row['position']}\n"
            f"distractors: {row['distractors']}\nneedle_id: {needle['id']}\n"
            f"ctx_mode: {row['ctx_mode']}\nnum_ctx: {effective_ctx}\n\n"
            f"[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
            f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "arm": row["arm"],
            "length": row["length"],
            "position": row["position"],
            "distractors": row["distractors"],
            "distractor_values": built["distractor_values"],
            "ctx_mode": row["ctx_mode"],
            "repeat": row["repeat"],
            "needle_id": needle["id"],
            "needle_kind": needle["kind"],
            "question": needle["question"],
            "answer_aliases": needle["answer_aliases"],
            "answer_label": needle["answer_label"],
            "doc_chars": built["doc_chars"],
            "clause_count": built["clause_count"],
            "needle_clause_index": built["needle_clause_index"],
            "num_ctx": effective_ctx,
            "prompt": prompt_text,
            "payload": {key: value for key, value in payload.items() if key != "prompt"},
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
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_yaml(args.run_dir, args.model, metadata, data, args.num_ctx)
        tokens = (invocation["api_metrics"] or {}).get("prompt_eval_count")
        print(f"[{position_index}/{len(pending)}] {row['key']} elapsed={invocation['elapsed_s']}s "
              f"prompt_tokens={tokens}" + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    rows, aggregate = score_run_dir(args.run_dir)
    write_run_yaml(args.run_dir, args.model, metadata, data, args.num_ctx)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if args.mode == "pilot":
        decision = json.loads((args.run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
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
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats는 1 이상")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
