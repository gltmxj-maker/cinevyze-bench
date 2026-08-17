#!/usr/bin/env python3
"""Self-grading reliability benchmark against local Ollama.

두 단계로 나뉜다.
  --stage answer : 모델이 문항에 답한다(모델별로 따로 실행 — GPU에 모델을 하나씩만 올린다).
  --stage grade  : 채점 모델이 그 답들을 채점한다. 정답키는 채점 모델에게 주지 않는 팔이 기본이고,
                   정답을 주는 팔은 채점 능력의 상한을 재는 대조군이다.

축이 두 개다. ①채점 방식 3종 ②답을 쓴 주체(자기 답 / 다른 모델 답) — 후자가 자기 편애를 잰다.
답의 옳고 그름은 채점 모델의 말이 아니라 selfgrade_score의 정답키 대조로 정해진다.

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

from selfgrade_score import answer_is_correct, score_run_dir

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("selfgrade_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-selfgrade-20260803"
ARMS = ("blind", "solve_first", "with_key")


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    if len(questions) < 12:
        raise ValueError("문항이 12개 미만이면 비율의 분모가 너무 작다")
    if set(data.get("grading_arms") or {}) != set(ARMS):
        raise ValueError(f"grading_arms는 {ARMS} 셋")
    seen: set[str] = set()
    for question in questions:
        qid = question.get("id")
        if not qid or qid in seen:
            raise ValueError(f"중복/빈 question id: {qid}")
        seen.add(qid)
        if not question.get("answer_aliases"):
            raise ValueError(f"{qid} answer_aliases 없음")
        if not question.get("answer_label"):
            raise ValueError(f"{qid} answer_label 없음")
        # 정답 라벨 자체가 정답키에 안 걸리면 with_key 팔의 대조가 성립하지 않는다.
        if not answer_is_correct(question["answer_aliases"], question["answer_label"])[0]:
            raise ValueError(f"{qid} answer_label이 answer_aliases에 안 걸림")
    return data


def build_answer_prompt(question: dict[str, Any]) -> str:
    return (
        "다음 질문에 답하세요.\n"
        "- 설명 없이 답만 한 줄로 쓰세요.\n\n"
        f"질문: {question['question']}\n"
        "답:"
    )


def build_grade_prompt(question: dict[str, Any], answer_text: str, arm: str) -> str:
    header = (
        "당신은 채점자입니다. 아래 질문과 그에 대한 답을 보고 그 답이 맞았는지 판정하세요.\n\n"
        f"[질문]\n{question['question']}\n\n"
        f"[채점할 답]\n{answer_text.strip()}\n\n"
    )
    if arm == "blind":
        instruction = (
            "판정만 하세요. 마지막 줄에 정확히 '판정: 정답' 또는 '판정: 오답' 형식으로 쓰세요.\n"
        )
    elif arm == "solve_first":
        # ★알려진 교란(T2 적대검증 2026-08-03): 이 지시는 [채점할 답] 뒤에 붙는다 — 모델은 이미 답을
        #   읽은 뒤 "직접 풀라"는 말을 듣는다. 완전한 독립 재풀이가 아니므로 다음 라운드에서
        #   답을 가린 상태로 먼저 풀리는 팔을 따로 만들어 비교해야 한다(현 결과는 '프롬프트 끝에
        #   한 줄 덧붙이기'의 효과로만 읽어야 한다).
        instruction = (
            "먼저 당신이 직접 이 질문을 풀어 보고, 그 결과와 위 답을 비교해 판정하세요.\n"
            "마지막 줄에 정확히 '판정: 정답' 또는 '판정: 오답' 형식으로 쓰세요.\n"
        )
    else:
        instruction = (
            f"참고: 이 질문의 정답은 '{question['answer_label']}'입니다.\n"
            "이 정답과 비교해 판정하세요. 마지막 줄에 정확히 '판정: 정답' 또는 '판정: 오답' 형식으로 쓰세요.\n"
        )
    return header + instruction


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


def generate(base_url: str, model: str, prompt: str,
             timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None, dict[str, Any]]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0}
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


def _existing(run_dir: Path, kind: str) -> set[str]:
    keys: set[str] = set()
    for path in sorted((run_dir / "raw").glob(f"*-{kind}.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys


def _next_id(run_dir: Path) -> int:
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-*.json")):
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
        rows.append({"metric": "agreement_rate_by_arm", "arm": arm, "value": entry["agreement_rate"],
                     "agree": entry["agree"], "total": entry["scored"],
                     "false_pass_rate": entry["false_pass_rate"]})
    for label, entry in (aggregate.get("self_vs_other") or {}).items():
        rows.append({"metric": "false_pass_rate_by_authorship", "arm": label,
                     "value": entry["false_pass_rate"], "false_pass": entry["false_pass"],
                     "total": entry["wrong_answers"]})
    for model, entry in (aggregate.get("answer_accuracy") or {}).items():
        rows.append({"metric": "answer_accuracy", "arm": model, "value": entry["accuracy"],
                     "correct": entry["correct"], "total": entry["answers"]})
    return rows


def write_run_yaml(run_dir: Path, metadata: dict[str, Any], models: dict[str, Any]) -> None:
    entries = []
    for kind in ("answer", "grade"):
        for path in sorted((run_dir / "raw").glob(f"*-{kind}.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            entries.append({
                "key": row["key"],
                "stage": kind,
                "question_id": row["question_id"],
                "author_model": row["author_model"],
                "grader_model": row.get("grader_model"),
                "arm": row.get("arm"),
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
        "method": ("SGR-SELFGRADE — 정답이 하나로 정해지는 24문항에 모델 2종이 답하고, 채점 모델이 그 답을 "
                   "채점한다. 축 = 채점 방식 3종(정답 비공개 / 먼저 풀고 비교 / 정답 공개) x 답을 쓴 주체"
                   "(자기 답 / 다른 모델 답). 옳고 그름은 정답키 대조로 결정론 판정."),
        "access": "local",
        "model": models.get("grader"),
        "author_models": models.get("authors"),
        "generated_by": "run_selfgrade_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "axis": "grading_arm_x_answer_authorship",
        "arms": list(ARMS),
        "compare": _compare_rows(run_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "ollama_models": os.environ.get("OLLAMA_MODELS"),
        },
        "model_metadata": metadata,
        "runs": entries,
    }
    target_path = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target_path)


def _models_seen(run_dir: Path) -> dict[str, Any]:
    authors: set[str] = set()
    grader: str | None = None
    for path in sorted((run_dir / "raw").glob("*-answer.json")):
        authors.add(json.loads(path.read_text(encoding="utf-8"))["author_model"])
    for path in sorted((run_dir / "raw").glob("*-grade.json")):
        grader = json.loads(path.read_text(encoding="utf-8"))["grader_model"]
    return {"authors": sorted(authors), "grader": grader}


def _write_records(run_dir: Path, run_id: int, kind: str, record: dict[str, Any],
                   prompt_text: str, response: dict[str, Any] | None, response_text: str) -> dict[str, Any]:
    stem = f"{run_id:03d}"
    response_rel = f"raw/{stem}-response.txt"
    transcript_rel = f"raw/{stem}-output.txt"
    (run_dir / response_rel).write_text(response_text, encoding="utf-8")
    header = "\n".join(f"{key}: {value}" for key, value in record.items()
                       if key in ("key", "question_id", "author_model", "grader_model", "arm"))
    (run_dir / transcript_rel).write_text(
        f"run_id: {run_id}\n{header}\n\n[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
        f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8")
    record.update({
        "run_id": run_id,
        "prompt": prompt_text,
        "response_file": response_rel,
        "transcript_file": transcript_rel,
        "api_metrics": {key: response.get(key) for key in (
            "done_reason", "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration"
        )} if response else None,
    })
    (run_dir / f"raw/{stem}-{kind}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def stage_answer(args: argparse.Namespace, data: dict[str, Any]) -> int:
    questions = data["questions"]
    existing = _existing(args.run_dir, "answer")
    run_id = _next_id(args.run_dir)
    pending = [q for q in questions if f"answer/{args.model}/{q['id']}" not in existing]
    print(f"stage=answer model={args.model} planned={len(questions)} pending={len(pending)}")
    if not pending:
        return 0
    metadata = model_metadata(args.base_url, args.model, args.timeout)
    (args.run_dir / "model_metadata").mkdir(exist_ok=True)
    (args.run_dir / "model_metadata" / f"{args.model.replace(':', '_')}.json").write_text(
        json.dumps({"modified_at": metadata.get("modified_at"), "details": metadata.get("details"),
                    "parameters": metadata.get("parameters")}, ensure_ascii=False, indent=2), encoding="utf-8")

    for index, question in enumerate(pending, 1):
        run_id += 1
        prompt_text = build_answer_prompt(question)
        response, attempts, infra_error, payload = generate(
            args.base_url, args.model, prompt_text, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        record = {
            "key": f"answer/{args.model}/{question['id']}",
            "question_id": question["id"],
            "category": question["category"],
            "question": question["question"],
            "answer_aliases": question["answer_aliases"],
            "answer_label": question["answer_label"],
            "author_model": args.model,
            "payload": {key: value for key, value in payload.items() if key != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(args.run_dir, run_id, "answer", record, prompt_text, response, response_text)
        print(f"[{index}/{len(pending)}] {record['key']} elapsed={record['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)
    return 0


def stage_grade(args: argparse.Namespace, data: dict[str, Any]) -> int:
    answers = []
    for path in sorted((args.run_dir / "raw").glob("*-answer.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        response_path = args.run_dir / row["response_file"]
        row["answer_text"] = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        answers.append(row)
    if not answers:
        print("채점할 답이 없음 — --stage answer를 먼저 실행하세요.")
        return 3
    questions = {q["id"]: q for q in data["questions"]}
    arms = ARMS if args.mode == "full" else ARMS[:1]
    subset = answers if args.mode == "full" else answers[:6]

    existing = _existing(args.run_dir, "grade")
    run_id = _next_id(args.run_dir)
    plan = []
    for answer in subset:
        for arm in arms:
            key = f"grade/{args.model}/{arm}/{answer['author_model']}/{answer['question_id']}"
            if key not in existing:
                plan.append((key, answer, arm))
    print(f"stage=grade grader={args.model} mode={args.mode} planned={len(plan)}")
    if not plan:
        return 0
    metadata = model_metadata(args.base_url, args.model, args.timeout)

    for index, (key, answer, arm) in enumerate(plan, 1):
        run_id += 1
        question = questions[answer["question_id"]]
        prompt_text = build_grade_prompt(question, answer["answer_text"], arm)
        response, attempts, infra_error, payload = generate(
            args.base_url, args.model, prompt_text, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        correct, matched = answer_is_correct(question["answer_aliases"], answer["answer_text"])
        record = {
            "key": key,
            "question_id": question["id"],
            "category": question["category"],
            "arm": arm,
            "author_model": answer["author_model"],
            "grader_model": args.model,
            "graded_answer": answer["answer_text"].strip(),
            # 진실값은 채점 모델의 말이 아니라 정답키 대조 결과다. 여기서 한 번, 채점기에서 또 한 번 계산된다.
            "answer_correct": correct and not answer.get("infra_error"),
            "matched_alias": matched,
            "payload": {key_: value for key_, value in payload.items() if key_ != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(args.run_dir, run_id, "grade", record, prompt_text, response, response_text)
        print(f"[{index}/{len(plan)}] {key} truth={'O' if record['answer_correct'] else 'X'} "
              f"elapsed={record['elapsed_s']}s" + (f" ERROR={infra_error}" if infra_error else ""), flush=True)
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    if args.dry_run:
        question = data["questions"][0]
        print(json.dumps({
            "questions": len(data["questions"]),
            "answer_prompt": build_answer_prompt(question),
            "grade_prompts": {arm: build_grade_prompt(question, "8월 15일", arm) for arm in ARMS},
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    if args.mode == "pilot":
        (args.run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot"}, ensure_ascii=False), encoding="utf-8")
    elif args.stage == "grade":
        decision_path = args.run_dir / "pilot_decision.json"
        if not decision_path.exists():
            print("파일럿 판정 없음 — 같은 --run-dir에서 --mode pilot을 먼저 완료하세요.")
            return 3
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("proceed"):
            print(f"파일럿 중단조건 발동: {decision.get('reasons')}")
            return 3

    status = stage_answer(args, data) if args.stage == "answer" else stage_grade(args, data)
    if status != 0:
        return status

    grades, aggregate = score_run_dir(args.run_dir)
    metadata = {}
    metadata_dir = args.run_dir / "model_metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.glob("*.json")):
            metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    write_run_yaml(args.run_dir, metadata, _models_seen(args.run_dir))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if args.mode == "pilot" and (args.run_dir / "pilot_decision.json").exists():
        decision = json.loads((args.run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
        if not decision["proceed"]:
            print(f"파일럿 중단조건: {decision['reasons']}")
            return 3
    return 0 if aggregate["infra_errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b", help="이 단계를 수행할 모델")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--stage", choices=("answer", "grade"), default="answer")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
