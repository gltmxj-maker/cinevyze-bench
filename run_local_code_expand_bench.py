#!/usr/bin/env python3
"""27번 확장: 두 Ollama 모델에 같은 세 코딩 과제를 순차 실행."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import time
import urllib.request
from pathlib import Path

import yaml

from code_expand_score import load_plan, score_response
from live_mark import mark


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "code_expand_cases.json"
DEFAULT_RUN = ROOT / "test_runs" / f"local-code-expand-{dt.date.today():%Y%m%d}"
WRITER = Path(__file__).name


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api_json(base_url: str, endpoint: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + endpoint,
                                 data=json.dumps(payload, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Ollama API 응답 객체 아님")
    return value


def generate(base_url: str, model: str, prompt: str, timeout: int) -> tuple[dict, float]:
    start = time.monotonic()
    data = api_json(base_url, "/api/generate",
                    {"model": model, "prompt": prompt, "stream": False,
                     "keep_alive": "5m", "options": {"temperature": 0, "seed": 27}}, timeout)
    elapsed = round(time.monotonic() - start, 3)
    if data.get("model") != model or not isinstance(data.get("response"), str) or not data["response"].strip():
        raise ValueError(f"Ollama 모델 불일치 또는 빈 응답: {data.get('model')!r}")
    return data, elapsed


def write_run_yaml(run: Path, plan: dict, cases_path: Path, cases_sha: str, records: list[dict],
                   *, condition_changes: list | None = None) -> None:
    value = {"tool": "ollama", "date": dt.date.today().isoformat(),
             "method": "TXT-02 + ROTATE-01 + BUGFIX-01", "access": "local",
             "model": ", ".join(plan["models"]), "generated_by": WRITER,
             "harness_version": "1.0", "tos_confirmed": True,
             "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
             "cases_file": Path(cases_path).name, "cases_sha256": cases_sha,
             "request_parallelism": 1, "keep_alive": "5m",
             "generation_options": {"temperature": 0, "seed": 27},
             "score_rules": "code_expand_score.py:score_response; fixed cases; Python subprocess",
             "condition_changes": condition_changes or [],
             "environment": {"python": platform.python_version(),
                             "ollama_models": os.environ.get("OLLAMA_MODELS")},
             "runs": records, "compare": [], "disclosure": True}
    tmp = run / "run.yaml.tmp"
    tmp.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, run / "run.yaml")


def record_score_change(run: Path, *, old_sha: str, new_sha: str, reason: str,
                        before: dict, after: dict) -> None:
    """측정 후 채점기 교정 이력을 하네스가 run.yaml에 원자적으로 기록한다."""
    path = Path(run) / "run.yaml"
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    if meta.get("generated_by") != WRITER or not meta.get("runs"):
        raise ValueError("하네스 런 정의 아님")
    if meta.get("condition_changes"):
        raise ValueError("채점 변경이 이미 기록됨")
    if not reason.strip() or len(old_sha) != 64 or len(new_sha) != 64:
        raise ValueError("채점 변경 근거·SHA256 누락")
    meta["condition_changes"] = [{
        "kind": "scorer_correction", "reason": reason,
        "old_scorer_sha256": old_sha, "new_scorer_sha256": new_sha,
        "before_by_model": before, "after_by_model": after,
        "rechecked_runs": len(meta["runs"]),
    }]
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def _verify_record(run: Path, item: dict, task: dict, stem: str) -> str:
    out = run / f"{stem}-output.txt"
    log = run / f"{stem}-invocation.log"
    raw = out.read_bytes()
    prompt = task["prompt"]
    expected_input, expected_output = sha(prompt.encode()), sha(raw)
    if (item.get("input") != prompt or item.get("input_sha256") != expected_input
            or item.get("output_file") != out.name or item.get("output_sha256") != expected_output
            or item.get("log_file") != log.name):
        raise ValueError(f"SHA256·고정 입력·파일 연결 불일치: {stem}")
    transcript = log.read_text(encoding="utf-8")
    if "--- RESPONSE ---\n" not in transcript:
        raise ValueError(f"호출 로그 구분자 불일치: {stem}")
    header, response = transcript.split("--- RESPONSE ---\n", 1)
    if response != raw.decode("utf-8") or "--- PROMPT ---\n" + prompt + "\n" not in header:
        raise ValueError(f"호출 로그 응답·프롬프트 불일치: {stem}")
    fields = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
    for key in ("model", "task", "repetition", "input_sha256", "output_sha256",
                "elapsed_s", "eval_count", "eval_duration_ns"):
        expected = item.get(key)
        if str(expected) != fields.get(key):
            raise ValueError(f"호출 로그 지표 불일치: {stem}/{key}")
    return response


def _summary(rows: list[dict], plan: dict) -> dict:
    by_model = {}
    by_task = {}
    for model in plan["models"]:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        by_model[model] = {
            "responses": len(selected), "contract_pass": sum(row["contract_pass"] for row in selected),
            "cases_passed": sum(row["cases_passed"] for row in selected),
            "cases_total": sum(row["cases_total"] for row in selected),
            "median_elapsed_s": round(statistics.median(row["elapsed_s"] for row in selected), 3),
        }
        by_task[model] = {task["id"]: {
            "responses": sum(row["task"] == task["id"] for row in selected),
            "contract_pass": sum(row["contract_pass"] for row in selected if row["task"] == task["id"]),
            "cases_passed": sum(row["cases_passed"] for row in selected if row["task"] == task["id"]),
            "cases_total": sum(row["cases_total"] for row in selected if row["task"] == task["id"]),
        } for task in plan["tasks"]}
    return {"sample_n": len(rows), "infra_errors": 0, "by_model": by_model,
            "by_task": by_task, "condition_changes": []}


def rescore_run(run: Path, *, cases_path: Path = DEFAULT_CASES,
                require_complete: bool = True, emit_marks: bool = False) -> dict:
    run = Path(run)
    plan = load_plan(cases_path)
    cases_sha = sha(Path(cases_path).read_bytes())
    meta = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    records = meta.get("runs") or []
    if (meta.get("generated_by") != WRITER or meta.get("model") != ", ".join(plan["models"])
            or meta.get("cases_sha256") != cases_sha
            or (meta.get("cases_file") and meta.get("cases_file") != Path(cases_path).name)):
        raise ValueError("모델·기록자·고정 사례 SHA256 불일치")
    expected = [(model, task["id"], repetition) for model in plan["models"]
                for task in plan["tasks"] for repetition in range(1, plan["repetitions"] + 1)]
    if require_complete and len(records) != len(expected):
        raise ValueError(f"회차 수 불일치: {len(records)}/{len(expected)}")
    if len(records) > len(expected):
        raise ValueError("계획보다 많은 회차")
    task_by_id = {task["id"]: task for task in plan["tasks"]}
    rows = []
    first_break = False
    for index, item in enumerate(records):
        model, task_id, repetition = expected[index]
        if (item.get("model"), item.get("task"), item.get("repetition")) != (model, task_id, repetition):
            raise ValueError(f"계획·모델·과제·반복 불일치: {index + 1}")
        stem = f"{index + 1:02d}"
        response = _verify_record(run, item, task_by_id[task_id], stem)
        scored = score_response(task_by_id[task_id], response)
        if emit_marks and not first_break and scored["first_failure"]:
            first_break = True
            fail = scored["first_failure"]
            mark("break", f"첫 사례 실패 — {model} {task_id} {repetition}/3 {fail['case_id']} "
                          f"기대 {json.dumps(fail['expected'], ensure_ascii=False)} / "
                          f"실제 {json.dumps(fail['actual'], ensure_ascii=False)} "
                          f"오류 {fail['error'] or '-'} 입력변형 {fail['mutated']}")
        rows.append({"model": model, "task": task_id, "repetition": repetition,
                     "output_file": item["output_file"], "log_file": item["log_file"],
                     "input_sha256": item["input_sha256"],
                     "output_sha256": item["output_sha256"],
                     "elapsed_s": item["elapsed_s"],
                     "eval_count": item.get("eval_count"),
                     "eval_duration_ns": item.get("eval_duration_ns"), **scored})
    summary = _summary(rows, plan)
    summary["condition_changes"] = meta.get("condition_changes") or []
    write_json(run / "results.json", rows)
    write_json(run / "aggregate.json", summary)
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ("model", "task", "repetition", "output_file", "elapsed_s",
                   "contract_pass", "cases_passed", "cases_total")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    if emit_marks:
        labels = [f"{model} {data['contract_pass']}/{data['responses']}회 "
                  f"사례 {data['cases_passed']}/{data['cases_total']}"
                  for model, data in summary["by_model"].items()]
        mark("agg", "코드 확장 두 모델 집계 — " + " · ".join(labels) + " · infra 0")
    return summary


def run_live(run: Path, base_url: str, timeout: int, cases_path: Path) -> dict:
    plan = load_plan(cases_path)
    if run.exists() and any(path.name != "live" for path in run.iterdir()):
        raise ValueError(f"런 폴더가 비어 있지 않습니다: {run}")
    run.mkdir(parents=True, exist_ok=True)
    cases_sha = sha(cases_path.read_bytes())
    records = []
    first_break = False
    for model_index, model in enumerate(plan["models"]):
        if model_index:
            mark("turn", f"모델 전환 — {plan['models'][0]} 종료 → {model} 시작; 같은 세 과제·사례·채점")
            try:
                api_json(base_url, "/api/generate", {"model": plan["models"][0],
                         "prompt": "", "keep_alive": 0}, 15)
            except Exception as exc:
                mark("infra", f"이전 모델 언로드 확인 실패 — {type(exc).__name__}: {str(exc)[:100]}")
                raise
        else:
            mark("turn", f"코드 확장 측정 시작 — {model} · 고정 과제 3개×3회 · API 원응답")
        for task in plan["tasks"]:
            for repetition in range(1, plan["repetitions"] + 1):
                stem = f"{len(records) + 1:02d}"
                try:
                    data, elapsed = generate(base_url, model, task["prompt"], timeout)
                except Exception as exc:
                    mark("infra", f"코드 확장 호출 실패 — {model} {task['id']} {repetition}/3 "
                                  f"{type(exc).__name__}: {str(exc)[:100]}")
                    raise
                response = data["response"]
                output = run / f"{stem}-output.txt"
                log = run / f"{stem}-invocation.log"
                output.write_text(response, encoding="utf-8")
                input_sha = sha(task["prompt"].encode())
                output_sha = sha(output.read_bytes())
                record = {"model": model, "task": task["id"], "repetition": repetition,
                          "input": task["prompt"], "input_sha256": input_sha,
                          "output_file": output.name, "output_sha256": output_sha,
                          "log_file": log.name, "elapsed_s": elapsed,
                          "eval_count": data.get("eval_count"),
                          "eval_duration_ns": data.get("eval_duration")}
                lines = ["# run_local_code_expand_bench.py invocation",
                         *[f"{key}: {record[key]}" for key in
                           ("model", "task", "repetition", "input_sha256", "output_sha256",
                            "elapsed_s", "eval_count", "eval_duration_ns")],
                         f"total_duration_ns: {data.get('total_duration')}",
                         f"load_duration_ns: {data.get('load_duration')}",
                         "temperature: 0", "seed: 27", "keep_alive: 5m",
                         "--- PROMPT ---", task["prompt"], "--- RESPONSE ---"]
                log.write_text("\n".join(lines) + "\n" + response, encoding="utf-8")
                records.append(record)
                write_run_yaml(run, plan, cases_path, cases_sha, records)
                scored = score_response(task, response)
                if not first_break and scored["first_failure"]:
                    first_break = True
                    fail = scored["first_failure"]
                    mark("break", f"첫 사례 실패 — {model} {task['id']} {repetition}/3 {fail['case_id']} "
                                  f"기대 {json.dumps(fail['expected'], ensure_ascii=False)} / "
                                  f"실제 {json.dumps(fail['actual'], ensure_ascii=False)} "
                                  f"오류 {fail['error'] or '-'} 입력변형 {fail['mutated']}")
                print(f"[{len(records):02d}/18] {model} {task['id']} {repetition}/3 "
                      f"사례 {scored['cases_passed']}/{scored['cases_total']} "
                      f"계약 {scored['contract_pass']} · {elapsed}초", flush=True)
    summary = rescore_run(run, cases_path=cases_path, emit_marks=False)
    labels = [f"{model} {data['contract_pass']}/{data['responses']}회 "
              f"사례 {data['cases_passed']}/{data['cases_total']}"
              for model, data in summary["by_model"].items()]
    mark("agg", "코드 확장 두 모델 집계 — " + " · ".join(labels) + " · infra 0")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--record-score-correction", action="store_true")
    parser.add_argument("--old-scorer-sha256", default="")
    parser.add_argument("--old-aggregate", type=Path)
    parser.add_argument("--change-reason", default="")
    args = parser.parse_args()
    if args.record_score_correction:
        if not args.old_aggregate or not args.old_scorer_sha256 or not args.change_reason:
            parser.error("score correction requires --old-aggregate, --old-scorer-sha256, --change-reason")
        before = json.loads(args.old_aggregate.read_text(encoding="utf-8"))["by_model"]
        after = rescore_run(args.run_dir, cases_path=args.cases)
        scorer_sha = sha((ROOT / "code_expand_score.py").read_bytes())
        record_score_change(args.run_dir, old_sha=args.old_scorer_sha256,
                            new_sha=scorer_sha, reason=args.change_reason,
                            before=before, after=after["by_model"])
        rescore_run(args.run_dir, cases_path=args.cases)
        print(json.dumps(after, ensure_ascii=False))
    elif args.rescore:
        print(json.dumps(rescore_run(args.run_dir, cases_path=args.cases), ensure_ascii=False))
    else:
        run_live(args.run_dir, args.base_url, args.timeout, args.cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
