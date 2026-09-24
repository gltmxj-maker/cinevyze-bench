#!/usr/bin/env python3
"""27번 TXT-02: qwen2.5-coder:7b 단일 모델 재측정과 전건 재채점."""
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

from live_mark import mark
from localcloud_score import CODING_CASES, score_coding

MODEL = "qwen2.5-coder:7b"
PROMPT = ("파이썬으로 주어진 정수 리스트에서 중복을 제거하고 내림차순 정렬하는 함수 "
          "dedup_sort(nums)를 작성하고, 예시 입력 [3,1,2,3,1]에 대한 출력도 보여줘.")
PLAN = (1, 2, 3)
DEFAULT_RUN = Path(__file__).resolve().parent / "test_runs" / f"local-code-qwen2.5-coder-{dt.date.today():%Y%m%d}"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")


def generate(base_url: str, timeout: int) -> tuple[dict, float]:
    payload = {"model": MODEL, "prompt": PROMPT, "stream": False, "keep_alive": 0}
    request = urllib.request.Request(base_url.rstrip("/") + "/api/generate",
                                     data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    elapsed = round(time.monotonic() - start, 3)
    if not isinstance(data, dict) or not isinstance(data.get("response"), str):
        raise ValueError("Ollama 응답 형식 불일치")
    if not data["response"].strip():
        raise ValueError("Ollama 빈 응답")
    if data.get("model") != MODEL:
        raise ValueError(f"응답 모델 불일치: {data.get('model')!r}")
    return data, elapsed


def write_run_yaml(run: Path, records: list[dict]) -> None:
    payload = {
        "tool": "ollama", "date": dt.date.today().isoformat(), "method": "TXT-02",
        "access": "local", "model": MODEL, "generated_by": "run_local_code_bench.py",
        "harness_version": "1.0", "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0, "request_parallelism": 1,
        "input_sha256": digest(PROMPT.encode()),
        "score_rules": "localcloud_score.py:score_coding; CODING_CASES 5건; 예시 출력 요구",
        "environment": {"python": platform.python_version(),
                        "ollama_models": os.environ.get("OLLAMA_MODELS")},
        "runs": records, "compare": [], "disclosure": True,
    }
    temp = run / "run.yaml.tmp"
    temp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temp, run / "run.yaml")


def rescore_run(run: Path, *, emit_marks: bool = False) -> dict:
    metadata = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    records = metadata.get("runs") or []
    if (metadata.get("model") != MODEL or metadata.get("generated_by") != "run_local_code_bench.py"
            or len(records) != len(PLAN)):
        raise ValueError("모델·기록자·회차 수 불일치")
    rows = []
    first_break = False
    for index, item in enumerate(records, 1):
        stem = f"{index:02d}"
        output = run / f"{stem}-output.txt"
        log = run / f"{stem}-invocation.log"
        if (item.get("task") != "TXT-02" or item.get("input") != PROMPT
                or item.get("output_file") != output.name or item.get("log_file") != log.name):
            raise ValueError(f"TXT-02 입력·회차 연결 불일치: {stem}")
        raw = output.read_bytes()
        response_text = raw.decode("utf-8")
        transcript = log.read_text(encoding="utf-8")
        expected_input = digest(PROMPT.encode())
        expected_output = digest(raw)
        if (item.get("input_sha256") != expected_input
                or item.get("output_sha256") != expected_output
                or f"input_sha256: {expected_input}" not in transcript
                or f"output_sha256: {expected_output}" not in transcript
                or f"model: {MODEL}" not in transcript
                or "task: TXT-02" not in transcript
                or transcript.split("--- RESPONSE ---" + chr(10), 1)[-1] != response_text):
            raise ValueError(f"SHA256·호출 로그 불일치: {stem}")
        header = transcript.split("--- RESPONSE ---" + chr(10), 1)[0]
        timings = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
        if float(timings.get("elapsed_s", "nan")) != item.get("elapsed_s"):
            raise ValueError(f"시간 로그 불일치: {stem}")
        for field, log_field in (("eval_count", "eval_count"),
                                 ("eval_duration_ns", "eval_duration_ns")):
            if item.get(field) is not None and str(item[field]) != timings.get(log_field):
                raise ValueError(f"API 지표 로그 불일치: {stem}/{field}")
        score = score_coding(response_text)
        if emit_marks and not first_break and not score["coding_contract_pass"]:
            first_break = True
            mark("break", f"코드 과업 첫 실패 — qwen2.5-coder:7b TXT-02 {stem}/03 · "
                          f"기능 사례 {score['functional_cases_passed']}/{len(CODING_CASES)} · "
                          f"예시 출력 {score['example_output_shown']}")
        rows.append({"task": "TXT-02", "repetition": index, "model": MODEL,
                     "output_file": output.name, "log_file": log.name,
                     "input_sha256": expected_input, "output_sha256": expected_output,
                     "elapsed_s": item["elapsed_s"], "eval_count": item.get("eval_count"),
                     "eval_duration_ns": item.get("eval_duration_ns"), **score})
    write_json(run / "results.json", rows)
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as file:
        columns = ("repetition", "output_file", "elapsed_s", "coding_contract_pass",
                   "functional_cases_passed", "functional_cases_total", "example_output_shown")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    speeds = [row["eval_count"] / (row["eval_duration_ns"] / 1e9)
              for row in rows if row.get("eval_count") and row.get("eval_duration_ns")]
    summary = {"model": MODEL, "sample_n": len(rows), "infra_errors": 0,
               "contract_pass": sum(row["coding_contract_pass"] for row in rows),
               "functional_cases_passed": sum(row["functional_cases_passed"] for row in rows),
               "functional_cases_total": len(rows) * len(CODING_CASES),
               "median_elapsed_s": round(statistics.median(row["elapsed_s"] for row in rows), 3),
               "median_token_per_s": round(statistics.median(speeds), 2) if speeds else None,
               "input_sha256": digest(PROMPT.encode())}
    write_json(run / "aggregate.json", summary)
    if emit_marks:
        mark("agg", f"코드 TXT-02 qwen2.5-coder:7b 3회 집계 — 계약 "
                    f"{summary['contract_pass']}/3 · 실제 실행 사례 "
                    f"{summary['functional_cases_passed']}/{summary['functional_cases_total']} · "
                    f"중앙 대기 {summary['median_elapsed_s']}초 · infra 0건")
    return summary


def run_live(run: Path, base_url: str, timeout: int) -> dict:
    if run.exists() and any(path.name != "live" for path in run.iterdir()):
        raise ValueError(f"런 폴더가 비어 있지 않습니다: {run}")
    run.mkdir(parents=True, exist_ok=True)
    mark("turn", "코드 TXT-02 측정 시작 — qwen2.5-coder:7b · 고정 입력 · 독립 요청 3회 · keep_alive=0")
    records = []
    for index in PLAN:
        stem = f"{index:02d}"
        try:
            data, elapsed = generate(base_url, timeout)
        except Exception as exc:
            mark("infra", f"코드 TXT-02 {stem}/03 호출 실패 — {type(exc).__name__}: {str(exc)[:100]}")
            raise
        output = run / f"{stem}-output.txt"
        log = run / f"{stem}-invocation.log"
        output.write_text(data["response"], encoding="utf-8")
        output_sha = digest(output.read_bytes())
        input_sha = digest(PROMPT.encode())
        lines = [
            "# run_local_code_bench.py invocation", f"model: {MODEL}", "task: TXT-02",
            f"repetition: {index}", f"input_sha256: {input_sha}",
            f"output_sha256: {output_sha}", f"elapsed_s: {elapsed}",
            f"eval_count: {data.get('eval_count')}",
            f"eval_duration_ns: {data.get('eval_duration')}",
            f"total_duration_ns: {data.get('total_duration')}",
            f"load_duration_ns: {data.get('load_duration')}", "keep_alive: 0",
            "--- PROMPT ---", PROMPT, "--- RESPONSE ---",
        ]
        log.write_text(chr(10).join(lines) + chr(10) + data["response"], encoding="utf-8")
        records.append({"task": "TXT-02", "input": PROMPT, "output_file": output.name,
                        "log_file": log.name, "elapsed_s": elapsed, "input_sha256": input_sha,
                        "output_sha256": output_sha, "eval_count": data.get("eval_count"),
                        "eval_duration_ns": data.get("eval_duration")})
        write_run_yaml(run, records)
        print(f"[{index}/3] qwen2.5-coder:7b · {elapsed}초 · 출력 {len(output.read_bytes())}B", flush=True)
    summary = rescore_run(run, emit_marks=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    if args.rescore:
        print(json.dumps(rescore_run(args.run_dir, emit_marks=True), ensure_ascii=False))
    else:
        run_live(args.run_dir, args.base_url, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
