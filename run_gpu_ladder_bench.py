#!/usr/bin/env python3
"""28번: TXT-01 고정 입력으로 qwen2.5:7b VRAM·속도를 다시 잰다."""
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


ROOT = Path(__file__).resolve().parent
MODEL = "qwen2.5:7b"
PLAN = (1, 2, 3)
DEFAULT_RUN = ROOT / "test_runs" / f"gpu-ladder-qwen2.5-7b-{dt.date.today():%Y%m%d}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_prompt(path: Path = ROOT / "gpu_ladder_cases.json") -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("task") != "TXT-01" or not isinstance(data.get("prompt"), str):
        raise ValueError("TXT-01 고정 입력 파일 형식 불일치")
    return data["prompt"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api_json(base_url: str, endpoint: str, payload: dict | None, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Ollama {endpoint} 응답 형식 불일치")
    return result


def generate(base_url: str, prompt: str, timeout: int) -> tuple[dict, dict, float]:
    start = time.monotonic()
    data = api_json(base_url, "/api/generate", {"model": MODEL, "prompt": prompt,
                    "stream": False, "keep_alive": "5m"}, timeout)
    elapsed = round(time.monotonic() - start, 3)
    if data.get("model") != MODEL or not str(data.get("response") or "").strip():
        raise ValueError("Ollama 모델 불일치 또는 빈 응답")
    ps = api_json(base_url, "/api/ps", None, 10)
    loaded = next((item for item in ps.get("models", []) if item.get("name") == MODEL), None)
    if not loaded or not isinstance(loaded.get("size"), int) or not isinstance(loaded.get("size_vram"), int):
        raise ValueError("Ollama /api/ps 에 지정 모델·메모리 바이트 없음")
    return data, loaded, elapsed


def write_run_yaml(run: Path, prompt: str, records: list[dict]) -> None:
    payload = {"tool": "ollama", "date": dt.date.today().isoformat(), "method": "TXT-01",
               "access": "local", "model": MODEL, "generated_by": Path(__file__).name,
               "harness_version": "1.0", "tos_confirmed": True,
               "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
               "input_sha256": sha(prompt.encode()), "request_parallelism": 1,
               "keep_alive": "5m", "score_rules": "Ollama eval_count/eval_duration; /api/ps size_vram",
               "environment": {"python": platform.python_version(),
                               "ollama_models": os.environ.get("OLLAMA_MODELS")},
               "runs": records, "compare": [], "disclosure": True}
    temp = run / "run.yaml.tmp"
    temp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temp, run / "run.yaml")


def rescore_run(run: Path, *, emit_marks: bool = True,
                cases_path: Path = ROOT / "gpu_ladder_cases.json") -> dict:
    run = Path(run)
    prompt = load_prompt(cases_path)
    meta = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    records = meta.get("runs") or []
    if (meta.get("model") != MODEL or meta.get("generated_by") != Path(__file__).name
            or meta.get("input_sha256") != sha(prompt.encode()) or len(records) != len(PLAN)):
        raise ValueError("모델·기록자·고정 입력·회차 수 불일치")
    rows = []
    breaks = 0
    for index, item in enumerate(records, 1):
        stem = f"{index:02d}"
        output = run / f"{stem}-output.txt"
        log = run / f"{stem}-invocation.log"
        if (item.get("task") != "TXT-01" or item.get("input") != prompt
                or item.get("output_file") != output.name or item.get("log_file") != log.name):
            raise ValueError(f"TXT-01 회차 연결 불일치: {stem}")
        raw = output.read_bytes()
        transcript = log.read_text(encoding="utf-8")
        header, response = transcript.split("--- RESPONSE ---\n", 1)
        if (item.get("input_sha256") != sha(prompt.encode()) or
                item.get("output_sha256") != sha(raw) or response != raw.decode() or
                "--- PROMPT ---\n" + prompt + "\n" not in header):
            raise ValueError(f"SHA256·로그 불일치: {stem}")
        logged = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
        if (logged.get("model") != MODEL or logged.get("task") != "TXT-01" or
                logged.get("repetition") != str(index) or
                logged.get("input_sha256") != sha(prompt.encode()) or
                logged.get("output_sha256") != sha(raw)):
            raise ValueError(f"SHA256·호출 로그 불일치: {stem}")
        for key in ("elapsed_s", "eval_count", "eval_duration_ns", "ps_size_bytes", "ps_size_vram_bytes"):
            if str(item.get(key)) != logged.get(key):
                raise ValueError(f"API/PS 지표 로그 불일치: {stem}/{key}")
        eval_ns = int(item["eval_duration_ns"])
        if eval_ns <= 0 or int(item["eval_count"]) <= 0:
            raise ValueError(f"생성 토큰·시간 비정상: {stem}")
        speed = round(int(item["eval_count"]) / (eval_ns / 1e9), 2)
        vram = int(item["ps_size_vram_bytes"])
        total = int(item["ps_size_bytes"])
        if vram < 0 or total <= 0 or vram > total:
            raise ValueError(f"PS 메모리 비정상: {stem}")
        hangul = any("가" <= char <= "힣" for char in raw.decode())
        if emit_marks and vram < total and breaks < 2:
            mark("break", f"7B GPU 전량 적재 실패 — {stem}/03 · GPU {vram / 1e9:.2f}GB / 모델 {total / 1e9:.2f}GB")
            breaks += 1
        if emit_marks and not hangul and breaks < 2:
            mark("break", f"한국어 요약 지시 이탈 — {stem}/03 출력에 한글 음절 0개")
            breaks += 1
        rows.append({"task": "TXT-01", "repetition": index, "model": MODEL,
                     "output_file": output.name, "log_file": log.name,
                     "input_sha256": sha(prompt.encode()), "output_sha256": sha(raw),
                     "elapsed_s": item["elapsed_s"], "eval_count": item["eval_count"],
                     "eval_duration_ns": eval_ns, "token_per_s": speed,
                     "ps_size_bytes": total, "ps_size_vram_bytes": vram,
                     "gpu_loaded_all": vram == total, "hangul_present": hangul})
    write_json(run / "results.json", rows)
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": MODEL, "sample_n": len(rows), "infra_errors": 0,
               "median_elapsed_s": statistics.median(row["elapsed_s"] for row in rows),
               "median_token_per_s": statistics.median(row["token_per_s"] for row in rows),
               "median_ps_size_vram_bytes": int(statistics.median(row["ps_size_vram_bytes"] for row in rows)),
               "gpu_loaded_all_count": sum(row["gpu_loaded_all"] for row in rows),
               "hangul_present_count": sum(row["hangul_present"] for row in rows),
               "input_sha256": sha(prompt.encode())}
    write_json(run / "aggregate.json", summary)
    if emit_marks:
        mark("agg", f"7B TXT-01 3회 집계 — GPU 전량 {summary['gpu_loaded_all_count']}/3 · "
                    f"중앙 생성 {summary['median_token_per_s']}tok/s · "
                    f"중앙 적재 {summary['median_ps_size_vram_bytes'] / 1e9:.2f}GB · infra 0")
    return summary


def run_live(run: Path, base_url: str, timeout: int) -> dict:
    run = Path(run)
    if run.exists() and any(path.name != "live" for path in run.iterdir()):
        raise FileExistsError(f"런 폴더가 비어 있지 않습니다: {run}")
    run.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt()
    mark("turn", "7B TXT-01 측정 시작 — qwen2.5:7b · 고정 한국어 요약 입력 · 순차 3회")
    records = []
    for index in PLAN:
        stem = f"{index:02d}"
        try:
            data, loaded, elapsed = generate(base_url, prompt, timeout)
        except Exception as exc:
            mark("infra", f"7B TXT-01 {stem}/03 호출·PS 실패 — {type(exc).__name__}: {str(exc)[:100]}")
            raise
        out = run / f"{stem}-output.txt"
        log = run / f"{stem}-invocation.log"
        out.write_text(data["response"], encoding="utf-8")
        output_sha = sha(out.read_bytes())
        input_sha = sha(prompt.encode())
        fields = {"model": MODEL, "task": "TXT-01", "repetition": index,
                  "input_sha256": input_sha, "output_sha256": output_sha,
                  "elapsed_s": elapsed, "eval_count": data.get("eval_count"),
                  "eval_duration_ns": data.get("eval_duration"),
                  "ps_size_bytes": loaded["size"], "ps_size_vram_bytes": loaded["size_vram"]}
        if not isinstance(fields["eval_count"], int) or not isinstance(fields["eval_duration_ns"], int):
            raise ValueError("Ollama 생성 토큰·시간 지표 없음")
        log.write_text("# run_gpu_ladder_bench.py invocation\n" +
                       "\n".join(f"{key}: {value}" for key, value in fields.items()) +
                       "\n--- PROMPT ---\n" + prompt + "\n--- RESPONSE ---\n" + data["response"],
                       encoding="utf-8")
        records.append({"task": "TXT-01", "input": prompt, "output_file": out.name,
                        "log_file": log.name, **{k: v for k, v in fields.items() if k not in
                        ("model", "task", "repetition")}})
        write_run_yaml(run, prompt, records)
        print(f"[{index}/3] qwen2.5:7b · {elapsed}s · {loaded['size_vram'] / 1e9:.2f}GB GPU", flush=True)
        if index == 1:
            mark("turn", "7B 첫 적재 완료 — 이후 2회는 상주 모델 반복 요청(냉시작과 분리)")
    result = rescore_run(run, emit_marks=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    if args.rescore:
        print(json.dumps(rescore_run(args.run_dir), ensure_ascii=False))
    else:
        run_live(args.run_dir, args.base_url, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
