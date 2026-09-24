#!/usr/bin/env python3
"""29번: 7월의 세 고정 과제를 같은 Q4 태그로 3회씩 재측정."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from experiment_rebuild_common import ROOT, base_yaml, digest, fresh_run, generate, save_results, save_yaml, verify_text_run
from live_mark import mark


MODEL = "qwen2.5:7b-instruct-q4_K_M"
CASES = ROOT / "quant_rebuild_cases.json"
DEFAULT_RUN = ROOT / "test_runs/quant-q4-qwen2.5-20260924"


def load_cases(*, source_run: Path | None = None) -> list[dict]:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = data["cases"]
    if [c["task"] for c in cases] != ["TXT-01", "TXT-02", "TXT-04"]:
        raise ValueError("7월 태스크 연결 불일치")
    if any(not c.get("prompt") for c in cases):
        raise ValueError("입력 누락")
    import yaml
    source = source_run if source_run is not None else ROOT / data["source_run"]
    if source.is_file():
        prior_doc = yaml.safe_load(source.read_text(encoding="utf-8"))
        if prior_doc.get("model") != MODEL:
            raise ValueError("7월 원기록 모델 불일치")
        prior = prior_doc["runs"]
        if [prior[i]["input"] for i in (0, 2, 4)] != [c["prompt"] for c in cases]:
            raise ValueError("7월 원기록의 고정 입력과 불일치")
    elif source_run is None and (ROOT / "test_runs/ollama-qwen2.5:7b-instruct-q4_k_m-20260726").is_dir():
        raise FileNotFoundError(f"이 저장소의 7월 고정 입력 원기록 없음: {source}")
    # 공개 저장소는 비공개 test_runs를 배포하지 않는다. 커밋된 cases JSON이 재현 입력이다.
    return cases


def score_answer(task: str, answer: str) -> dict:
    if task == "TXT-01":
        hangul = bool(re.search("[가-힣]", answer))
        return {"pass": hangul, "rule": "한글 음절 존재만 진단; 요약 품질 점수 아님"}
    if task == "TXT-02":
        # 이 과제의 엄격한 실행 채점은 7월 코드 셀 평가와 별도. 여기서는 최소 계약만.
        return {"pass": "dedup_sort" in answer and all(x in answer for x in ("3", "2", "1")),
                "rule": "함수명·예시 세 수 존재(코드 실행 품질 UNKNOWN)"}
    if task == "TXT-04":
        bits = ["세종" in answer, "이성계" in answer,
                "1592" in answer and "1682" in answer,
                any(x in answer for x in ("존재하지", "없습니다", "없다")),
                "천지" in answer]
        return {"pass": all(bits), "items": bits,
                "rule": "5개 정답 키워드; 1682 오류 지적 포함. 모순·문맥 판정은 사람 검수"}
    raise ValueError(task)


def run_live(run: Path, base_url: str) -> None:
    fresh_run(run)
    cases = load_cases()
    meta = base_yaml(Path(__file__).name, "TXT-01 + TXT-02 + TXT-04", MODEL)
    meta.update({"cases_file": CASES.name, "cases_sha256": digest(CASES.read_bytes()),
                 "generation_options": {"temperature": 0, "seed": 29}, "repetitions": 3,
                 "score_rules": "run_quant_rebuild_bench.py:score_answer; TXT-02 코드 실행 품질 미판정",
                 "condition_changes": []})
    rows = []
    mark("turn", "Q4_K_M 단일 모델 — 7월 고정 질문 3종을 각 3회 재측정")
    first_break = False
    for ci, case in enumerate(cases):
        if ci:
            mark("turn", f"과제 전환 — {case['task']} · 동일 Q4 체크포인트")
        for rep in range(1, 4):
            stem = f"{len(rows)+1:02d}"
            try:
                data, ps, elapsed = generate(base_url, MODEL, case["prompt"])
            except Exception as exc:
                mark("infra", f"Q4 {case['task']} {rep}/3 호출 또는 /api/ps 실패: {type(exc).__name__} {exc}")
                raise
            answer = data["response"]
            verdict = score_answer(case["task"], answer)
            output = run / f"{stem}-output.txt"
            log = run / f"{stem}-invocation.log"
            output.write_text(answer, encoding="utf-8")
            row = {"task": case["task"], "repetition": rep, "model": MODEL,
                   "output_file": output.name, "log_file": log.name,
                   "input_sha256": digest(case["prompt"].encode()),
                   "output_sha256": digest(output.read_bytes()),
                   "elapsed_s": elapsed, "eval_count": data.get("eval_count"),
                   "eval_duration_ns": data.get("eval_duration"),
                   "ps_size_bytes": ps["size"], "ps_size_vram_bytes": ps["size_vram"],
                   "pass": verdict["pass"], "score_rule": verdict["rule"]}
            log.write_text(json.dumps({"request": {"model": MODEL, "prompt": case["prompt"],
                "temperature": 0, "seed": 29}, "response": data, "ps": ps, "row": row},
                ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            meta["runs"].append({"task": case["task"], "input": case["prompt"],
                                 "output_file": output.name, "log_file": log.name,
                                 "elapsed_s": elapsed, "input_sha256": row["input_sha256"],
                                 "output_sha256": row["output_sha256"]})
            save_yaml(run, meta)
            print(f"[{stem}/09] {case['task']} {rep}/3 · {elapsed}s · VRAM {ps['size_vram']/1e9:.2f}GB · 진단 {verdict['pass']}", flush=True)
            if not verdict["pass"] and not first_break:
                mark("break", f"Q4 첫 규칙 이탈 — {case['task']} {rep}/3 · {stem}-output.txt · 진단 {verdict['rule']}")
                first_break = True
    aggregate = {"sample_n": len(rows), "model": MODEL, "infra_errors": 0,
                 "median_ps_size_vram_bytes": int(statistics.median(r["ps_size_vram_bytes"] for r in rows)),
                 "median_token_per_s": round(statistics.median(r["eval_count"] / (r["eval_duration_ns"] / 1e9)
                                                          for r in rows), 2),
                 "pass_by_task": {c["task"]: sum(r["pass"] for r in rows if r["task"] == c["task"])
                                  for c in cases}, "n_by_task": {c["task"]: 3 for c in cases}}
    save_results(run, rows, aggregate)
    mark("agg", f"Q4 고정 3과제×3회 — 중앙 GPU 적재 {aggregate['median_ps_size_vram_bytes']/1e9:.2f}GB · 진단 {aggregate['pass_by_task']} · infra 0")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_text_run(args.run_dir), ensure_ascii=False))
    else:
        run_live(args.run_dir, args.base_url)


if __name__ == "__main__":
    main()
