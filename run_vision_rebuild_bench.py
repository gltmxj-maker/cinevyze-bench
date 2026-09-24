#!/usr/bin/env python3
"""31번: 7월 자작 이미지 4장을 두 설치 비전 모델에 순차 3회씩 질문."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
from pathlib import Path

from experiment_rebuild_common import ROOT, base_yaml, digest, fresh_run, generate, materialize_short_outputs, save_results, save_yaml, verify_text_run
from live_mark import mark


MODELS = ["qwen2.5vl:7b", "qwen3-vl:8b"]
IMAGE_DIR = Path(os.environ.get("VISION_INPUT_DIR", str(ROOT / "corpus/vision-20260723")))
CASES = [
    {"id": "menu", "image": str(IMAGE_DIR / "01_menu.png"),
     "question": "이 메뉴판에서 가장 비싼 메뉴와 그 가격은?", "ground_truth": "콜드브루 6,000원"},
    {"id": "chart", "image": str(IMAGE_DIR / "02_chart.png"),
     "question": "이 막대차트에서 매출이 가장 높은 달은?", "ground_truth": "4월(90만원)"},
    {"id": "table", "image": str(IMAGE_DIR / "03_table.png"),
     "question": "표에서 세 과목 총점이 더 높은 팀은? 두 팀 총점도 알려줘.",
     "ground_truth": "B팀 (A팀 250 vs B팀 265)"},
    {"id": "shapes", "image": str(IMAGE_DIR / "04_shapes.png"),
     "question": "이 그림에 빨간색 원은 몇 개 있어?", "ground_truth": "3개"},
]
DEFAULT_RUN = ROOT / "test_runs/vision-local-20260924"


def load_cases() -> list[dict]:
    cases = []
    for item in CASES:
        case = dict(item)
        path = Path(case["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        case["sha256"] = digest(path.read_bytes())
        cases.append(case)
    return cases


def score(case_id: str, answer: str) -> bool:
    compact = re.sub(r"[\s,*]", "", answer).lower()
    if case_id == "menu":
        return "콜드브루" in compact and "6000" in compact
    if case_id == "chart":
        return "4월" in compact
    if case_id == "table":
        return ("b팀" in compact or "teamb" in compact) and "250" in compact and "265" in compact
    if case_id == "shapes":
        return bool(re.search(r"(^|\D)3(\D|$)", answer) or "세 개" in answer or "3개" in compact)
    raise ValueError(case_id)


def run_live(run: Path, base_url: str) -> None:
    fresh_run(run)
    cases = load_cases()
    meta = base_yaml(Path(__file__).name, "VIS-01", ", ".join(MODELS))
    meta.update({"cases": cases, "repetitions": 3,
                 "generation_options": {"temperature": 0, "seed": 31},
                 "score_rules": "run_vision_rebuild_bench.py:score; 4장 정답 고정",
                 "condition_changes": []})
    rows = []
    first_break = False
    for mi, model in enumerate(MODELS):
        mark("turn", f"비전 모델 전환 — {model} · 동일 자작 이미지 4장 × 3회")
        for rep in range(1, 4):
            for case in cases:
                stem = f"{len(rows)+1:02d}"
                data_image = base64.b64encode(Path(case["image"]).read_bytes()).decode("ascii")
                try:
                    data, ps, elapsed = generate(base_url, model, case["question"],
                                                 images=[data_image], seed=31, timeout=300)
                except Exception as exc:
                    mark("infra", f"31번 {model} {case['id']} 반복 {rep} 실패: {type(exc).__name__} {exc}")
                    raise
                answer = data["response"]
                output = run / f"{stem}-output.txt"
                log = run / f"{stem}-invocation.log"
                output.write_text(answer, encoding="utf-8")
                row = {"task": "VIS-01", "model": model, "case": case["id"],
                       "repetition": rep, "image": case["image"], "image_sha256": case["sha256"],
                       "question": case["question"], "ground_truth": case["ground_truth"],
                       "pass": score(case["id"], answer), "answer": answer,
                       "output_file": output.name, "log_file": log.name,
                       "output_sha256": digest(output.read_bytes()), "elapsed_s": elapsed,
                       "eval_count": data.get("eval_count"), "eval_duration_ns": data.get("eval_duration"),
                       "ps_size_vram_bytes": ps["size_vram"]}
                log.write_text(json.dumps({"request": {"model": model, "question": case["question"],
                                "image": case["image"], "image_sha256": case["sha256"], "seed": 31},
                                "response": data, "ps": ps, "row": row}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                rows.append(row)
                meta["runs"].append({"task": "VIS-01", "input": case["image"] + " :: " + case["question"],
                                     "model": model, "output_file": output.name,
                                     "log_file": log.name, "elapsed_s": elapsed,
                                     "output_sha256": row["output_sha256"]})
                save_yaml(run, meta)
                print(f"[{stem}/24] {model} {case['id']} 반복 {rep} · {'맞음' if row['pass'] else '틀림'} · {elapsed}s", flush=True)
                if not row["pass"] and not first_break:
                    mark("break", f"비전 첫 오답 — {model} · {case['id']} 반복 {rep} · {output.name}")
                    first_break = True
        # 한 모델씩 GPU를 반납한다. /api/generate 빈 프롬프트 + keep_alive=0은 언로드 명령.
        from experiment_rebuild_common import api
        api(base_url, "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    aggregate = {"sample_n": len(rows), "cases_n": 4, "repetitions": 3, "infra_errors": 0,
                 "models": {model: {"correct": sum(r["pass"] for r in rows if r["model"] == model),
                                    "n": 12, "median_elapsed_s": statistics.median(r["elapsed_s"] for r in rows if r["model"] == model),
                                    "median_ps_size_vram_bytes": int(statistics.median(r["ps_size_vram_bytes"] for r in rows if r["model"] == model))}
                            for model in MODELS}}
    save_results(run, rows, aggregate)
    materialize_short_outputs(run)
    mark("agg", "비전 4장×3회·두 모델 — " + "; ".join(
         f"{m} 정답 {aggregate['models'][m]['correct']}/12" for m in MODELS) + " · infra 0")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_text_run(args.run_dir), ensure_ascii=False))
    elif args.verify_evidence:
        print(f"API 원응답 증거 연결 {materialize_short_outputs(args.run_dir)}건; 원문 답변은 변경하지 않음")
    else:
        run_live(args.run_dir, args.base_url)


if __name__ == "__main__":
    main()
