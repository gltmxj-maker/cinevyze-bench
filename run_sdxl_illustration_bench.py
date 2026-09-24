#!/usr/bin/env python3
"""23번 로컬 SDXL 일러스트 재실험. 생성과 제한적 자동 지표를 구분한다."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import tool_adapters
from live_mark import mark
from sdxl_illustration_score import score_image, aggregate


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "sdxl_illustration_cases.json"
DEFAULT_RUN = ROOT / "test_runs" / "comfyui-animagine-20260924"


def build_plan(cases: dict, repeats: int | None = None) -> list[dict]:
    count = int(repeats if repeats is not None else cases["repeats"])
    if count < 1:
        raise ValueError("repeats must be positive")
    plan = []
    for case in cases["cases"]:
        for repeat in range(1, count + 1):
            plan.append({"task": case["task"], "prompt": case["prompt"],
                         "repeat": repeat, "seed": int(cases["seed_start"]) + len(plan)})
    return plan


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args(argv)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    plan = build_plan(cases, args.repeats)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        # live_capture_runner creates the live directory before the child starts.
        if any(p.name != "live" for p in run_dir.iterdir()):
            raise SystemExit(f"run folder already has evidence: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "cases.json", {**cases, "repeats": args.repeats or cases["repeats"]})
    adapter = tool_adapters.ComfyUIAdapter()
    checkpoint = cases["checkpoint"]
    if not adapter.available():
        raise SystemExit("ComfyUI and installed checkpoint are unavailable; no generation attempted")
    if checkpoint not in adapter._ckpts():
        raise SystemExit(f"required checkpoint not installed: {checkpoint}")
    adapter._seed = int(cases["seed_start"])
    workflow = adapter._workflow(plan[0]["prompt"], checkpoint, plan[0]["seed"])
    condition = {
        "checkpoint": checkpoint,
        "steps": adapter._STEPS,
        "cfg": adapter._CFG,
        "sampler": adapter._SAMPLER,
        "scheduler": workflow["3"]["inputs"]["scheduler"],
        "resolution": [adapter._RES, adapter._RES],
        "negative_prompt": workflow["7"]["inputs"]["text"],
        "seed_start": cases["seed_start"],
        "repeats_per_task": args.repeats or cases["repeats"],
        "score_rule": cases["score_rule"],
    }
    run_record = {
        "tool": "comfyui", "method": "IMG-01", "date": dt.date.today().isoformat(),
        "model": "animagine", "checkpoint": checkpoint,
        "harness": "run_sdxl_illustration_bench.py", "harness_version": "1.0",
        "python": platform.python_version(), "cases_file": "cases.json", "conditions": condition,
        "condition_changes": [], "attempted_runs": len(plan),
        "score_boundary": "white corner coverage proxy only; visual usability, typography, licensing, and cost are not automatically scored",
    }
    save_json(run_dir / "run.yaml", run_record)
    mark("turn", f"IMG-01/02/03 · {checkpoint} · {len(plan)}회 시작 · 26 steps · 1024x1024")
    rows = []
    by_task = {case["task"]: case for case in cases["cases"]}
    previous_task = None
    for index, item in enumerate(plan, 1):
        if item["task"] != previous_task and previous_task is not None:
            mark("turn", f"조건 전환 · {previous_task} → {item['task']} · 프롬프트만 변경")
        previous_task = item["task"]
        started = time.monotonic()
        row = {**item, "run_index": index, "checkpoint": checkpoint}
        try:
            image = adapter.run(item["prompt"], model=checkpoint, timeout=900)
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            row["status"] = "ok"
            row["bytes"] = len(image)
            row["sha256"] = hashlib.sha256(image).hexdigest()
            row["output_file"] = f"{index:03d}-output.png"
            (run_dir / row["output_file"]).write_bytes(image)
            row["generation_meta"] = dict(adapter.last_meta)
            row.update(score_image(run_dir / row["output_file"], by_task[item["task"]]))
            if row["white_background_proxy_pass"] is False:
                mark("break", f"{item['task']}/r{item['repeat']} · 흰 배경 모서리 대리지표 실패 · 흰색 비율 {row['white_corner_fraction']:.4f}")
            print(f"[{index}/{len(plan)}] {item['task']}/r{item['repeat']} seed={item['seed']} "
                  f"elapsed={row['elapsed_s']:.3f}s bytes={row['bytes']} "
                  f"white-corner={row['white_corner_fraction']:.4f}", flush=True)
        except Exception as exc:
            row["status"] = "infra_error"
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            row["error"] = f"{type(exc).__name__}: {exc}"
            mark("infra", f"{item['task']}/r{item['repeat']} · {row['error'][:140]}")
        rows.append(row)
        save_json(run_dir / "results.json", rows)
    result = aggregate(rows)
    save_json(run_dir / "aggregate.json", result)
    mark("agg", f"완료 · 생성 {result['successful_images']}/{result['attempted_runs']} · infra {result['infra_errors']} · 흰 배경 대리지표 실패 {result['white_background_proxy_failures']}/{result['white_background_proxy_checked']}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["infra_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
