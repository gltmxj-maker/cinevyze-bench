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
from sdxl_illustration_score import score_image, aggregate, score_run_dir


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


def rebuild_run_record(run_dir: Path, manifest_path: Path | None = None) -> dict:
    """Recreate the write-origin record from captured results, PNGs and live evidence.

    This is an offline repair path. It never invokes ComfyUI or invents a new run.
    """
    cases = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    plan = build_plan(cases)
    if len(rows) != len(plan):
        raise ValueError(f"회차 불일치: results={len(rows)}, 계획={len(plan)}")
    for expected, row in zip(plan, rows):
        for key in ("task", "prompt", "repeat", "seed"):
            if row.get(key) != expected[key]:
                raise ValueError(f"실행 계획 불일치: {key} {row.get(key)!r} != {expected[key]!r}")
        if row.get("status") != "ok":
            raise ValueError(f"미완료 회차: {row.get('run_index')}")
        if (row.get("generation_meta") or {}).get("seed") != row["seed"]:
            raise ValueError(f"전송 시드 불일치: {row.get('run_index')}")
        if (row.get("generation_meta") or {}).get("checkpoint") != cases["checkpoint"]:
            raise ValueError(f"체크포인트 불일치: {row.get('run_index')}")
    # score_run_dir checks byte counts and SHA256 for every PNG, then writes the
    # same deterministic score from the persisted image rather than trusting a flag.
    aggregate_result = score_run_dir(run_dir)
    if aggregate_result["successful_images"] != len(plan):
        raise ValueError("집계의 성공 회차 수 불일치")

    if manifest_path is None:
        date = dt.date.today().isoformat()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_dir") not in (None, f"test_runs/{run_dir.name}"):
            raise ValueError("캡처 manifest 런 경로 불일치")
        shots = manifest.get("images") or []
        if not any(" agg " in f" {shot.get('desc', '')} " for shot in shots):
            raise ValueError("본실행 집계 캡처 없음")
        for shot in shots:
            filename = shot.get("png") or ""
            if Path(filename).name != filename or not filename:
                raise ValueError("캡처 파일명 오류")
            image = manifest_path.parent / filename
            if not image.is_file():
                raise FileNotFoundError(image)
            if hashlib.sha256(image.read_bytes()).hexdigest() != shot.get("src_sha256"):
                raise ValueError(f"캡처 SHA256 불일치: {image}")
        date = str(manifest["captured_at"])[:10]
    dt.date.fromisoformat(date)

    adapter = tool_adapters.ComfyUIAdapter()
    first = rows[0]["generation_meta"]
    workflow = adapter._workflow(rows[0]["prompt"], cases["checkpoint"], rows[0]["seed"])
    for row in rows:
        meta = row["generation_meta"]
        for key in ("steps", "cfg", "sampler", "resolution"):
            if meta.get(key) != first.get(key):
                raise ValueError(f"생성 조건 혼합: {key}")
    if (first["steps"], first["cfg"], first["sampler"], first["resolution"]) != (
            adapter._STEPS, adapter._CFG, adapter._SAMPLER,
            f"{adapter._RES}x{adapter._RES}"):
        raise ValueError("기록된 생성 조건과 하네스 기본값 불일치")

    prior_path = run_dir / "run.yaml"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    return {
        "tool": "comfyui", "method": "IMG-01", "date": date,
        "access": "image", "model": "animagine", "checkpoint": cases["checkpoint"],
        "generated_by": Path(__file__).name, "harness_version": "1.1",
        "python": prior.get("python") or platform.python_version(),
        "cases_file": "cases.json", "conditions": {
            "checkpoint": cases["checkpoint"], "steps": first["steps"],
            "cfg": first["cfg"], "sampler": first["sampler"],
            "scheduler": workflow["3"]["inputs"]["scheduler"],
            "resolution": [int(n) for n in first["resolution"].split("x")],
            "negative_prompt": workflow["7"]["inputs"]["text"],
            "seed_start": cases["seed_start"], "repeats_per_task": cases["repeats"],
            "score_rule": cases["score_rule"],
        },
        "condition_changes": prior.get("condition_changes", []),
        "attempted_runs": len(rows),
        "score_boundary": "white corner coverage proxy only; visual usability, typography, licensing, and cost are not automatically scored",
        "code_provenance": prior.get("code_provenance"),
        "rescore": prior.get("rescore"),
        "runs": [{
            "task": row["task"], "repeat": row["repeat"], "seed": row["seed"],
            "input": row["prompt"], "output_file": row["output_file"],
            "screenshot": row["output_file"], "log_file": "results.json",
            "elapsed_s": row["elapsed_s"], "sha256": row["sha256"],
        } for row in rows],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--rebuild-record", action="store_true",
                        help="기존 PNG·results·실촬영 manifest를 검증해 run.yaml 형식만 재생성(GPU 없음)")
    parser.add_argument("--capture-manifest", type=Path,
                        help="--rebuild-record 때 확인할 실제 촬영 manifest 경로")
    args = parser.parse_args(argv)
    if args.rebuild_record:
        if not args.capture_manifest:
            parser.error("--rebuild-record 는 --capture-manifest 필요")
        run_dir = args.run_dir.resolve()
        record = rebuild_run_record(run_dir, args.capture_manifest.resolve())
        save_json(run_dir / "run.yaml", record)
        print(f"[record] {run_dir / 'run.yaml'} · 검증된 {len(record['runs'])}회")
        return 0
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
    if result["infra_errors"] == 0:
        save_json(run_dir / "run.yaml", rebuild_run_record(run_dir))
    mark("agg", f"완료 · 생성 {result['successful_images']}/{result['attempted_runs']} · infra {result['infra_errors']} · 흰 배경 대리지표 실패 {result['white_background_proxy_failures']}/{result['white_background_proxy_checked']}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["infra_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
