#!/usr/bin/env python3
"""24번: 소유 일러스트 3장을 APISR 4x와 Lanczos로 복원해 원본과 비교한다.

run.yaml, 결과 JSON, 호출 로그, 캡처 마커는 이 실행 파일의 실측 결과에서만 쓴다.
--rescore는 저장된 PNG와 SHA256을 대조한 뒤 같은 규칙으로 전건 다시 계산한다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from live_mark import mark


ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("BENCH_CORPUS_DIR", ROOT / "corpus")) / "ups"
DEFAULT_RUN = ROOT / "test_runs" / f"upscale-apisr4x-{dt.date.today():%Y%m%d}"
PYTHON = Path(os.environ.get("UPSCALE_PYTHON", os.path.expanduser("~/mcf-tools/ComfyUI/.venv/bin/python")))
WORKER = Path(os.environ.get("UPSCALE_WORKER", ROOT / "upscale_worker.py"))
MODEL = Path(os.environ.get("APISR4X_PATH", "./models/4x_APISR_GRL_GAN_generator.pth"))
CASES = (("ups01", "마스코트(플랫·볼드)"), ("ups02", "차트(선·디테일)"),
         ("ups03", "복잡한 일러스트"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pair_metrics(original: Image.Image, comparison: Image.Image) -> dict:
    a, b = np.asarray(original), np.asarray(comparison)
    if a.shape != b.shape:
        raise ValueError(f"출력 해상도 불일치: {a.shape} != {b.shape}")
    return {
        "psnr": round(float(peak_signal_noise_ratio(a, b, data_range=255)), 2),
        "ssim": round(float(structural_similarity(a, b, channel_axis=2, data_range=255)), 4),
    }


def score_case(orig: Path, inp: Path, output: Path) -> tuple[dict, dict]:
    with Image.open(orig) as image:
        original = image.convert("RGB")
    with Image.open(inp) as image:
        low = image.convert("RGB")
    with Image.open(output) as image:
        upscaled = image.convert("RGB")
    if original.size != (low.width * 4, low.height * 4):
        raise ValueError(f"원본·입력 4x 관계 불일치: {inp}")
    if ImageChops.difference(original.resize(low.size, Image.Resampling.LANCZOS), low).getbbox():
        raise ValueError(f"원본 Lanczos 축소본과 입력 불일치: {inp}")
    baseline = low.resize(original.size, Image.Resampling.LANCZOS)
    return _pair_metrics(original, baseline), _pair_metrics(original, upscaled)


def rescore_run(run_dir: Path, corpus: Path = CORPUS) -> dict:
    """저장된 3장과 원본 입력을 해시 확인하고 동일 지표를 전건 재채점한다."""
    run_dir, corpus = Path(run_dir), Path(corpus)
    recorded = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if len(recorded) != len(CASES):
        raise ValueError(f"계획 회차 불일치: {len(recorded)} != {len(CASES)}")
    rows = []
    for index, ((sid, kind), item) in enumerate(zip(CASES, recorded), 1):
        if item.get("task") != f"UPS-{index:02d}" or item.get("img") != sid:
            raise ValueError(f"회차 순서 불일치: {index}")
        if item.get("status") != "ok":
            raise ValueError(f"미완료 회차: {sid}: {item.get('status')}")
        output = run_dir / f"{index:02d}-output.png"
        inp = corpus / f"{sid}-input256.png"
        orig = corpus / f"{sid}-orig1024.png"
        if item.get("output_file") != output.name or item.get("log_file") != f"{index:02d}-invocation.log":
            raise ValueError(f"출력·로그 이름 불일치: {sid}")
        for path, expected in ((output, item.get("sha256")), (inp, item.get("input_sha256")),
                               (orig, item.get("orig_sha256"))):
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"SHA256 불일치: {path}")
        if output.stat().st_size != item.get("output_bytes"):
            raise ValueError(f"출력 크기 불일치: {output}")
        if not (run_dir / item["log_file"]).is_file():
            raise ValueError(f"호출 로그 없음: {sid}")
        lanczos, apisr = score_case(orig, inp, output)
        rows.append({"img": sid, "kind": kind, "task": item["task"],
                     "lanczos": lanczos, "apisr4x": apisr,
                     "apisr_t_load_s": item.get("t_load_s"),
                     "apisr_t_infer_s": item.get("t_infer_s"),
                     "output_sha256": item["sha256"]})
    def mean(tool: str, metric: str, digits: int) -> float:
        return round(sum(row[tool][metric] for row in rows) / len(rows), digits)
    summary = {
        "model": "APISR 4x (4x_APISR_GRL_GAN_generator.pth)",
        "baseline": "PIL Lanczos 4x", "downscale_method": "PIL Lanczos 1024→256",
        "sample_n": len(rows), "infra_errors": 0,
        "sample_note": "자가 생성 일러스트 3장. 실사·사진 없음; 화질 일반화 불가.",
        "score_rule": "같은 1024px 원본 대비 RGB PSNR(data_range=255), RGB SSIM(channel_axis=2, data_range=255)",
        "avg_psnr": {tool: mean(tool, "psnr", 2) for tool in ("lanczos", "apisr4x")},
        "avg_ssim": {tool: mean(tool, "ssim", 4) for tool in ("lanczos", "apisr4x")},
        "rows": rows,
    }
    save_json(run_dir / "comparison.json", summary)
    save_json(run_dir / "aggregate.json", {
        "sample_n": summary["sample_n"], "infra_errors": summary["infra_errors"],
        "avg_psnr": summary["avg_psnr"], "avg_ssim": summary["avg_ssim"],
        "score_rule": summary["score_rule"],
    })
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "img", "kind", "lanczos_psnr",
                                                    "apisr4x_psnr", "lanczos_ssim", "apisr4x_ssim",
                                                    "apisr_t_load_s", "apisr_t_infer_s", "output_sha256"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"task": row["task"], "img": row["img"], "kind": row["kind"],
                             "lanczos_psnr": row["lanczos"]["psnr"],
                             "apisr4x_psnr": row["apisr4x"]["psnr"],
                             "lanczos_ssim": row["lanczos"]["ssim"],
                             "apisr4x_ssim": row["apisr4x"]["ssim"],
                             "apisr_t_load_s": row["apisr_t_load_s"],
                             "apisr_t_infer_s": row["apisr_t_infer_s"],
                             "output_sha256": row["output_sha256"]})
    prior_path = run_dir / "run.yaml"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    record = {
        "tool": "upscale", "date": prior.get("date") or dt.date.today().isoformat(), "method": "UPS-01",
        "access": "local", "model": "apisr4x", "generated_by": Path(__file__).name,
        "harness_version": "1.0", "tos_confirmed": True,
        "tos_source_url": "로컬 오픈 업스케일러(APISR GAN·spandrel 자체 구동)",
        "model_file": MODEL.name, "model_sha256": digest(MODEL) if MODEL.is_file() else None,
        "score_rule": summary["score_rule"], "score_file": "comparison.json",
        "condition_changes": [], "rescore": {"count": len(rows), "rule_changes": 0},
        "runs": [{"task": item["task"], "input": str(corpus / f"{item['img']}-input256.png"),
                  "output_file": item["output_file"], "screenshot": item["output_file"],
                  "log_file": item["log_file"], "elapsed_s": item["elapsed_s"],
                  "output_sha256": item["sha256"]} for item in recorded],
        "compare": [], "disclosure": True,
    }
    save_json(run_dir / "run.yaml", record)
    return summary


def check_fresh_run(run_dir: Path) -> None:
    """촬영기가 미리 만든 live/만 허용하고 과거 모델 출력은 보호한다."""
    if run_dir.exists() and any(path.name != "live" for path in run_dir.iterdir()):
        raise ValueError(f"런 폴더가 비어 있지 않음(기존 증거 덮기 금지): {run_dir}")


def run_benchmark(run_dir: Path, corpus: Path = CORPUS) -> int:
    run_dir, corpus = Path(run_dir), Path(corpus)
    check_fresh_run(run_dir)
    if not (PYTHON.is_file() and WORKER.is_file() and MODEL.is_file()):
        raise FileNotFoundError(f"워커·Python·체크포인트 확인 필요: {PYTHON}, {WORKER}, {MODEL}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (sid, _kind) in enumerate(CASES, 1):
        inp, orig = (corpus / f"{sid}-input256.png", corpus / f"{sid}-orig1024.png")
        if not inp.is_file() or not orig.is_file():
            raise FileNotFoundError(f"입력 원본 누락: {sid}")
        mark("turn", f"{sid} 자가 생성 일러스트: 256px 입력을 APISR 4x로 복원, 원본 1024px과 비교")
        output = run_dir / f"{index:02d}-output.png"
        log = run_dir / f"{index:02d}-invocation.log"
        command = [str(PYTHON), str(WORKER), str(inp), "apisr4x", str(output)]
        started = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
        elapsed = round(time.monotonic() - started, 3)
        log.write_text(json.dumps({"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                                   "command": command, "returncode": proc.returncode,
                                   "stdout": proc.stdout, "stderr": proc.stderr}, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        if proc.returncode != 0 or not output.is_file():
            mark("infra", f"{sid} 워커 실패 rc={proc.returncode}; 로그 {log.name}")
            raise RuntimeError(f"{sid} 워커 실패: {proc.stderr[-300:]}")
        meta_lines = [line[5:] for line in proc.stdout.splitlines() if line.startswith("META:")]
        if len(meta_lines) != 1:
            mark("infra", f"{sid} META 레코드 누락; 로그 {log.name}")
            raise ValueError(f"META 레코드 1개 필요: {sid}")
        meta = json.loads(meta_lines[0])
        if meta.get("method") != "apisr4x" or meta.get("device") != "cuda":
            mark("infra", f"{sid} 모델·장치가 계획과 다름")
            raise ValueError(f"모델·장치 불일치: {meta}")
        rows.append({"task": f"UPS-{index:02d}", "img": sid, "status": "ok",
                     "output_file": output.name, "log_file": log.name, "sha256": digest(output),
                     "output_bytes": output.stat().st_size, "input_sha256": digest(inp),
                     "orig_sha256": digest(orig), "elapsed_s": elapsed,
                     "t_load_s": meta.get("t_load_s"), "t_infer_s": meta.get("t_infer_s"),
                     "device": meta.get("device"), "model_file": meta.get("model_file")})
        save_json(run_dir / "results.json", rows)
        lanczos, apisr = score_case(orig, inp, output)
        if apisr["psnr"] < lanczos["psnr"] and apisr["ssim"] < lanczos["ssim"]:
            mark("break", f"{sid} 두 지표 모두 Lanczos보다 낮음: PSNR {lanczos['psnr']}→{apisr['psnr']}, SSIM {lanczos['ssim']}→{apisr['ssim']}")
    summary = rescore_run(run_dir, corpus)
    mark("agg", f"전건 {summary['sample_n']}/3·infra {summary['infra_errors']}; PSNR Lanczos {summary['avg_psnr']['lanczos']} / APISR {summary['avg_psnr']['apisr4x']}; SSIM Lanczos {summary['avg_ssim']['lanczos']} / APISR {summary['avg_ssim']['apisr4x']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args(argv)
    if args.rescore:
        result = rescore_run(args.run_dir, args.corpus)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return run_benchmark(args.run_dir, args.corpus)


if __name__ == "__main__":
    sys.exit(main())
