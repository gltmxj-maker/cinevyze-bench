#!/usr/bin/env python3
"""32번: 7월 입력 3장을 rembg u2net CPU로 재실행하고 시각 대조 시트를 남긴다."""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiment_rebuild_common import ROOT, base_yaml, digest, fresh_run, save_json, save_results, save_yaml
from live_mark import mark


INPUTS = [
    {"task": "BG-01", "path": str(Path(os.environ.get("REMBG_INPUT_DIR", str(ROOT / "corpus/rembg-rebuild"))) / "char-hair.png"), "kind": "hair"},
    {"task": "BG-02", "path": str(Path(os.environ.get("REMBG_INPUT_DIR", str(ROOT / "corpus/rembg-rebuild"))) / "creature.png"), "kind": "simple"},
    {"task": "BG-03", "path": str(Path(os.environ.get("REMBG_INPUT_DIR", str(ROOT / "corpus/rembg-rebuild"))) / "gen-mascot.png"), "kind": "mascot"},
]
DEFAULT_RUN = ROOT / "test_runs/rembg-u2net-20260924"
MODEL_HOME = Path(os.environ.get("REMBG_MODEL_DIR", "./models/rembg"))


def summarize_alpha(values: list[int]) -> dict:
    n = len(values)
    if not n:
        raise ValueError("빈 알파 채널")
    return {"quality_claim": "descriptive_only", "pixels": n,
            "transparent_ratio": round(sum(v == 0 for v in values) / n, 4),
            "opaque_ratio": round(sum(v == 255 for v in values) / n, 4),
            "soft_edge_ratio": round(sum(0 < v < 255 for v in values) / n, 4)}


def comparison_sheet(input_path: Path, result_path: Path, target: Path) -> None:
    """원본·투명 PNG를 흰/체커보드 배경에 같은 크기로 배열한 관찰용 시트."""
    with Image.open(input_path) as raw, Image.open(result_path) as cut:
        original = raw.convert("RGB")
        rgba = cut.convert("RGBA")
        width = 384
        height = round(original.height * width / original.width)
        original = original.resize((width, height), Image.Resampling.LANCZOS)
        rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
        checker = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(checker)
        step = 32
        for y in range(0, height, step):
            for x in range(0, width, step):
                if (x // step + y // step) % 2:
                    draw.rectangle((x, y, x + step - 1, y + step - 1), fill="#d9d9d9")
        pasted = Image.alpha_composite(checker.convert("RGBA"), rgba).convert("RGB")
        sheet = Image.new("RGB", (width * 2 + 24, height + 48), "#ffffff")
        sheet.paste(original, (8, 40))
        sheet.paste(pasted, (width + 16, 40))
        heading = ImageDraw.Draw(sheet)
        heading.text((8, 10), "SOURCE", fill="#000000")
        heading.text((width + 16, 10), "REMBG / CHECKERBOARD", fill="#000000")
        sheet.save(target)


def run_live(run: Path) -> None:
    fresh_run(run)
    if not (MODEL_HOME / "u2net.onnx").is_file():
        raise FileNotFoundError(MODEL_HOME / "u2net.onnx")
    os.environ["U2NET_HOME"] = str(MODEL_HOME)
    from rembg import new_session, remove

    meta = base_yaml(Path(__file__).name, "BG-01 + BG-02 + BG-03", "u2net")
    meta.update({"model_file_sha256": digest((MODEL_HOME / "u2net.onnx").read_bytes()),
                 "score_rules": "알파 채널 분포는 기술 통계만; 품질 점수는 정답 마스크 없어 산출하지 않음",
                 "condition_changes": [], "input_count": len(INPUTS)})
    rows = []
    audit = []
    mark("turn", "rembg u2net CPU — 7월 입력 3장 고정·원본/투명 결과 시각 대조")
    session = new_session("u2net")
    for index, case in enumerate(INPUTS, 1):
        path = Path(case["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            original = img.convert("RGB")
        started = time.monotonic()
        result = remove(original, session=session)
        elapsed = round(time.monotonic() - started, 3)
        output = run / f"{index:02d}-output.png"
        result.save(output)
        alpha = list(result.getchannel("A").getdata())
        measures = summarize_alpha(alpha)
        comparison = run / f"{index:02d}-comparison.png"
        comparison_sheet(path, output, comparison)
        row = {"task": case["task"], "kind": case["kind"], "input": case["path"],
               "input_sha256": digest(path.read_bytes()), "output_file": output.name,
               "output_sha256": digest(output.read_bytes()), "comparison_file": comparison.name,
               "comparison_sha256": digest(comparison.read_bytes()),
               "log_file": f"{index:02d}-invocation.log", "elapsed_s": elapsed,
               "width": result.width, "height": result.height, **measures}
        (run / row["log_file"]).write_text(json.dumps({"writer": Path(__file__).name,
              "input": case, "row": row, "u2net_model_sha256": meta["model_file_sha256"]},
              ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        meta["runs"].append({"task": case["task"], "input": case["path"],
                             "output_file": output.name, "log_file": row["log_file"],
                             "elapsed_s": elapsed, "input_sha256": row["input_sha256"],
                             "output_sha256": row["output_sha256"]})
        save_yaml(run, meta)
        audit.append({"task": case["task"], "input": case["path"], "output": output.name,
                      "comparison": comparison.name, "observation": "시각 대조 필요",
                      "ground_truth_mask": None, "quality_score": None,
                      "alpha_distribution": measures})
        print(f"[{index}/3] {case['task']} · {elapsed}s · 투명 {measures['transparent_ratio']:.1%} · 비교 {comparison.name}", flush=True)
        if index == 1:
            mark("turn", "첫 이미지 처리 후 재사용 세션 — 이후 2장 처리시간은 모델 로드 제외")
    save_json(run / "visual_audit.json", audit)
    aggregate = {"sample_n": len(rows), "model": "u2net", "infra_errors": 0,
                 "median_elapsed_s": statistics.median(r["elapsed_s"] for r in rows),
                 "quality_score": None, "ground_truth_mask": None,
                 "comparison_files": [r["comparison_file"] for r in rows]}
    save_results(run, rows, aggregate)
    mark("agg", f"rembg u2net 3장 완료 — 처리시간 중앙 {aggregate['median_elapsed_s']:.3f}초 · 비교 시트 3장 · 품질 점수 없음")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--verify-input", action="store_true",
                        help="7월 출력 PNG와 이번 출력 PNG 세 장을 바이트 대조해 이동된 입력 동일성 확인")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        import yaml
        meta = yaml.safe_load((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
        rows = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
        if len(rows) != 3 or len(meta["runs"]) != 3:
            raise ValueError("입력 3장 전건 기록 없음")
        for row, record in zip(rows, meta["runs"]):
            old_input = ROOT / row["input"]
            alternate = ROOT / INPUTS[int(row["task"][-2:]) - 1]["path"]
            if not old_input.is_file() and digest(alternate.read_bytes()) != row["input_sha256"]:
                raise ValueError(f"이동 입력 SHA 불일치: {alternate}")
            for path, expected in (((old_input if old_input.is_file() else alternate), row["input_sha256"]),
                                   (args.run_dir / row["output_file"], row["output_sha256"]),
                                   (args.run_dir / row["comparison_file"], row["comparison_sha256"])):
                if digest(path.read_bytes()) != expected:
                    raise ValueError(f"SHA 불일치: {path}")
            if record["output_sha256"] != row["output_sha256"]:
                raise ValueError(f"run.yaml 연결 불일치: {row['task']}")
            logged = json.loads((args.run_dir / row["log_file"]).read_text(encoding="utf-8"))
            if logged["row"] != row:
                raise ValueError(f"호출 로그 불일치: {row['task']}")
        print("3/3 입력·출력·시각 대조 SHA 및 호출 로그 PASS")
    elif args.verify_input:
        for index, case in enumerate(INPUTS, 1):
            old = ROOT / "test_runs/rembg-20260630" / f"{index:02d}-output.png"
            new = args.run_dir / f"{index:02d}-output.png"
            if digest(old.read_bytes()) != digest(new.read_bytes()):
                raise ValueError(f"7월 결과와 새 결과 바이트 불일치: {case['task']}")
        print("3/3 결과 PNG SHA256 동일; 같은 u2net 체크포인트·입력으로 재현된 결과")
    else:
        run_live(args.run_dir)


if __name__ == "__main__":
    main()
