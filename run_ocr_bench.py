#!/usr/bin/env python3
"""25번: 같은 한국어 문단의 화질 6조건을 Tesseract kor로 재측정한다.

실행 결과만 run.yaml, 로그, 점수, 실촬영 마커로 기록한다. 입력은 자체 생성
corpus/ocr/clip-*.png와 refs.json이며 OCR 설정은 --psm 6 --oem 1로 고정한다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import yaml

from live_mark import mark
from ocr_score import cer, wer, _selftest


ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("BENCH_CORPUS_DIR", ROOT / "corpus")) / "ocr"
DEFAULT_RUN = ROOT / "test_runs" / f"tesseract-kor-{dt.date.today():%Y%m%d}"
CONDITIONS = ("clean", "small", "lowres", "blur", "noise", "rotate")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark(run_dir: Path, corpus: Path = CORPUS) -> dict:
    run_dir, corpus = Path(run_dir), Path(corpus)
    if (run_dir / "run.yaml").exists():
        raise FileExistsError(f"기존 런 덮어쓰기 금지: {run_dir / 'run.yaml'}")
    refs_path = corpus / "refs.json"
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    expected = [f"clip-{i:02d}.png" for i in range(1, 7)]
    if list(refs) != expected:
        raise ValueError(f"OCR 입력 6종 순서·개수 불일치: {list(refs)}")
    for name, condition in zip(expected, CONDITIONS):
        if refs[name].get("condition") != condition or not (corpus / name).is_file():
            raise ValueError(f"OCR 조건 또는 입력 누락: {name}/{condition}")

    _selftest()
    run_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    rows, runs = [], []
    first_break = False
    for index, name in enumerate(expected, 1):
        item = refs[name]
        image = corpus / name
        if index == 2:
            mark("turn", "같은 참조 문단을 깨끗한 인쇄체에서 작은 글씨·저해상도·흐림·노이즈·기울기로 전환")
        command = ["tesseract", str(image), "stdout", "-l", "kor", "--psm", "6", "--oem", "1"]
        started = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
        elapsed = round(time.monotonic() - started, 3)
        if proc.returncode != 0:
            mark("infra", f"{item['id']} Tesseract 실패 rc={proc.returncode}: {proc.stderr.strip()[:100]}")
            raise RuntimeError(f"Tesseract 실패 {name}: {proc.stderr.strip()[:200]}")
        transcript = proc.stdout.strip()
        output = f"{index:02d}-output.txt"
        log = f"{index:02d}-invocation.log"
        (run_dir / output).write_text(transcript, encoding="utf-8")
        logged_at = dt.datetime.now().isoformat(timespec="seconds")
        (run_dir / log).write_text(
            f"# run_ocr_bench.py invocation (write-origin)\n"
            f"timestamp: {logged_at}\ntool: tesseract\nmodel: kor\ntask: {item['id']}\n"
            f"condition: {item['condition']}\nlang: kor\npsm: 6\noem: 1\n"
            f"elapsed_s: {elapsed}\ntess: {version}\ninput_sha256: {_digest(image)}\n"
            f"--- COMMAND ---\n{' '.join(command)}\n--- RESPONSE ---\n{proc.stdout}\n",
            encoding="utf-8",
        )
        score_cer, score_wer = cer(item["ref"], transcript), wer(item["ref"], transcript)
        row = {"task": item["id"], "clip": name, "condition": item["condition"],
               "desc": item["desc"], "cer": round(score_cer, 6), "wer": round(score_wer, 6),
               "elapsed_s": elapsed, "input_sha256": _digest(image),
               "output_sha256": _digest(run_dir / output), "output_file": output, "log_file": log,
               "ref": item["ref"], "hyp": transcript}
        rows.append(row)
        runs.append({"task": item["id"], "input": str(image), "output_file": output,
                     "log_file": log, "elapsed_s": elapsed, "notes": item["condition"]})
        print(f"[{index}/6] {name} {item['condition']} CER {score_cer:.1%} WER {score_wer:.1%} {elapsed:.3f}s", flush=True)
        if not first_break and score_cer > 0:
            first_break = True
            mark("break", f"첫 전사 불일치 — {item['id']} {item['desc']}; CER {score_cer:.1%}, WER {score_wer:.1%}")

    summary = {"sample_n": len(rows), "infra_errors": 0, "model": "kor",
               "mean_cer": round(statistics.mean(r["cer"] for r in rows), 6),
               "mean_wer": round(statistics.mean(r["wer"] for r in rows), 6),
               "per_condition": {r["condition"]: {"cer": r["cer"], "wer": r["wer"]} for r in rows}}
    _json(run_dir / "results.json", rows)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "clip", "condition", "cer", "wer", "elapsed_s"))
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    _json(run_dir / "aggregate.json", summary)
    metadata = {"tool": "tesseract", "date": dt.date.today().isoformat(), "method": "OCR-01",
                "access": "local", "model": "kor", "generated_by": "run_ocr_bench.py",
                "tos_confirmed": True, "tos_source_url": "로컬 오픈 OCR(Tesseract·CPU·오프라인)",
                "corpus_refs_sha256": _digest(refs_path), "tesseract_version": version,
                "scoring": "ocr_score.py cer/wer; NFKC, 소문자, 구두점 제거, Levenshtein 거리/정답 길이",
                "condition_changes": [], "runs": runs, "compare": [], "disclosure": True}
    (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    mark("agg", f"6/6조건·infra 0; 평균 CER {summary['mean_cer']:.1%}, 평균 WER {summary['mean_wer']:.1%}; kor·psm6·oem1")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.run_dir, args.corpus), ensure_ascii=False))


if __name__ == "__main__":
    main()
