#!/usr/bin/env python3
"""26번: 고정 한국어 낭독 6클립을 Whisper large-v3 CPU int8로 재측정한다.

실제 모델 응답에서 전사·호출 로그·run.yaml을 기록한다. --rescore는 해시를 확인한
저장 출력 전건을 같은 CER/WER 규칙으로 다시 채점한다.
"""
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
from pathlib import Path

import yaml

from live_mark import mark
from stt_score import _selftest, cer, wer


ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("BENCH_CORPUS_DIR", ROOT / "corpus")) / "stt"
DEFAULT_RUN = ROOT / "test_runs" / f"whisper-large-v3-{dt.date.today():%Y%m%d}"
MODEL = "large-v3"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _inputs(corpus: Path) -> list[tuple[Path, dict]]:
    refs = json.loads((corpus / "refs.json").read_text(encoding="utf-8"))
    expected = [f"clip-{index:02d}.wav" for index in range(1, 7)]
    if list(refs) != expected:
        raise ValueError(f"음성 입력 6건 순서·개수 불일치: {list(refs)}")
    items = []
    for index, name in enumerate(expected, 1):
        source = corpus / name
        ref = refs[name]
        if not source.is_file() or ref.get("id") != f"STT-{index:02d}" or not ref.get("ref"):
            raise ValueError(f"입력 또는 정답 누락: {name}")
        items.append((source, ref))
    return items


def _score(run_dir: Path, corpus: Path, *, emit_marks: bool, emit_break: bool = True) -> dict:
    _selftest()
    metadata = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    stored = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    inputs = _inputs(corpus)
    if metadata.get("model") != MODEL or len(metadata.get("runs") or []) != 6 or len(stored) != 6:
        raise ValueError("모델·회차 수 불일치")
    rows = []
    first_break = False
    for index, ((source, ref), item, saved) in enumerate(zip(inputs, metadata["runs"], stored), 1):
        output = run_dir / f"{index:02d}-output.txt"
        log = run_dir / f"{index:02d}-invocation.log"
        if (item.get("task") != ref["id"] or Path(item.get("input", "")).name != source.name
                or item.get("output_file") != output.name or item.get("log_file") != log.name
                or saved.get("task") != ref["id"] or saved.get("clip") != source.name
                or saved.get("output_file") != output.name or saved.get("log_file") != log.name):
            raise ValueError(f"회차 연결 불일치: {index}")
        if (_digest(source) != saved.get("input_sha256")
                or _digest(output) != saved.get("output_sha256")):
            raise ValueError(f"SHA256 불일치: {source.name} / {output.name}")
        raw_log = log.read_text(encoding="utf-8")
        hyp = output.read_text(encoding="utf-8").strip()
        if (f"task: {ref['id']}" not in raw_log or f"model: {MODEL}" not in raw_log
                or raw_log.split("--- RESPONSE ---\n", 1)[-1].strip() != hyp):
            raise ValueError(f"원출력·호출 로그 불일치: {log.name}")
        if metadata.get("generated_by") == "run_whisper_bench.py":
            required = ("device: cpu", "compute_type: int8", "beam_size: 5", "language: ko",
                        f"input_sha256: {saved['input_sha256']}",
                        f"output_sha256: {saved['output_sha256']}")
            if any(field not in raw_log for field in required):
                raise ValueError(f"실행 조건·해시 로그 불일치: {log.name}")
            timing = {key: value for key, value in
                      (line.split(": ", 1) for line in raw_log.split("--- PROMPT ---", 1)[0].splitlines()
                       if ": " in line)}
            for field in ("audio_s", "infer_s", "elapsed_s"):
                if float(timing.get(field, "nan")) != saved.get(field):
                    raise ValueError(f"실행 시간 로그 불일치: {log.name}/{field}")
            if float(item.get("elapsed_s", -1)) != saved.get("elapsed_s"):
                raise ValueError(f"실행 시간 run.yaml 불일치: {item['task']}")
        score_cer, score_wer = cer(ref["ref"], hyp), wer(ref["ref"], hyp)
        if emit_marks and emit_break and not first_break and score_cer > 0:
            first_break = True
            mark("break", f"첫 전사 불일치 — {ref['id']} 깨끗한 한국어 낭독; CER {score_cer:.1%}, WER {score_wer:.1%}")
        rows.append({**saved, "cer": round(score_cer, 6), "wer": round(score_wer, 6),
                     "ref": ref["ref"], "hyp": hyp})
    summary = {"model": MODEL, "sample_n": len(rows), "infra_errors": 0,
               "mean_cer": round(statistics.mean(row["cer"] for row in rows), 6),
               "median_cer": round(statistics.median(row["cer"] for row in rows), 6),
               "mean_wer": round(statistics.mean(row["wer"] for row in rows), 6),
               "total_audio_s": round(sum(row.get("audio_s") or 0 for row in rows), 2),
               "total_infer_s": round(sum(row.get("infer_s") or 0 for row in rows), 2)}
    warm = rows[1:]
    audio = sum(row.get("audio_s") or 0 for row in warm)
    summary["rtf_warm"] = round(sum(row.get("infer_s") or 0 for row in warm) / audio, 3) if audio else None
    _json(run_dir / "results.json", rows)
    _json(run_dir / "aggregate.json", summary)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "clip", "cer", "wer", "audio_s", "infer_s", "elapsed_s"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    if emit_marks:
        mark("agg", f"Whisper large-v3 CPU 집계 — {len(rows)}클립 · 평균 CER {summary['mean_cer']:.1%} · 평균 WER {summary['mean_wer']:.1%} · 예열 제외 RTF {summary['rtf_warm']}")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def rescore_run(run_dir: Path, corpus: Path = CORPUS) -> dict:
    """저장된 음성·전사·호출 로그를 연결 검증하고 여섯 건 전부 재채점."""
    return _score(Path(run_dir), Path(corpus), emit_marks=True)


def run_benchmark(run_dir: Path, corpus: Path = CORPUS) -> dict:
    from faster_whisper import WhisperModel, __version__ as faster_whisper_version

    run_dir, corpus = Path(run_dir), Path(corpus)
    if run_dir.exists() and any(p.name != "live" for p in run_dir.iterdir()):
        raise FileExistsError(f"기존 런 덮어쓰기 금지: {run_dir}")
    inputs = _inputs(corpus)
    run_dir.mkdir(parents=True, exist_ok=True)
    _selftest()
    started_load = time.monotonic()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", local_files_only=True)
    load_s = round(time.monotonic() - started_load, 2)
    rows, runs = [], []
    first_break = False
    for index, (source, ref) in enumerate(inputs, 1):
        started = time.monotonic()
        try:
            segments, info = model.transcribe(str(source), language="ko", beam_size=5)
            hyp = "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            mark("infra", f"{ref['id']} Whisper 실행 실패 — {type(exc).__name__}: {str(exc)[:90]}")
            raise
        infer_s = round(time.monotonic() - started, 2)
        elapsed_s = round(infer_s + (load_s if index == 1 else 0), 2)
        audio_s = round(info.duration, 2)
        output = run_dir / f"{index:02d}-output.txt"
        log = run_dir / f"{index:02d}-invocation.log"
        output.write_text(hyp, encoding="utf-8")
        log.write_text(
            f"# run_whisper_bench.py invocation (write-origin)\n"
            f"timestamp: {dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"tool: whisper\naccess: local\nmodel: {MODEL}\ntask: {ref['id']}\n"
            f"elapsed_s: {elapsed_s}\naudio_s: {audio_s}\ninfer_s: {infer_s}\n"
            f"load_s: {load_s if index == 1 else 0}\ndevice: cpu\ncompute_type: int8\n"
            f"beam_size: 5\nlanguage: ko\ninput_sha256: {_digest(source)}\n"
            f"output_sha256: {_digest(output)}\nfaster_whisper: {faster_whisper_version}\n"
            f"--- PROMPT ---\n{source}\n--- RESPONSE ---\n{hyp}\n", encoding="utf-8")
        row = {"task": ref["id"], "clip": source.name, "input_sha256": _digest(source),
               "output_sha256": _digest(output), "output_file": output.name, "log_file": log.name,
               "audio_s": audio_s, "infer_s": infer_s, "elapsed_s": elapsed_s}
        rows.append(row)
        runs.append({"task": ref["id"], "input": str(source), "output_file": output.name,
                     "log_file": log.name, "elapsed_s": elapsed_s, "notes": "CPU int8·ko·beam5"})
        score_cer, score_wer = cer(ref["ref"], hyp), wer(ref["ref"], hyp)
        print(f"[{index}/6] {ref['id']} CER {score_cer:.1%} WER {score_wer:.1%} infer {infer_s:.2f}s", flush=True)
        if not first_break and score_cer > 0:
            first_break = True
            mark("break", f"첫 전사 불일치 — {ref['id']} 깨끗한 한국어 낭독; CER {score_cer:.1%}, WER {score_wer:.1%}")
    payload = {"tool": "whisper", "date": dt.date.today().isoformat(), "method": "STT-01",
               "access": "local", "model": MODEL, "generated_by": "run_whisper_bench.py",
               "harness_version": "1.0", "tos_confirmed": True,
               "tos_source_url": "로컬 오픈 음성인식(faster-whisper·CTranslate2·CPU·오프라인·구독/계정 무관)",
               "environment": {"python": platform.python_version(), "faster_whisper": faster_whisper_version},
               "scoring": "stt_score.py CER/WER; NFKC+lower; punctuation removed; warm RTF excludes first clip",
               "condition_changes": [], "runs": runs, "compare": [], "disclosure": True}
    (run_dir / "run.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _json(run_dir / "results.json", rows)
    return _score(run_dir, corpus, emit_marks=True, emit_break=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    rescore_run(args.run_dir, args.corpus) if args.rescore else run_benchmark(args.run_dir, args.corpus)


if __name__ == "__main__":
    main()
