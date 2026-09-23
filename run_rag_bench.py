#!/usr/bin/env python3
"""로컬 RAG 재측정 하네스(2026-09-23) — 내 문서 4개 → 추출 → 청크 → bge-m3 검색 → gemma3:4b 답변.

`rag_bench.py`(2026-07-12 첫 런의 write-origin)는 그 런의 증거라 건드리지 않는다. 이 하네스는
새 런 폴더에만 쓴다. 달라진 것:
  - 정답 판정을 사람 라벨 대신 `rag_score.py`(결정론)가 한다. 문항·정답 패턴 = `rag_bench_cases.json`.
  - 문항 12 → 38(답 있음 32 · 함정 6). 검색 조각 수 두 팔(top_k 3·6) × 반복 3회.
  - 질문 임베딩을 생성 전에 한 번에 끝낸다 → 생성 동안 GPU 에는 gemma3:4b 하나만 올라간다.
  - num_ctx 를 8192 로 고정하고 prompt_eval_count 를 남긴다(6조각 프롬프트가 잘리는지 확인).
  - 온도·seed 는 요청에 넣지 않는다(모델 기본 설정 그대로 · 반복 3회가 흔들림을 본다).
  - 실촬영 마커: 추출 실패·첫 오답·함정 첫 지어냄 = break · 팔/반복 전환 = turn · 집계 = agg.

usage:
  python3 run_rag_bench.py --mode pilot --run-dir test_runs/local-rag-20260923
  python3 run_rag_bench.py --mode full  --run-dir test_runs/local-rag-20260923
  python3 run_rag_bench.py --rescore     --run-dir test_runs/local-rag-20260923   # 채점기 수리 후 재채점(GPU 불필요)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from live_mark import mark as live_mark
from rag_score import score_answer, score_run_dir

HARNESS_VERSION = "1.0"
BASE = Path(__file__).resolve().parent
DEFAULT_CASES = BASE / "rag_bench_cases.json"
DEFAULT_RUN_DIR = BASE / "test_runs" / "local-rag-20260923"
EMBED_MODEL = "bge-m3"
CHUNK_CHARS = 700
CHUNK_OVERLAP = 150
ARMS = (3, 6)
REPS = 3
NUM_CTX = 8192
PILOT_IDS = ("Q01", "Q04", "Q07", "Q08", "Q11", "Q12", "Q19", "Q24", "Q31", "Q35")

PROMPT = """아래 [문서]만 근거로 질문에 답하세요. 문서에 답이 없으면 반드시 "문서에 없음"이라고만 답하세요.
추측하거나 아는 지식으로 채우지 마세요.

[문서]
{context}

[질문] {question}
[답변]"""


# ── 추출 ────────────────────────────────────────────────
def extract_text(path: Path, tmp_dir: Path) -> tuple[str, str]:
    """문서 → (텍스트, 도구). 실패해도 예외를 내지 않는다 — 추출 실패도 결과다."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, timeout=120)
            return out.stdout.decode("utf-8", "ignore"), "pdftotext -layout"
        if ext == ".hwpx":
            tmp_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["libreoffice", "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                            "--outdir", str(tmp_dir), str(path)], capture_output=True, timeout=240)
            txt = tmp_dir / (path.stem + ".txt")
            if txt.exists() and txt.read_text(encoding="utf-8", errors="ignore").strip():
                return txt.read_text(encoding="utf-8", errors="ignore"), "libreoffice --convert-to txt"
            z = zipfile.ZipFile(path)
            runs = []
            for name in [n for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n)]:
                for el in ET.fromstring(z.read(name)).iter():
                    if el.tag.endswith("}t") and el.text:
                        runs.append(el.text)
            if runs:
                return " ".join(runs), "직접 파싱(zip+xml) — libreoffice 변환 실패 후 우회"
            return "", "libreoffice(변환 실패)·zip 파싱 0"
    except Exception as exc:  # noqa: BLE001 — 추출 실패도 데이터
        return "", f"{ext} 실패: {type(exc).__name__}"
    return "", f"{ext} 미지원"


def chunk(text: str) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return [c for c in out if len(c.strip()) > 50]


def build_corpus(corpus_dir: Path, run_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    report, chunks, sources = [], [], []
    for path in sorted(Path(p) for p in glob.glob(str(corpus_dir / "*")) if os.path.isfile(p)):
        t0 = time.perf_counter()
        text, tool = extract_text(path, run_dir / "tmp_convert")
        report.append({
            "file": path.name,
            "size_kb": round(path.stat().st_size / 1024),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tool": tool,
            "chars": len(text.strip()),
            "korean_chars": sum(1 for c in text if "가" <= c <= "힣"),
            "extract_s": round(time.perf_counter() - t0, 2),
        })
        for c in chunk(text):
            chunks.append(c)
            sources.append(path.name)
        report[-1]["chunks"] = sum(1 for s in sources if s == path.name)
    return report, chunks, sources


# ── ollama ──────────────────────────────────────────────
def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(base: str, texts: list[str], timeout: int) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 64):
        r = api_json(base + "/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + 64], "keep_alive": "2m"},
                     timeout)
        vecs.extend(r["embeddings"])
    v = np.array(vecs, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def unload(base: str, model: str, timeout: int) -> None:
    try:
        api_json(base + "/api/generate", {"model": model, "keep_alive": 0}, timeout)
    except Exception:  # noqa: BLE001
        pass


def generate(base: str, model: str, prompt: str, timeout: int):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_ctx": NUM_CTX}}
    attempts = []
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            resp = api_json(base + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": None})
            return resp, attempts, None
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": err})
            if attempt == 2:
                return None, attempts, err
            time.sleep(1)
    return None, attempts, attempts[-1]["error"]


# ── 계획 ────────────────────────────────────────────────
def planned(questions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """pilot = 반복1 · 10문항 · 두 팔. full = 반복 1..3 · 38문항 · 두 팔(파일럿 포함)."""
    runs = []
    reps = (1,) if mode == "pilot" else tuple(range(1, REPS + 1))
    for rep in reps:
        for k in ARMS:
            for q in questions:
                if mode == "pilot" and q["id"] not in PILOT_IDS:
                    continue
                runs.append({"key": f"r{rep}/k{k}/{q['id']}", "rep": rep, "top_k": k, "q": q})
    return runs


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        keys.add(row["key"])
    # ★번호는 raw 의 모든 파일에서 복원한다 — 원출력만 쓰고 invocation 전에 죽은 회차를 덮어쓰지 않게
    for p in (run_dir / "raw").glob("*-*"):
        m = re.match(r"(\d+)-", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return keys, maximum


def _seen_breaks(run_dir: Path, qmap: dict[str, dict[str, Any]]) -> set[str]:
    """재개 시 이미 터진 break 종류를 복원한다(같은 종류 마커 재발화 방지)."""
    seen = set()
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("infra_error"):
            continue
        ans = (run_dir / row["response_file"]).read_text(encoding="utf-8")
        v = score_answer(qmap[row["qid"]], ans, [c["text"] for c in row["retrieved"]])["verdict"]
        if v in ("wrong", "hallucinated"):
            seen.add(v)
    return seen


def _clip(text: str, limit: int = 185) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_yaml(run_dir: Path, model: str, meta: dict[str, Any], corpus: list[dict[str, Any]],
                   index_info: dict[str, Any]) -> None:
    entries = []
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        entries.append({"key": row["key"], "qid": row["qid"], "top_k": row["top_k"], "rep": row["rep"],
                        "mode": row["mode"], "output_file": row["transcript_file"],
                        "response_file": row["response_file"], "log_file": str(p.relative_to(run_dir)),
                        "gen_s": row.get("gen_s"), "infra_error": row.get("infra_error")})
    agg = {}
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8")).get("arms", {})
    compare = []
    for arm, block in agg.items():
        compare.append({"metric": "answerable_correct", "arm": arm, **block.get("answerable_correct", {})})
        compare.append({"metric": "trap_refused", "arm": arm, **block.get("trap_refused", {})})
    payload = {
        "tool": "ollama (bge-m3 임베딩 + gemma3:4b 생성) + pdftotext + 자작 RAG 파이프라인",
        "date": dt.date.today().isoformat(),
        "method": "RAG-02",
        "access": "local",
        "model": model,
        "embed_model": EMBED_MODEL,
        "generated_by": "run_rag_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관) · 코퍼스=공공기관 공개문서",
        "prompt": PROMPT,
        "chunk_chars": CHUNK_CHARS, "chunk_overlap": CHUNK_OVERLAP, "arms_top_k": list(ARMS), "reps": REPS,
        "options": {"num_ctx": NUM_CTX, "temperature": "미지정(모델 기본값)", "seed": "미지정"},
        "request_parallelism": 1,
        "corpus": corpus,
        "index": index_info,
        "compare": compare,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "git_sha_at_run": _git_sha(), "ollama_version": meta.get("ollama_version"),
                        "hardware": meta.get("hardware")},
        "model_metadata": meta.get("model_metadata"),
        "runs": entries,
    }
    tmp = run_dir / "run.yaml.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run.yaml")


def _gpu_name() -> str | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


# ── 본체 ────────────────────────────────────────────────
def rescore(args: argparse.Namespace) -> int:
    """raw/ 원출력은 그대로 두고 채점만 다시 한다 — aggregate·results·run.yaml compare 를 코드가 다시 쓴다."""
    prev = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
    meta = {"ollama_version": prev["environment"].get("ollama_version"),
            "hardware": prev["environment"].get("hardware"), "model_metadata": prev.get("model_metadata")}
    _, agg = score_run_dir(args.run_dir, args.cases, NUM_CTX, write_pilot=False)
    write_run_yaml(args.run_dir, prev["model"], meta, prev["corpus"], prev["index"])
    runs = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
    # ★원 실행 기록은 보존한다 — 재채점이 바꾸는 것은 판정(compare·runs)뿐이다(T2 적대검증 2026-09-23).
    for key in ("date", "prompt", "chunk_chars", "chunk_overlap", "options", "corpus", "index",
                "environment", "model_metadata", "harness_version"):
        if key in prev:
            runs[key] = prev[key]
    runs["rescored_at"] = dt.datetime.now().isoformat(timespec="seconds")
    runs["rescored_git_sha"] = _git_sha()
    (args.run_dir / "run.yaml").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["answerable_correct"] for k, v in agg["arms"].items()}, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.rescore:
        return rescore(args)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    questions = cases["questions"]
    qmap = {q["id"]: q for q in questions}
    plan = planned(questions, args.mode)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": plan[0]["key"]}, ensure_ascii=False))
        return 0
    run_dir: Path = args.run_dir
    if args.mode == "full":
        dp = run_dir / "pilot_decision.json"
        if not dp.exists():
            print("파일럿 판정 없음 — 같은 --run-dir 에서 --mode pilot 을 먼저 완료하세요.")
            return 3
        if not json.loads(dp.read_text(encoding="utf-8")).get("proceed"):
            print(f"파일럿 중단조건: {dp.read_text(encoding='utf-8')}")
            return 3
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")

    # 1) 추출·청크
    first_extract = not (run_dir / "corpus_report.json").exists()
    corpus, chunks, sources = build_corpus(BASE / cases["corpus_dir"], run_dir)
    (run_dir / "corpus_report.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[코퍼스] {len(corpus)}개 · 청크 {len(chunks)}개 (chunk={CHUNK_CHARS}자·overlap={CHUNK_OVERLAP})")
    for d in corpus:
        flag = "✓" if d["chars"] > 200 else "✗ 추출 실패"
        print(f"  {flag} {d['file']:<38} {d['chars']:>7}자 (한글 {d['korean_chars']:>6}) · 청크 {d['chunks']:>3} · {d['tool']}")
    failed = [d for d in corpus if d["chars"] <= 200]
    if first_extract:
        for d in failed:
            live_mark("break", _clip(
                f"추출 실패 — {d['file']} · {d['size_kb']:,}KB 파일에서 텍스트 {d['chars']}자 · 이 문서는 검색 대상에서 빠짐"))
    live_mark("turn", _clip(
        f"추출 끝 → 임베딩 — 문서 {len(corpus)}개 중 {len(corpus) - len(failed)}개에서 청크 {len(chunks)}개 · {EMBED_MODEL}"))

    # 2) 임베딩(청크·질문 한 번에) — 생성 전에 bge-m3 를 내린다
    t0 = time.perf_counter()
    cvec = embed(base, chunks, args.timeout)
    embed_s = time.perf_counter() - t0
    qvec = embed(base, [q["q"] for q in questions], args.timeout)
    unload(base, EMBED_MODEL, args.timeout)
    index_info = {"n_chunks": len(chunks), "embed_s": round(embed_s, 1),
                  "chunks_per_s": round(len(chunks) / embed_s, 1) if embed_s else None,
                  "chunk_sources": {s: sources.count(s) for s in sorted(set(sources))}}
    print(f"[임베딩] {EMBED_MODEL} · 청크 {len(chunks)}개 {embed_s:.1f}초 · 질문 {len(questions)}개 · 임베딩 모델 내림")
    qidx = {q["id"]: i for i, q in enumerate(questions)}

    meta = {"ollama_version": None, "hardware": _gpu_name(), "model_metadata": None}
    try:
        meta["ollama_version"] = api_get(base + "/api/version", args.timeout).get("version")
        show = api_json(base + "/api/show", {"model": args.model}, args.timeout)
        meta["model_metadata"] = {"details": show.get("details"), "parameters": show.get("parameters"),
                                  "modified_at": show.get("modified_at")}
    except Exception as exc:  # noqa: BLE001
        print(f"[메타] 조회 실패: {exc}")

    # 3) 생성
    existing, next_id = _existing(run_dir)
    pending = [r for r in plan if r["key"] not in existing]
    print(f"mode={args.mode} planned={len(plan)} existing={len(plan) - len(pending)} running_now={len(pending)}")
    seen = _seen_breaks(run_dir, qmap)
    prev = None
    if pending:
        if args.mode == "pilot":
            live_mark("turn", _clip(f"파일럿 시작 — {args.model} · 문항 {len(PILOT_IDS)}개 × 검색 조각 3·6개 · {len(pending)}회"))
        else:
            live_mark("turn", _clip(f"본측정으로 전환 — {args.model} · 문항 {len(questions)}개 × 조각 3·6개 × 반복 {REPS}회 · 기존 {len(plan) - len(pending)}회 포함 {len(pending)}회 추가"))
    for pos, row in enumerate(pending, 1):
        cur = (row["rep"], row["top_k"])
        if prev is not None and cur != prev:
            live_mark("turn", _clip(f"조건 전환 — 반복 {prev[0]}·조각 {prev[1]}개 → 반복 {cur[0]}·조각 {cur[1]}개"))
        prev = cur
        q = row["q"]
        t0 = time.perf_counter()
        sims = cvec @ qvec[qidx[q["id"]]]
        top = np.argsort(-sims)[: row["top_k"]]
        retrieve_s = time.perf_counter() - t0
        retrieved = [{"src": sources[i], "score": round(float(sims[i]), 4), "chunk_index": int(i), "text": chunks[i]}
                     for i in top]
        context = "\n\n".join(f"({r['src']}) {r['text']}" for r in retrieved)
        prompt = PROMPT.format(context=context, question=q["q"])
        resp, attempts, infra = generate(base, args.model, prompt, args.timeout)
        answer = (resp or {}).get("response", "") if isinstance((resp or {}).get("response", ""), str) else ""
        answer = answer.strip()
        if resp is not None and not answer and not infra:
            infra = "empty_response"  # 빈 응답은 측정 실패다 — 오답·지어냄으로 채점하지 않는다
        if attempts and (infra or any(a["error"] for a in attempts)):
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · {infra or attempts[0]['error']} · 시도 {len(attempts)}/2"))
        elif resp is not None and not answer:
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · 응답은 왔지만 빈 텍스트"))
        next_id += 1
        stem = f"{next_id:03d}"
        resp_rel, tr_rel, inv_rel = f"raw/{stem}-response.txt", f"raw/{stem}-output.txt", f"raw/{stem}-invocation.json"
        (run_dir / resp_rel).write_text(answer, encoding="utf-8")
        (run_dir / tr_rel).write_text(
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {args.model}\n\n[PROMPT]\n{prompt}\n\n[RAW RESPONSE]\n{answer}\n"
            f"\n[API METADATA]\n{json.dumps({k: v for k, v in (resp or {}).items() if k != 'context'}, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8")
        metrics = {k: (resp or {}).get(k) for k in ("done_reason", "total_duration", "load_duration",
                                                    "prompt_eval_count", "prompt_eval_duration", "eval_count",
                                                    "eval_duration")} if resp else None
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "qid": q["id"], "top_k": row["top_k"],
               "rep": row["rep"], "question": q["q"], "prompt": prompt,
               "payload": {"model": args.model, "stream": False, "options": {"num_ctx": NUM_CTX}},
               "retrieved": retrieved, "retrieve_s": round(retrieve_s, 4), "attempts": attempts,
               "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "response_file": resp_rel, "transcript_file": tr_rel, "api_metrics": metrics}
        (run_dir / inv_rel).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        s = score_answer(q, answer, [r["text"] for r in retrieved]) if not infra else {"verdict": "infra",
                                                                                       "gold_retrieved": None}
        print(f"[{pos}/{len(pending)}] {row['key']} {inv['gen_s']:.1f}s · {s['verdict']}"
              f" · 근거조각={s['gold_retrieved']} · {answer[:40]!r}", flush=True)
        v = s["verdict"]
        if v == "wrong" and "wrong" not in seen:
            seen.add(v)
            live_mark("break", _clip(
                f"첫 오답 — {q['id']}({q['kind']})/조각{row['top_k']}/반복{row['rep']} · 정답 {q['expect'].split('(')[0].strip()}"
                f" · 답변 「{answer[:40]}」 · 정답 조각 검색={'예' if s['gold_retrieved'] else '아니오'}"))
        elif v == "hallucinated" and "hallucinated" not in seen:
            seen.add(v)
            live_mark("break", _clip(
                f"함정 첫 지어냄 — {q['id']}/조각{row['top_k']}/반복{row['rep']} · 문서에 없는 질문 「{q['q'][:30]}」에 「{answer[:40]}」"))

    unload(base, args.model, args.timeout)
    write_run_yaml(run_dir, args.model, meta, corpus, index_info)
    rows, agg = score_run_dir(run_dir, args.cases, NUM_CTX, write_pilot=(args.mode == "pilot"))
    write_run_yaml(run_dir, args.model, meta, corpus, index_info)
    arms = agg["arms"]

    def part(k: str) -> str:
        a = arms[k]
        return (f"조각{k[1:]}개 정답 {a['answerable_correct']['hit']}/{a['answerable_correct']['total']}"
                f"·함정 거절 {a['trap_refused']['hit']}/{a['trap_refused']['total']}")

    complete = len(rows) >= len(plan) and all(r["key"] in {x["key"] for x in rows} for r in plan)
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(rows)}회 집계 — {part('k3')} · {part('k6')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(
            f"{len(rows)}회 집계 — {part('k3')} · {part('k6')} · 정답 조각 검색 "
            f"{arms['k3']['gold_retrieved']['hit']}/{arms['k3']['gold_retrieved']['total']}→"
            f"{arms['k6']['gold_retrieved']['hit']}/{arms['k6']['gold_retrieved']['total']} · 인프라 {agg['infra_errors']}건"))
    status = {"mode": args.mode, "runs": len(rows), "infra_errors": agg["infra_errors"], "complete": complete,
              "pilot_decision": agg.get("pilot_decision")}
    (run_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.mode == "pilot" and not (agg.get("pilot_decision") or {}).get("proceed"):
        return 3
    return 0 if complete or args.mode == "pilot" else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"))
    ap.add_argument("--rescore", action="store_true", help="기존 raw 를 재채점(GPU 불필요)")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.mode and not args.rescore:
        ap.error("--mode 또는 --rescore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
