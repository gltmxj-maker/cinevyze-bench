#!/usr/bin/env python
"""로컬 RAG 실측 벤치 — tested 글(내 PC에서 내 문서 검색·요약)의 수치를 받치는 재현 아티팩트.

파이프라인(전부 로컬·무료): 문서 → 텍스트 추출(pdftotext/libreoffice) → 청크 → 임베딩(ollama bge-m3)
→ 코사인 top-k 검색 → 로컬 LLM(ollama) 답변. 각 단계의 시간·산출을 그대로 기록한다.

★tested 무결성([[tested-evidence-integrity-committed-artifact]]): 여기서 나온 숫자만 글에 쓴다.
  - 추출 실패(스캔 PDF·HWP)도 '실패 그대로' 기록 — 깨지는 지점이 이 글의 핵심.
  - 정답 여부(correct)는 코드가 못 판정 → 답변 원문을 남기고 사람(메인)이 라벨링해 labels.json에 기록.
  - 문서에 답이 없는 질문(answerable=false)을 섞어 환각 여부를 본다.

usage:
  python rag_bench.py --extract        # 1단계: 코퍼스 텍스트 추출만(추출률 표 산출)
  python rag_bench.py --run            # 2단계: 임베딩+검색+생성 실측
  python rag_bench.py --run --model qwen2.5vl:7b
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus/rag")
RUN_DIR = os.path.join(BASE, "test_runs/local-rag-20260712")
OLLAMA = "http://localhost:11434"

EMBED_MODEL = "bge-m3"
GEN_MODEL = "gemma3:4b"
CHUNK_CHARS = 700
CHUNK_OVERLAP = 150
TOP_K = 3

PROMPT = """아래 [문서]만 근거로 질문에 답하세요. 문서에 답이 없으면 반드시 "문서에 없음"이라고만 답하세요.
추측하거나 아는 지식으로 채우지 마세요.

[문서]
{context}

[질문] {question}
[답변]"""


# ── 1. 추출 ────────────────────────────────────────────────
def extract_text(path: str) -> tuple[str, str]:
    """문서 → 텍스트. (텍스트, 사용한 도구). 실패해도 예외 안 내고 빈 문자열 반환."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            out = subprocess.run(["pdftotext", "-layout", path, "-"],
                                 capture_output=True, timeout=120)
            return out.stdout.decode("utf-8", "ignore"), "pdftotext -layout"
        if ext in (".hwp", ".hwpx"):
            # ① 흔한 방법 = libreoffice 변환. HWPX(신형 XML 포맷)는 여기서 'source file could not be
            #    loaded'로 실패한다(firsthand 2026-07-12). 실패 사실 자체가 이 글의 데이터.
            tmp = os.path.join(RUN_DIR, "tmp_convert")
            os.makedirs(tmp, exist_ok=True)
            subprocess.run(["libreoffice", "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                            "--outdir", tmp, path], capture_output=True, timeout=240)
            txt = os.path.join(tmp, os.path.splitext(os.path.basename(path))[0] + ".txt")
            if os.path.exists(txt):
                return open(txt, encoding="utf-8", errors="ignore").read(), "libreoffice --convert-to txt"
            # ② 우회 = HWPX는 사실 ZIP + XML. 본문 섹션의 텍스트 런(<hp:t>)만 긁으면 표준 라이브러리로 뽑힌다.
            if ext == ".hwpx":
                import zipfile
                from xml.etree import ElementTree as ET
                z = zipfile.ZipFile(path)
                secs = [n for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n)]
                runs = []
                for s in secs:
                    for el in ET.fromstring(z.read(s)).iter():
                        if el.tag.endswith("}t") and el.text:
                            runs.append(el.text)
                if runs:
                    return " ".join(runs), "직접 파싱(zip+xml) — libreoffice 변환 실패 후 우회"
            return "", "libreoffice(변환 실패)"
    except Exception as e:  # 추출 실패도 데이터다
        return "", f"{ext} 실패: {type(e).__name__}"
    return "", f"{ext} 미지원"


def corpus_report() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*"))):
        if os.path.isdir(path):
            continue
        t0 = time.perf_counter()
        text, tool = extract_text(path)
        rows.append({
            "file": os.path.basename(path),
            "size_kb": round(os.path.getsize(path) / 1024),
            "tool": tool,
            "chars": len(text.strip()),
            "korean_chars": sum(1 for c in text if "가" <= c <= "힣"),
            "extract_s": round(time.perf_counter() - t0, 2),
        })
    return rows


# ── 2. 청크·임베딩·검색 ────────────────────────────────────
def chunk(text: str) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return [c for c in out if len(c.strip()) > 50]


def embed(texts: list[str]) -> np.ndarray:
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=600)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def generate(model: str, prompt: str) -> dict:
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0}}, timeout=900)
    r.raise_for_status()
    return r.json()


# ── 3. 증거 산출(run.yaml·출력·호출로그) — 실행 결과 JSON에서 기계가 자동 기록 ──
# ★write-origin(C3): run.yaml을 사람이 손으로 쓰면 '가짜 실측'이 가능해진다 → 이 함수만이 run.yaml을
#   쓴다(generated_by=rag_bench.py). 게이트는 이 값을 기계 write-origin 화이트리스트로 검증한다.
def emit_artifacts():
    out_dir = os.path.join(RUN_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    docs = json.load(open(os.path.join(RUN_DIR, "corpus_report.json"), encoding="utf-8"))
    runs = []
    for jf in sorted(glob.glob(os.path.join(RUN_DIR, "rag_bench_*.json"))):
        d = json.load(open(jf, encoding="utf-8"))
        model = d["gen_model"]
        stem = model.replace(":", "-")
        # 출력파일 = 모델이 실제로 낸 답변 원문
        with open(os.path.join(out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            for r in d["results"]:
                f.write(f"[{r['id']}] {r['q']}\n답변: {r['answer']}\n\n")
        # 호출로그 = 질문별 검색 근거·소요시간·토큰(호출 1건당 1줄 · access=local의 증거)
        with open(os.path.join(out_dir, f"{stem}.log"), "w", encoding="utf-8") as f:
            for r in d["results"]:
                f.write(json.dumps({
                    "id": r["id"], "model": model, "endpoint": f"{OLLAMA}/api/generate",
                    "retrieved": r["retrieved_srcs"], "top_scores": [x["score"] for x in r["retrieved"]],
                    "answer": r["answer"], "retrieve_s": r["retrieve_s"], "gen_s": r["gen_s"],
                    "eval_tokens": r["eval_tokens"], "tokens_per_s": r["tokens_per_s"],
                }, ensure_ascii=False) + "\n")
        runs.append({
            "model": model, "task": "RAG-01",
            "output_file": f"outputs/{stem}.txt", "log_file": f"outputs/{stem}.log",
            "raw_json": os.path.basename(jf),
            "questions": len(d["results"]),
            "n_chunks": d["n_chunks"], "embed_s": d["embed_s"], "chunks_per_s": d["chunks_per_s"],
            "retrieve_s_median": round(sorted(r["retrieve_s"] for r in d["results"])[len(d["results"]) // 2], 3),
            "gen_s_median": round(sorted(r["gen_s"] for r in d["results"])[len(d["results"]) // 2], 1),
        })

    meta = {
        "tool": "ollama (bge-m3 임베딩 + 로컬 LLM 생성) + pdftotext + 자작 RAG 파이프라인",
        "date": "2026-07-12",
        "method": "RAG-01",  # 한국 공공문서 4종 → 질문 12개(정답 10 + 함정 2)
        "access": "local",
        "generated_by": "rag_bench.py",   # ★기계 write-origin(사람 수작업 아님)
        "harness_version": "2.0",
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관) · 코퍼스=공공기관 공개문서",
        "hardware": "RTX 4070 Ti SUPER 16GB · ollama 0.31.2",
        "gpu_policy": "모델 1개씩 순차 적재(동시 적재 금지) · 실측 후 즉시 unload + 서버 종료",
        "prompt": PROMPT,
        "chunk_chars": CHUNK_CHARS, "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K,
        "corpus": docs,
        "runs": runs,
    }
    import yaml
    with open(os.path.join(RUN_DIR, "run.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    print(f"[emit] run.yaml + outputs/*.txt + outputs/*.log 기록 (모델 {len(runs)}종·GPU 미사용)")
    for r in runs:
        print(f"  · {r['model']}: {r['output_file']} · {r['log_file']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="추출 단계만(코퍼스 표)")
    ap.add_argument("--run", action="store_true", help="전체 실측")
    ap.add_argument("--emit", action="store_true",
                    help="이미 실행된 결과 JSON에서 run.yaml·출력파일·호출로그를 자동 생성(GPU 불필요)")
    ap.add_argument("--model", default=GEN_MODEL)
    args = ap.parse_args()

    if args.emit:
        emit_artifacts()
        return

    os.makedirs(RUN_DIR, exist_ok=True)
    docs = corpus_report()
    print(f"[코퍼스] {len(docs)}개")
    for d in docs:
        flag = "✓" if d["chars"] > 200 else "✗ 추출 실패"
        print(f"  {flag} {d['file']:<38} {d['chars']:>7}자 (한글 {d['korean_chars']:>6}) · {d['tool']} · {d['extract_s']}s")
    json.dump(docs, open(os.path.join(RUN_DIR, "corpus_report.json"), "w"),
              ensure_ascii=False, indent=2)
    if not args.run:
        return

    questions = json.load(open(os.path.join(BASE, "corpus/rag_questions.json"), encoding="utf-8"))

    # 인덱싱 — 추출 성공한 문서만(실패분은 애초에 검색 대상이 못 됨 = 그 자체가 결과)
    chunks, sources = [], []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*"))):
        if os.path.isdir(path):
            continue
        text, _ = extract_text(path)
        for c in chunk(text):
            chunks.append(c)
            sources.append(os.path.basename(path))
    print(f"[인덱싱] 청크 {len(chunks)}개 (chunk={CHUNK_CHARS}자·overlap={CHUNK_OVERLAP})")

    t0 = time.perf_counter()
    vecs = embed(chunks)
    embed_s = time.perf_counter() - t0
    print(f"[임베딩] {EMBED_MODEL} · {len(chunks)}청크 · {embed_s:.1f}초 ({len(chunks)/embed_s:.1f}청크/초)")

    results = []
    for q in questions:
        t0 = time.perf_counter()
        qv = embed([q["q"]])[0]
        sims = vecs @ qv
        idx = np.argsort(-sims)[:TOP_K]
        retrieve_s = time.perf_counter() - t0

        context = "\n\n".join(f"({sources[i]}) {chunks[i]}" for i in idx)
        t0 = time.perf_counter()
        gen = generate(args.model, PROMPT.format(context=context, question=q["q"]))
        gen_s = time.perf_counter() - t0

        results.append({
            "id": q["id"],
            "q": q["q"],
            "answerable": q["answerable"],
            "expect_doc": q.get("expect_doc"),
            "expect": q.get("expect"),
            "retrieved": [{"src": sources[i], "score": round(float(sims[i]), 3),
                           "head": chunks[i][:90]} for i in idx],
            "retrieved_srcs": [sources[i] for i in idx],
            "answer": gen.get("response", "").strip(),
            "retrieve_s": round(retrieve_s, 3),
            "gen_s": round(gen_s, 1),
            "eval_tokens": gen.get("eval_count"),
            "tokens_per_s": round(gen.get("eval_count", 0) / (gen.get("eval_duration", 1) / 1e9), 1)
            if gen.get("eval_duration") else None,
        })
        hit = q.get("expect_doc") in results[-1]["retrieved_srcs"] if q["answerable"] else None
        print(f"  [{q['id']}] 검색 {retrieve_s*1000:.0f}ms · 생성 {gen_s:.1f}s · "
              f"근거문서 적중={hit} · 답변: {results[-1]['answer'][:45]}…")

    payload = {
        "date": "2026-07-12",
        "hardware": "RTX 4070 Ti SUPER 16GB",
        "embed_model": EMBED_MODEL,
        "gen_model": args.model,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "corpus": docs,
        "n_chunks": len(chunks),
        "embed_s": round(embed_s, 1),
        "chunks_per_s": round(len(chunks) / embed_s, 1),
        "results": results,
    }
    out = os.path.join(RUN_DIR, f"rag_bench_{args.model.replace(':', '-')}.json")
    json.dump(payload, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n[저장] {out}")
    print("  ※ 정답 여부(correct)·환각 여부는 사람이 답변 원문을 읽고 라벨링(labels.json)")


if __name__ == "__main__":
    main()
