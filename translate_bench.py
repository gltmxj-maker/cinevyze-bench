#!/usr/bin/env python
"""오프라인(로컬) 번역 벤치 — tested 글 수치를 받치는 재현 아티팩트.

로컬 gemma3(ollama HTTP API·CLI 아님→깨끗한 출력·정확한 tok/s)로 영문 3종을 EN→KO 번역하고,
다시 KO→EN 왕복번역까지 돌려 *속도(객관)* + *실제 출력(투명)* + *왕복 결과(의미보존 가시화)* 를
test_runs/offline-translate-gemma3/translate_bench.json 에 남긴다. (run.yaml=하네스 tested 스탬프 / 이 bench=tok/s·왕복 분석.)

★번역 '품질 점수'는 일부러 숫자로 안 박는다(번역 품질=주관·가짜정밀 회피). 속도만 실측·품질은 실제 출력으로 독자 판단.
ollama 데몬은 어댑터가 자동 기동. usage: python translate_bench.py
"""
import os, json, time
import tool_adapters as TA

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "test_runs/offline-translate-gemma3")
SAMPLES = [
    ("기술 문서", "corpus/translate/s1-tech.txt"),
    ("캐주얼 후기", "corpus/translate/s2-casual.txt"),
    ("비즈니스 메일", "corpus/translate/s3-email.txt"),
]
MODEL = "gemma3:4b"
EN2KO = "다음 영어 텍스트를 자연스러운 한국어로 번역해줘. 번역문만 출력하고 다른 설명·머리말은 붙이지 마.\n\n"
KO2EN = "Translate the following Korean text into natural English. Output only the English translation, no preamble.\n\n"


def main():
    ad = TA.OllamaAdapter()
    if not ad.available():
        raise SystemExit("ollama/gemma3 미가용 — prepare 확인.")
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for label, rel in SAMPLES:
        en = open(os.path.join(BASE, rel), encoding="utf-8").read().strip()
        # EN→KO (속도 측정)
        t0 = time.monotonic()
        ko = ad.run(EN2KO + en, model=MODEL)
        wall = round(time.monotonic() - t0, 2)
        m = dict(ad.last_meta)
        # KO→EN 왕복(의미보존 가시화·품질 점수 아님)
        back = ad.run(KO2EN + ko, model=MODEL)
        ec, ed = m.get("eval_count"), m.get("eval_duration_ns")
        rows.append({
            "sample": label, "src_file": rel,
            "en_words": len(en.split()), "en_chars": len(en), "ko_chars": len(ko),
            "ko_output": ko, "back_en_output": back,
            # ★tok/s 재현 가능하게 분모(eval_duration)까지 저장(적대검증 GLM 지적): tok/s = eval_count/(eval_duration_ns/1e9)
            "tok_per_s": m.get("tok_per_s"), "eval_count": ec, "eval_duration_ns": ed,
            "gen_s_pure": round(ed / 1e9, 3) if ed else None,   # 순수 생성시간(tok/s의 분모)
            "wall_s_en2ko": wall,   # 전체 wall(프롬프트 처리+오버헤드 포함·gen_s_pure보다 큼)
        })
        print(f"[{label}] tok/s={m.get('tok_per_s')} wall={wall}s ko_chars={len(ko)}")

    toks = [r["tok_per_s"] for r in rows if r["tok_per_s"]]
    summary = {
        "model": MODEL + " (로컬·ollama HTTP API)",
        "device": "로컬 PC(GPU)·인터넷/계정 불필요",
        "samples": len(rows),
        "avg_tok_per_s": round(sum(toks) / len(toks), 1) if toks else None,
        "tok_per_s_range": [min(toks), max(toks)] if toks else None,
        "quality_note": "번역 품질은 점수화하지 않음(주관·가짜정밀 회피) — 실제 출력·왕복번역으로 독자가 판단. 측정한 건 속도뿐.",
        "measured": "2026-06",
        "rows": rows,
    }
    with open(os.path.join(OUTDIR, "translate_bench.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\navg tok/s:", summary["avg_tok_per_s"], "· saved", os.path.join(OUTDIR, "translate_bench.json"))


if __name__ == "__main__":
    main()
