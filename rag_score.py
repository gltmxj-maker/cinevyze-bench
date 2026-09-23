#!/usr/bin/env python3
"""로컬 RAG 벤치 채점기 — 답변 원문과 검색 조각만 보고 결정론으로 판정한다.

왜 따로 두는가: 2026-07-12 첫 런은 정답 여부를 사람이 읽고 labels.json 에 적었다. 그 방식은
재현이 안 되고 회차가 늘면 불가능하다. 여기서는 문항마다 정답 패턴(`accept`)과 근거 조각
패턴(`gold_chunk`)을 미리 박아 두고 코드가 판정한다. 판정이 애매한 조합(정답값과 「문서에 없음」이
함께 나온 답)은 따로 `hedged` 로 세고 정답에 넣지 않는다.

판정(verdict)
  답이 있는 문항: correct(정답값 있음) · abstain(「문서에 없음」만) · hedged(둘 다) · wrong(둘 다 없음)
  함정 문항:      refused(「문서에 없음」) · hedged(거절하면서 수치를 덧붙임) · hallucinated(거절 없이 답함)

usage:
  python3 rag_score.py --selftest
  python3 rag_score.py --run-dir test_runs/local-rag-20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

ABSTAIN = "문서에없음"
ABSTAIN_RE = re.compile(r"문서에(?:는)?(?:해당(?:내용|정보)?(?:이|가)?)?없(?:음|습니다|어요|다)")  # 존댓말 거절도 거절이다
NUMERIC_CLAIM = re.compile(r"\d+(\.\d+)?(%|원|달러|억|만|명|개|건)")  # 연·월·일은 문서 범위 설명이라 수치 주장으로 안 본다
ARMS = (3, 6)


def norm(text: str) -> str:
    return re.sub(r"[\s,]", "", text or "")


def answer_matches(question: dict[str, Any], answer: str) -> bool:
    n = norm(answer)
    if question.get("accept_all"):
        return all(re.search(p, n, re.I) for p in question["accept_all"])
    return any(re.search(p, n, re.I) for p in question.get("accept") or [])


def gold_retrieved(question: dict[str, Any], chunks: list[str]) -> bool | None:
    pattern = question.get("gold_chunk")
    if not pattern:
        return None
    return any(re.search(pattern, norm(c), re.I) for c in chunks)


def score_answer(question: dict[str, Any], answer: str, chunks: list[str] | None = None) -> dict[str, Any]:
    n = norm(answer)
    abstained = bool(ABSTAIN_RE.search(n))
    if question["answerable"]:
        matched = answer_matches(question, answer)
        if matched and not abstained:
            verdict = "correct"
        elif matched and abstained:
            verdict = "hedged"
        elif abstained:
            verdict = "abstain"
        else:
            verdict = "wrong"
    else:
        matched = False
        if not abstained:
            verdict = "hallucinated"
        else:
            rest = ABSTAIN_RE.sub("", n)
            verdict = "hedged" if NUMERIC_CLAIM.search(rest) else "refused"
    return {
        "verdict": verdict,
        "abstained": abstained,
        "matched": matched,
        "gold_retrieved": gold_retrieved(question, chunks or []) if question["answerable"] else None,
    }


def load_questions(cases_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}


def _frac(rows: list[dict[str, Any]], pred) -> dict[str, int]:
    return {"hit": sum(1 for r in rows if pred(r)), "total": len(rows)}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("infra_error")]
    out: dict[str, Any] = {"runs": len(rows), "infra_errors": len(rows) - len(ok), "arms": {}}
    for k in ARMS:
        arm = [r for r in ok if r["top_k"] == k]
        ans = [r for r in arm if r["answerable"]]
        trap = [r for r in arm if not r["answerable"]]
        verdicts: dict[str, int] = {}
        for r in arm:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        kinds: dict[str, dict[str, int]] = {}
        for kind in sorted({r["kind"] for r in ans}):
            sub = [r for r in ans if r["kind"] == kind]
            kinds[kind] = _frac(sub, lambda r: r["verdict"] == "correct")
        gold = [r for r in ans if r["gold_retrieved"]]
        nogold = [r for r in ans if r["gold_retrieved"] is False]
        out["arms"][f"k{k}"] = {
            "answerable_correct": _frac(ans, lambda r: r["verdict"] == "correct"),
            "trap_refused": _frac(trap, lambda r: r["verdict"] == "refused"),
            "trap_hallucinated": _frac(trap, lambda r: r["verdict"] == "hallucinated"),
            "gold_retrieved": _frac(ans, lambda r: r["gold_retrieved"] is True),
            "correct_when_gold": _frac(gold, lambda r: r["verdict"] == "correct"),
            "correct_when_no_gold": _frac(nogold, lambda r: r["verdict"] == "correct"),
            "wrong_when_no_gold": _frac(nogold, lambda r: r["verdict"] == "wrong"),
            "abstain_when_no_gold": _frac(nogold, lambda r: r["verdict"] == "abstain"),
            "verdicts": dict(sorted(verdicts.items())),
            "by_kind": kinds,
            "gen_s_median": round(statistics.median([r["gen_s"] for r in arm]), 2) if arm else None,
            "prompt_tokens_max": max((r.get("prompt_eval_count") or 0) for r in arm) if arm else None,
        }
    # 문항별 3회 안정성: 같은 팔에서 반복마다 판정이 같았나
    per_q: dict[str, dict[str, list[str]]] = {}
    for r in ok:
        per_q.setdefault(r["qid"], {}).setdefault(f"k{r['top_k']}", []).append(r["verdict"])
    out["per_question"] = {q: {arm: v for arm, v in sorted(a.items())} for q, a in sorted(per_q.items())}
    unstable = sorted(q for q, a in per_q.items() for v in a.values() if len(v) > 1 and len(set(v)) > 1)
    out["unstable_questions"] = sorted(set(unstable))
    return out


def pilot_decision(rows: list[dict[str, Any]], num_ctx: int) -> dict[str, Any]:
    reasons = []
    if not rows:
        reasons.append("no_rows")
    infra = sum(1 for r in rows if r.get("infra_error"))
    if rows and infra / len(rows) > 0.10:
        reasons.append(f"infra_rate {infra}/{len(rows)}")
    empty = sum(1 for r in rows if not r.get("infra_error") and not (r.get("answer") or "").strip())
    if empty:
        reasons.append(f"empty_answers {empty}")
    trunc = [r["run_id"] for r in rows if (r.get("prompt_eval_count") or 0) >= num_ctx - 16]
    if trunc:
        reasons.append(f"context_truncation_suspected {trunc}")
    return {"proceed": not reasons, "reasons": reasons, "runs": len(rows), "infra_errors": infra,
            "num_ctx": num_ctx}


def load_rows(run_dir: Path, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        inv = json.loads(path.read_text(encoding="utf-8"))
        q = questions[inv["qid"]]
        answer = (run_dir / inv["response_file"]).read_text(encoding="utf-8")
        chunks = [c["text"] for c in inv.get("retrieved") or []]
        s = score_answer(q, answer, chunks) if not inv.get("infra_error") else {
            "verdict": "infra", "abstained": None, "matched": None, "gold_retrieved": None}
        metrics = inv.get("api_metrics") or {}
        rows.append({
            "run_id": inv["run_id"], "key": inv["key"], "mode": inv.get("mode"), "qid": inv["qid"],
            "kind": q["kind"], "answerable": q["answerable"], "top_k": inv["top_k"], "rep": inv["rep"],
            "verdict": s["verdict"], "gold_retrieved": s["gold_retrieved"],
            "doc_hit": (q.get("expect_doc") in [c["src"] for c in inv.get("retrieved") or []])
            if q["answerable"] else None,
            "gen_s": inv.get("gen_s"), "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"), "infra_error": inv.get("infra_error"),
            "answer": answer,
        })
    return rows


def score_run_dir(run_dir: Path, cases_path: Path, num_ctx: int, write_pilot: bool = False):
    questions = load_questions(cases_path)
    rows = load_rows(run_dir, questions)
    agg = aggregate_rows(rows)
    if write_pilot:
        decision = pilot_decision([r for r in rows if r["mode"] == "pilot"], num_ctx)
        (run_dir / "pilot_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
        agg["pilot_decision"] = decision
    elif (run_dir / "pilot_decision.json").exists():
        agg["pilot_decision"] = json.loads((run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
    (run_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    # results.json = 회차별 판정 원본(작가 팩·장면 추출기가 읽는다). all_rules = 그 회차가 기대 행동을 했나
    (run_dir / "results.json").write_text(json.dumps([
        {**{k: v for k, v in r.items() if k != "answer"}, "all_rules": r["verdict"] in ("correct", "refused"),
         "response_head": r["answer"][:200]} for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["run_id", "key", "mode", "qid", "kind", "answerable", "top_k", "rep", "verdict",
              "gold_retrieved", "doc_hit", "gen_s", "prompt_eval_count", "eval_count", "infra_error"]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows, agg


def self_test() -> None:
    qa = {"id": "T", "answerable": True, "kind": "plain", "accept": ["(?<![\\d.])1\\.1명"],
          "gold_chunk": "담당인력은1\\.1명"}
    assert score_answer(qa, "평균 1.1명입니다.")["verdict"] == "correct"
    assert score_answer(qa, "11.1명")["verdict"] == "wrong"          # 숫자 경계
    assert score_answer(qa, "문서에 없음")["verdict"] == "abstain"
    assert score_answer(qa, "1.1명 — 다만 문서에 없음")["verdict"] == "hedged"
    assert score_answer(qa, "0.3명")["verdict"] == "wrong"
    assert score_answer(qa, "x", ["정보보호 담당 인력은 1.1명, IT"])["gold_retrieved"] is True
    assert score_answer(qa, "x", ["정보보호 담당 인력은 1.3명"])["gold_retrieved"] is False
    qall = {"id": "A", "answerable": True, "kind": "plain", "accept_all": ["의료", "통신", "에너지"]}
    assert score_answer(qall, "의료, 통신, 에너지 분야입니다")["verdict"] == "correct"
    assert score_answer(qall, "의료와 통신")["verdict"] == "wrong"
    qc = {"id": "C", "answerable": True, "kind": "plain", "accept": ["6500개"]}
    assert score_answer(qc, "6,500개입니다")["verdict"] == "correct"   # 쉼표 정규화
    trap = {"id": "X", "answerable": False, "kind": "trap"}
    assert score_answer(trap, "문서에 없음")["verdict"] == "refused"
    assert score_answer(trap, "문서에 없음.")["verdict"] == "refused"
    assert score_answer(trap, "문서에 없음 (문서는 2024년 조사만 다룹니다)")["verdict"] == "refused"
    assert score_answer(trap, "문서에 없음. 다만 2024년 조사는 0.3%입니다")["verdict"] == "hedged"
    assert score_answer(trap, "문서에 없음 (문서에는 해당 내용이 없습니다)")["verdict"] == "refused"
    assert score_answer(trap, "0.3%입니다")["verdict"] == "hallucinated"
    assert score_answer(trap, "")["verdict"] == "hallucinated"
    # 2026-09-23 파일럿에서 잡은 오탐: 단위 없이 숫자만 답한 정답(Q19 「7」)을 오답으로 셌다
    B, A = r"(?<![\d.])", r"(?![\d.]|월|일|년)"
    q7 = {"id": "L", "answerable": True, "kind": "layout", "accept": [B + "7(개소|개|곳)?" + A]}
    assert score_answer(q7, "7")["verdict"] == "correct"
    assert score_answer(q7, "7개소입니다")["verdict"] == "correct"
    assert score_answer(q7, "17개소")["verdict"] == "wrong"
    assert score_answer(q7, "2024년 7월")["verdict"] == "wrong"
    q12 = {"id": "P", "answerable": True, "kind": "confusable", "accept": [B + r"12(\.0)?(%|퍼센트)?" + A]}
    assert score_answer(q12, "12.0%")["verdict"] == "correct"
    assert score_answer(q12, "12월 6일")["verdict"] == "wrong"
    assert score_answer(q12, "36.7%")["verdict"] == "wrong"
    # 2026-09-23 본측정 감사에서 잡은 관대 판정: 「1.  1.  1명」 같은 깨진 출력을 1.1명 정답으로 셌다
    q11 = {"id": "M", "answerable": True, "kind": "plain", "accept": [r"(?<![\d.])1\.1(명)?(?![\d.])"]}
    assert score_answer(q11, "1.  1.  1명")["verdict"] == "wrong"
    assert score_answer(q11, "기업체당 정보보호 담당 인력은 1.1명입니다.")["verdict"] == "correct"
    # 2026-09-23 T2 적대검증: 존댓말 거절을 거절로 안 봤다
    assert score_answer(trap, "문서에 없습니다.")["verdict"] == "refused"
    assert score_answer(trap, "문서에는 해당 내용이 없습니다.")["verdict"] == "refused"
    assert score_answer(qa, "문서에 없습니다")["verdict"] == "abstain"
    print("rag_score selftest OK (29 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_bench_cases.json"))
    ap.add_argument("--num-ctx", type=int, default=8192)
    args = ap.parse_args()
    if args.selftest:
        self_test()
        return 0
    if not args.run_dir:
        ap.error("--run-dir 또는 --selftest")
    _, agg = score_run_dir(args.run_dir, args.cases, args.num_ctx)
    print(json.dumps(agg["arms"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
