#!/usr/bin/env python3
"""Gemini 실무 3종(코딩·요약·번역) 결정론 채점기(2026-09-23 · 18번 재제작).

- 코딩: 응답의 파이썬 코드 블록을 별도 프로세스에서 실행해 숨은 케이스 7개를 넣는다.
- 요약: 문장 수 = 5 · 원문에 없는 숫자 0 (통과 조건). 추출 방식 4종·변수 3종 언급은 참고값(목록 기반 · 통과 조건 아님).
- 번역: 통용 용어 7개 존재(복수 표기 허용 · 목록 기반) · 괄호 밖 영어 단어 잔존.
★목록 기반 축(요약 언급·번역 용어)은 표기 변형을 다 못 잡는다 — 한계를 본문에 공시한다.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CODE_CASES = [
    ([3, 1, 2, 3, 1], [3, 2, 1]),
    ([], []),
    ([7], [7]),
    ([2, 2, 2], [2]),
    ([-1, 0, -1, 5], [5, 0, -1]),
    ([10, 9, 8], [10, 9, 8]),
    (list(range(1000)) * 2, list(range(999, -1, -1))),
]

_RUNNER = r'''
import contextlib, io, json, sys
src = open(sys.argv[1], encoding="utf-8").read()
cases = json.loads(open(sys.argv[2], encoding="utf-8").read())
ns = {"__name__": "__bench__"}
buf = io.StringIO()
out = {"load_error": None, "results": []}
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "answer.py", "exec"), ns)
except Exception as e:
    out["load_error"] = f"{type(e).__name__}: {e}"
fn = ns.get("dedup_sort")
if fn is None and not out["load_error"]:
    out["load_error"] = "dedup_sort 없음"
if fn is not None:
    for inp, exp in cases:
        arg = list(inp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = fn(arg)
            out["results"].append({"n": len(inp), "ok": list(got) == exp, "type": type(got).__name__,
                                   "mutated": arg != inp, "got_head": list(got)[:5]})
        except Exception as e:
            out["results"].append({"n": len(inp), "ok": False, "type": None, "mutated": None,
                                   "error": f"{type(e).__name__}: {e}"})
print(json.dumps(out, ensure_ascii=False))
'''

SUMMARY_METHODS = {"드립": ["드립"], "에스프레소": ["에스프레소"], "침지식": ["침지", "프렌치프레스", "프렌치 프레스"],
                   "콜드브루": ["콜드브루", "콜드 브루"]}
SUMMARY_VARS = {"분쇄도": ["분쇄", "갈"], "물 온도": ["온도"], "접촉 시간": ["시간"]}
TRANS_TERMS = {"latency": ["지연 시간", "지연시간", "레이턴시", "지연"],
               "throughput": ["처리량", "스루풋", "처리율"],
               "batch": ["배치", "배칭", "일괄"],
               "quantization": ["양자화"],
               "weights": ["가중치"],
               "context window": ["컨텍스트 윈도우", "컨텍스트 창", "문맥 창", "컨텍스트 윈도", "문맥 윈도우"],
               "token": ["토큰"]}
EN_ALLOW = {"GPU", "LLM", "AI"}


def code_blocks(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"```(?:python|py)?[^\n]*\n(.*?)```", text, re.S)]


def score_code(text: str) -> dict[str, Any]:
    blocks = code_blocks(text)
    src = next((b for b in blocks if "def dedup_sort" in b), None)
    detail: dict[str, Any] = {"blocks": len(blocks), "load_error": None, "cases": []}
    if src is None:
        detail["load_error"] = "dedup_sort 코드 블록 없음"
        return {"pass": False, "checks": {"all_cases": False, "example_shown": False}, "detail": detail}
    with tempfile.TemporaryDirectory() as td:
        sp, cp, rp = Path(td) / "answer.py", Path(td) / "cases.json", Path(td) / "runner.py"
        sp.write_text(src, encoding="utf-8")
        cp.write_text(json.dumps(CODE_CASES), encoding="utf-8")
        rp.write_text(_RUNNER, encoding="utf-8")
        try:
            from vertex_gemini import limit_child
            p = subprocess.run([sys.executable, "-I", str(rp), str(sp), str(cp)], cwd=td, capture_output=True,
                               text=True, timeout=30, env={"PATH": "/usr/bin:/bin", "HOME": td},
                               preexec_fn=limit_child)
            try:
                res = json.loads(p.stdout.strip().splitlines()[-1])
                if not isinstance(res, dict):
                    raise ValueError("결과가 dict 가 아님")
            except (IndexError, ValueError):  # 부분 출력·깨진 출력은 그 회차만 실패로 둔다
                res = {"load_error": f"러너 출력 파싱 실패 · stderr={p.stderr[-200:]}", "results": []}
        except subprocess.TimeoutExpired:
            res = {"load_error": "timeout 30s", "results": []}
    detail["load_error"] = res.get("load_error")
    detail["cases"] = res.get("results", [])
    all_ok = (not detail["load_error"]) and bool(detail["cases"]) and all(c.get("ok") for c in detail["cases"])
    example = bool(re.search(r"\[\s*3\s*,\s*2\s*,\s*1\s*\]", text))
    detail["return_types"] = sorted({c["type"] for c in detail["cases"] if c.get("type")})
    detail["mutates_input"] = any(c.get("mutated") for c in detail["cases"])
    return {"pass": all_ok and example, "checks": {"all_cases": all_ok, "example_shown": example}, "detail": detail}


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text)}


_LIST = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|\(\d+\)|[①-⑩])\s+")


def summary_body(text: str) -> tuple[list[str], list[str]]:
    """(본문 줄, 머리말 줄). 목록(번호·기호) 줄이 있으면 목록 줄만 요약 본문으로 보고 나머지는 머리말로 뺀다.
    목록이 없으면 전 줄이 본문이다."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    listed = [ln for ln in lines if _LIST.match(ln)]
    if listed:
        body = [re.sub(r"\*\*", "", _LIST.sub("", ln)).strip() for ln in listed]
        return body, [ln for ln in lines if not _LIST.match(ln)]
    return [re.sub(r"\*\*", "", ln) for ln in lines], []


def sentences(body: list[str]) -> list[str]:
    out = []
    for ln in body:
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if len(s.strip()) > 3]
    return out


def score_summary(text: str, source: str) -> dict[str, Any]:
    body, preface = summary_body(text)
    sents = sentences(body)
    body_txt = "\n".join(body)
    extra = sorted(_numbers(body_txt) - _numbers(source), key=lambda x: (len(x), x))
    methods = {k: any(v in body_txt for v in vs) for k, vs in SUMMARY_METHODS.items()}
    vars_ = {k: any(v in body_txt for v in vs) for k, vs in SUMMARY_VARS.items()}
    checks = {"five_sentences": len(sents) == 5, "no_extra_numbers": not extra}
    return {"pass": all(checks.values()), "checks": checks,
            "detail": {"sentence_count": len(sents), "extra_numbers": extra, "methods": methods,
                       "methods_all": all(methods.values()), "variables": vars_, "variables_all": all(vars_.values()),
                       "preface_lines": preface, "listed": bool(preface) or any(_LIST.match(l) for l in text.splitlines()),
                       "chars": len(body_txt)}}


def translate_body(text: str) -> tuple[str, list[str]]:
    """(번역 본문, 머리말·꼬리말 문단). 「번역」이라는 말이 든 문단은 원문(영어 단락)의 번역일 수 없으니
    머리말·꼬리말로 빼고 채점하지 않는다 — 머리말이 용어를 나열해 용어 축을 오염시키는 것을 막는다."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    extra = [p for p in paras if "번역" in p or p.endswith(":")]
    body = [p for p in paras if p not in extra]
    return "\n\n".join(body), extra


def score_translate(text: str) -> dict[str, Any]:
    body, extra_paras = translate_body(text)
    terms = {k: any(v in body for v in vs) for k, vs in TRANS_TERMS.items()}
    no_paren = re.sub(r"\([^)]*\)", "", body)
    en = [w for w in re.findall(r"[A-Za-z]{2,}", no_paren) if w not in EN_ALLOW]
    paren_en = re.findall(r"\(([A-Za-z][A-Za-z \-]*)\)", body)
    checks = {"terms_all": all(terms.values()), "no_stray_english": not en}
    return {"pass": all(checks.values()), "checks": checks,
            "detail": {"terms": terms, "missing_terms": [k for k, v in terms.items() if not v],
                       "stray_english": en, "paren_english": paren_en,
                       "preface_line": extra_paras[0] if extra_paras else None, "extra_paragraphs": len(extra_paras),
                       "chars": len(body)}}


def score(task: str, text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if task == "code":
        return score_code(text)
    if task == "summary":
        return score_summary(text, cfg["tasks"]["summary"]["source"])
    if task == "translate":
        return score_translate(text)
    raise ValueError(f"모르는 과업: {task!r}")


def _sim(texts: list[str]) -> dict[str, Any]:
    pairs = [difflib.SequenceMatcher(None, a, b).ratio() for i, a in enumerate(texts) for b in texts[i + 1:]]
    return {"n": len(texts), "identical_pairs": sum(1 for a in range(len(texts)) for b in range(a + 1, len(texts))
                                                   if texts[a] == texts[b]),
            "pairs": len(pairs), "sim_min": round(min(pairs), 3) if pairs else None,
            "sim_median": round(statistics.median(pairs), 3) if pairs else None}


def score_run_dir(run_dir: Path, cases: Path, write_pilot: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = json.loads(cases.read_text(encoding="utf-8"))
    rows, infra = [], 0
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        inv = json.loads(p.read_text(encoding="utf-8"))
        if inv.get("infra_error"):
            infra += 1
            continue
        text = (run_dir / inv["response_file"]).read_text(encoding="utf-8")
        s = score(inv["task"], text, cfg)
        u = inv.get("usage") or {}
        rows.append({"run_id": inv["run_id"], "key": inv["key"], "task": inv["task"], "arm": inv["arm"],
                     "rep": inv["rep"], "mode": inv["mode"], "pass": s["pass"], "checks": s["checks"],
                     # ★응답 시간 = 성공한 시도 1회의 시간. 429 로 재시도한 회차는 실패 시도 시간을 빼고 재시도 횟수를 따로 남긴다.
                     "detail": s["detail"], "gen_s": inv["attempts"][-1]["elapsed_s"],
                     "retries": len(inv["attempts"]) - 1, "finish_reason": inv.get("finish_reason"),
                     "prompt_tokens": u.get("promptTokenCount"), "output_tokens": u.get("candidatesTokenCount"),
                     "thought_tokens": u.get("thoughtsTokenCount"), "total_tokens": u.get("totalTokenCount"),
                     "response_file": inv["response_file"], "text": text})
    agg: dict[str, Any] = {"runs": len(rows), "infra_errors": infra, "retried_runs": sum(1 for r in rows if r["retries"]),
                           "arms": {}}
    for arm in sorted({r["arm"] for r in rows}):
        block = {}
        for task in ("code", "summary", "translate"):
            rs = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not rs:
                continue
            checks = {k: {"hit": sum(1 for r in rs if r["checks"][k]), "total": len(rs)} for k in rs[0]["checks"]}
            gen = [r["gen_s"] for r in rs]
            th = [r["thought_tokens"] or 0 for r in rs]
            ot = [r["output_tokens"] or 0 for r in rs]
            b = {"pass": {"hit": sum(1 for r in rs if r["pass"]), "total": len(rs)}, "checks": checks,
                 "gen_s_median": round(statistics.median(gen), 2), "gen_s_min": min(gen), "gen_s_max": max(gen),
                 "thought_tokens_median": statistics.median(th), "output_tokens_median": statistics.median(ot),
                 "thought_tokens_sum": sum(th), "output_tokens_sum": sum(ot),
                 "prompt_tokens_sum": sum(r["prompt_tokens"] or 0 for r in rs),
                 "similarity": _sim([r["text"] for r in rs])}
            if task == "summary":
                b["sentence_counts"] = sorted(r["detail"]["sentence_count"] for r in rs)
                b["methods_all"] = sum(1 for r in rs if r["detail"]["methods_all"])
                b["variables_all"] = sum(1 for r in rs if r["detail"]["variables_all"])
            if task == "translate":
                b["missing_terms"] = sorted({t for r in rs for t in r["detail"]["missing_terms"]})
                b["preface_runs"] = sum(1 for r in rs if r["detail"]["preface_line"])
            if task == "code":
                b["return_types"] = sorted({t for r in rs for t in r["detail"].get("return_types", [])})
                b["mutates_input_runs"] = sum(1 for r in rs if r["detail"].get("mutates_input"))
            block[task] = b
        agg["arms"][arm] = block
    (run_dir / "results.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "text"} for r in rows],
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "key", "task", "arm", "rep", "pass", "gen_s", "retries", "finish_reason", "prompt_tokens",
                    "output_tokens", "thought_tokens"])
        for r in rows:
            w.writerow([r["run_id"], r["key"], r["task"], r["arm"], r["rep"], r["pass"], r["gen_s"], r["retries"],
                        r["finish_reason"], r["prompt_tokens"], r["output_tokens"], r["thought_tokens"]])
    if write_pilot:
        reasons = []
        if infra:
            reasons.append(f"infra {infra}건")
        if not rows:
            reasons.append("채점된 회차 0")
        agg["pilot_decision"] = {"proceed": not reasons, "reasons": reasons, "runs": len(rows)}
        (run_dir / "pilot_decision.json").write_text(json.dumps(agg["pilot_decision"], ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    (run_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, agg
