#!/usr/bin/env python3
"""맞춤법·문장 교정 벤치 채점기 — 교정 결과 원문과 심은 오류 목록만 보고 결정론으로 판정한다.

오류 자리마다(`proofread_bench_cases.json` errors = [틀린 형태, 맞는 형태])
  fixed     : 틀린 형태가 사라지고 맞는 형태가 나왔다
  missed    : 틀린 형태가 그대로 남았다
  rewritten : 틀린 형태는 사라졌지만 맞는 형태도 없다(문장을 바꿔 써서 그 자리가 없어졌다)
의미 보존: anchors(숫자·이름·날짜)가 결과에 그대로 남았는지.
참고(판정 밖): 결과 글자 수 / 원문 글자 수 · 첫 줄(설명 머리말이 붙었는지 사람이 볼 수 있게).

★한계(본문에 공시한다)
  - 심은 오류 50개만 센다. 모델이 새로 만든 오류나 심지 않은 오류를 고친 것은 세지 않는다.
  - rewritten 은 틀린 것이 아니다 — 다듬기 지시에서는 문장을 바꾸는 게 정상일 수 있다.
  - 표지는 글자 그대로 찾는다. 「42분」을 「사십이 분」으로 바꾸면 사라진 것으로 센다.

usage:
  python3 proofread_score.py --selftest
  python3 proofread_score.py --run-dir test_runs/ollama-qwen3-vl-8b-proofread-20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

ARMS = ("spell", "polish")


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"(?m)^\s*```[a-z]*\s*$", "", t)     # 코드펜스 줄은 위치와 상관없이 지운다(T2)
    t = re.sub(r"\*\*|__", "", t)
    return t.strip()


def score(draft: dict[str, Any], text: str) -> dict[str, Any]:
    t = _clean(text)
    per = []
    for wrong, right in draft["errors"]:
        if wrong in t:
            v = "missed"
        elif right in t:
            v = "fixed"
        else:
            v = "rewritten"
        per.append({"wrong": wrong, "right": right, "verdict": v})
    lost = [a for a in draft["anchors"] if a not in t]
    first = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    return {"errors": per,
            "fixed": sum(1 for e in per if e["verdict"] == "fixed"),
            "missed": sum(1 for e in per if e["verdict"] == "missed"),
            "rewritten": sum(1 for e in per if e["verdict"] == "rewritten"),
            "all_fixed": all(e["verdict"] == "fixed" for e in per),
            "anchors_lost": lost, "anchors_kept": not lost,
            "len_ratio": round(len(t) / max(1, len(draft["text"])), 3), "first_line": first[:80]}


def _frac(rows, pred):
    return {"hit": sum(1 for r in rows if pred(r)), "total": len(rows)}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("infra_error")]
    out: dict[str, Any] = {"runs": len(rows), "infra_errors": len(rows) - len(ok),
                           "truncated": sum(1 for r in ok if r.get("done_reason") == "length"), "arms": {}}
    for arm in ARMS:
        a = [r for r in ok if r["arm"] == arm]
        errs = [e for r in a for e in r["errors"]]
        by_err: dict[str, dict[str, int]] = {}
        for r in a:
            for e in r["errors"]:
                k = f"{r['did']}:{e['wrong']}→{e['right']}"
                by_err.setdefault(k, {"fixed": 0, "missed": 0, "rewritten": 0})[e["verdict"]] += 1
        lost: dict[str, int] = {}
        for r in a:
            for x in r["anchors_lost"]:
                lost[f"{r['did']}:{x}"] = lost.get(f"{r['did']}:{x}", 0) + 1
        out["arms"][arm] = {
            "errors_fixed": _frac(errs, lambda e: e["verdict"] == "fixed"),
            "errors_missed": _frac(errs, lambda e: e["verdict"] == "missed"),
            "errors_rewritten": _frac(errs, lambda e: e["verdict"] == "rewritten"),
            "drafts_all_fixed": _frac(a, lambda r: r["all_fixed"]),
            "drafts_anchors_kept": _frac(a, lambda r: r["anchors_kept"]),
            "anchors_lost": dict(sorted(lost.items(), key=lambda kv: -kv[1])),
            "len_ratio_median": statistics.median([r["len_ratio"] for r in a]) if a else None,
            "len_ratio_min": min((r["len_ratio"] for r in a), default=None),
            "len_ratio_max": max((r["len_ratio"] for r in a), default=None),
            "per_error": dict(sorted(by_err.items())),
            "gen_s_median": round(statistics.median([r["gen_s"] for r in a]), 2) if a else None,
            "thinking_chars_median": statistics.median([r["thinking_chars"] for r in a]) if a else None,
        }
    return out


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    if not rows:
        reasons.append("no_rows")
    infra = sum(1 for r in rows if r.get("infra_error"))
    if rows and infra / len(rows) > 0.10:
        reasons.append(f"infra_rate {infra}/{len(rows)}")
    trunc = sum(1 for r in rows if r.get("done_reason") == "length")
    if trunc:
        reasons.append(f"truncated {trunc}")
    return {"proceed": not reasons, "reasons": reasons, "runs": len(rows), "infra_errors": infra}


def load_rows(run_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ds = {d["id"]: d for d in cfg["drafts"]}
    rows = []
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        inv = json.loads(p.read_text(encoding="utf-8"))
        text = (run_dir / inv["response_file"]).read_text(encoding="utf-8")
        s = score(ds[inv["did"]], text)
        m = inv.get("api_metrics") or {}
        rows.append({"run_id": inv["run_id"], "key": inv["key"], "mode": inv["mode"], "did": inv["did"],
                     "arm": inv["arm"], "rep": inv["rep"], "infra_error": inv.get("infra_error"),
                     "gen_s": inv.get("gen_s"), "eval_count": m.get("eval_count"), "done_reason": m.get("done_reason"),
                     "thinking_chars": inv.get("thinking_chars", 0), "text": text, **s})
    return rows


def score_run_dir(run_dir: Path, cases: Path, write_pilot: bool = False):
    cfg = json.loads(cases.read_text(encoding="utf-8"))
    rows = load_rows(run_dir, cfg)
    agg = aggregate_rows(rows)
    if write_pilot:
        d = pilot_decision([r for r in rows if r["mode"] == "pilot"])
        (run_dir / "pilot_decision.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        agg["pilot_decision"] = d
    elif (run_dir / "pilot_decision.json").exists():
        agg["pilot_decision"] = json.loads((run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
    (run_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "text"} | {"response": r["text"]} for r in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    flat = ["run_id", "key", "mode", "did", "arm", "rep", "fixed", "missed", "rewritten", "all_fixed",
            "anchors_kept", "len_ratio", "gen_s", "eval_count", "thinking_chars", "done_reason", "infra_error"]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(flat + ["anchors_lost", "missed_forms", "rewritten_forms"])
        for r in rows:
            w.writerow([r[k] for k in flat] + [" ".join(r["anchors_lost"]),
                                               " ".join(e["wrong"] for e in r["errors"] if e["verdict"] == "missed"),
                                               " ".join(e["wrong"] for e in r["errors"] if e["verdict"] == "rewritten")])
    return rows, agg


def self_test() -> None:
    d = {"id": "D", "text": "몇일 안 됐는데 금새 끝났다. 42분 걸렸다.",
         "errors": [["몇일", "며칠"], ["금새", "금세"]], "anchors": ["42분"]}
    s = score(d, "며칠 안 됐는데 금세 끝났다. 42분 걸렸다.")
    assert s["fixed"] == 2 and s["all_fixed"] and s["anchors_kept"], s
    s = score(d, "몇일 안 됐는데 금세 끝났다. 42분 걸렸다.")
    assert s["missed"] == 1 and s["fixed"] == 1, s
    s = score(d, "얼마 안 돼 곧 끝났습니다. 사십이 분 걸렸습니다.")
    assert s["rewritten"] == 2 and s["anchors_lost"] == ["42분"], s
    s = score(d, "```\n**며칠** 안 됐는데 금세 끝났다. 42분 걸렸다.\n```")
    assert s["all_fixed"], s                                  # 코드펜스·강조 벗김
    rows = [dict(s, run_id=1, key="k", mode="full", did="D", arm="spell", rep=1, infra_error=None, gen_s=1.0,
                 eval_count=1, done_reason="stop", thinking_chars=0, text="")]
    agg = aggregate_rows(rows)
    assert agg["arms"]["spell"]["errors_fixed"] == {"hit": 2, "total": 2}, agg
    print("proofread_score selftest OK (5 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("proofread_bench_cases.json"))
    args = ap.parse_args()
    if args.selftest:
        self_test()
        return 0
    if not args.run_dir:
        ap.error("--run-dir 또는 --selftest")
    _, agg = score_run_dir(args.run_dir, args.cases)
    print(json.dumps({a: {k: b[k] for k in ("errors_fixed", "errors_missed", "drafts_anchors_kept")}
                      for a, b in agg["arms"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
