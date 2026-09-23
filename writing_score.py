#!/usr/bin/env python3
"""블로그 글쓰기 벤치 채점기 — 원출력과 글감 메모만 보고 결정론으로 판정한다.

검사 항목(과업별)
  intro  (도입부)     : 키워드 · 존댓말 · 글자 수(공백 포함) · 마지막 문장 질문 · 메모 밖 숫자
  titles (제목 5개)   : 줄 수 5 · 번호·기호 없음 · 줄마다 글자 수 · 줄마다 키워드 · 메모 밖 숫자
  meta   (검색 설명문): 한 문단 · 글자 수 · 키워드 · 존댓말 · 메모 밖 숫자
  참고(판정 밖): 과장어 목록 적중 · 1인칭 표현(저는·제가·저희 등) 포함 — 둘 다 표면 표지라 통과/실패에 넣지 않는다.

★채점 전처리: 마크다운 강조(**·__)와 머리표(#)는 지우고 잰다 — 발행 화면에서 보이지 않는 기호라서다
  (2026-09-23 파일럿 감사: 강조 기호 4자 때문에 151자 설명문이 150자 상한에 걸렸다).
★과장어를 판정에서 뺀 이유(2026-09-23 파일럿 감사): 목록이 「완벽하게 만들기가 쉽지 않으신」 같은
  비과장 문장을 잡았다. 의미 판정을 손목록으로 하지 않는다(detector-design-standard).

★한계(본문에 그대로 공시한다)
  - 과장어는 `writing_bench_cases.json` 의 목록으로만 잡는다. 목록에 없는 과장은 못 잡는다.
  - 메모 밖 숫자는 아라비아 숫자만 센다. 「세 가지」처럼 말로 쓴 수는 세지 않는다.
  - 존댓말은 문장 끝이 「…다」인데 「…니다」가 아닌 문장(반말 평서문)이 있는지만 본다.

usage:
  python3 writing_score.py --selftest
  python3 writing_score.py --run-dir test_runs/ollama-gemma3-4b-writing-20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
FIRST_PERSON = re.compile(r"(?:^|[\s(“\"'‘])(저는|저도|제가|저 역시|저희|제 경험|직접 써)")
SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
LIST_MARK = re.compile(r"^\s*(?:\d+[.):]|\(\d+\)|\[\d+\]|[①-⑳]|[-*•·▶►]|#+)\s*")
ARMS = ("base", "guard")
TASKS = ("intro", "titles", "meta")


def _nums(text: str) -> set[str]:
    out = set()
    for m in NUM_RE.findall(text or ""):
        v = m.replace(",", "").rstrip(".")
        if v:
            out.add(v.lstrip("0") or "0")
    return out


def allowed_numbers(memo: dict[str, Any], task_cfg: dict[str, Any]) -> set[str]:
    allowed = set()
    for f in memo["facts"]:
        allowed |= _nums(f)
    allowed |= _nums(memo["keyword"]) | _nums(memo["name"])
    for k in ("min_chars", "max_chars", "count"):
        if task_cfg.get(k):
            allowed.add(str(task_cfg[k]))
    allowed.add(str(len(memo["facts"])))  # 메모 항목 수(「기능 5가지」)는 메모에서 셀 수 있는 수다
    return allowed


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t)
    t = re.sub(r"\*\*|__", "", t)
    t = re.sub(r"(?m)^\s*#+\s*", "", t)
    return t.strip()


def _sentences(text: str) -> list[str]:
    return [s.strip().strip('"“”\'') for s in SENT_SPLIT.split(text) if s.strip()]


def _plain_da(sentence: str) -> bool:
    """반말 평서문: 끝이 「다」인데 「니다」가 아니다(따옴표·마침표·이모지 제거 후)."""
    s = re.sub(r"[\s.!?…~\"'”’)\]]+$", "", sentence)
    s = re.sub(r"[^\w가-힣]+$", "", s)
    return s.endswith("다") and not s.endswith("니다")


def score(task: str, text: str, memo: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    t = _clean(text)
    kw = memo["keyword"]
    task_cfg = cfg["tasks"][task]
    hype = [w for w in cfg["hype_words"] if w in t]
    extra = sorted(_nums(t) - allowed_numbers(memo, task_cfg), key=lambda x: (len(x), x))
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {"hype": hype, "extra_numbers": extra,
                              "first_person": sorted(set(m.group(1) for m in FIRST_PERSON.finditer(t)))}
    if task == "intro":
        n = len(t)
        sents = _sentences(t)
        checks["keyword"] = kw in t
        checks["polite"] = not any(_plain_da(s) for s in sents)
        checks["length"] = task_cfg["min_chars"] <= n <= task_cfg["max_chars"]
        checks["ends_question"] = bool(sents) and sents[-1].rstrip().endswith("?")
        detail.update({"chars": n, "last_sentence": sents[-1][-60:] if sents else ""})
    elif task == "titles":
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        checks["count"] = len(lines) == task_cfg["count"]
        checks["no_marks"] = not any(LIST_MARK.match(ln) for ln in lines)
        stripped = [LIST_MARK.sub("", ln).strip().strip('"“”') for ln in lines]
        checks["length"] = bool(stripped) and all(len(ln) <= task_cfg["max_chars"] for ln in stripped)
        checks["keyword"] = bool(stripped) and all(kw in ln for ln in stripped)
        detail.update({"lines": len(lines), "max_line_chars": max((len(x) for x in stripped), default=0),
                       "missing_keyword_lines": sum(1 for ln in stripped if kw not in ln)})
    elif task == "meta":
        n = len(t)
        checks["single_paragraph"] = "\n" not in t
        checks["length"] = task_cfg["min_chars"] <= n <= task_cfg["max_chars"]
        checks["keyword"] = kw in t
        checks["polite"] = not any(_plain_da(s) for s in _sentences(t))
        detail.update({"chars": n})
    else:
        raise ValueError(task)
    checks["no_extra_numbers"] = not extra
    format_ok = all(v for k, v in checks.items() if k != "no_extra_numbers")
    return {"checks": checks, "format_ok": format_ok, "all_rules": format_ok and checks["no_extra_numbers"],
            "detail": detail}


def _frac(rows, pred):
    return {"hit": sum(1 for r in rows if pred(r)), "total": len(rows)}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("infra_error")]
    out: dict[str, Any] = {"runs": len(rows), "infra_errors": len(rows) - len(ok), "arms": {}}
    for arm in ARMS:
        a = [r for r in ok if r["arm"] == arm]
        block: dict[str, Any] = {
            "all_rules": _frac(a, lambda r: r["all_rules"]),
            "format_ok": _frac(a, lambda r: r["format_ok"]),
            "extra_numbers_runs": _frac(a, lambda r: not r["checks"]["no_extra_numbers"]),
            "tasks": {},
        }
        for task in TASKS:
            b = [r for r in a if r["task"] == task]
            checks = {}
            for key in sorted({k for r in b for k in r["checks"]}):
                checks[key] = _frac(b, lambda r, key=key: r["checks"].get(key) is True)
            block["tasks"][task] = {"all_rules": _frac(b, lambda r: r["all_rules"]),
                                    "format_ok": _frac(b, lambda r: r["format_ok"]),
                                    "checks": checks,
                                    "extra_numbers_runs": _frac(b, lambda r: not r["checks"]["no_extra_numbers"])}
            if task in ("intro", "meta") and b:
                chars = [r["detail"]["chars"] for r in b]
                block["tasks"][task]["chars_median"] = statistics.median(chars)
                block["tasks"][task]["chars_min"] = min(chars)
                block["tasks"][task]["chars_max"] = max(chars)
        block["hype_hit_runs"] = _frac(a, lambda r: bool(r["detail"]["hype"]))
        block["first_person_runs"] = _frac(a, lambda r: bool(r["detail"]["first_person"]))
        nums: dict[str, int] = {}
        for r in a:
            for n in r["detail"]["extra_numbers"]:
                nums[n] = nums.get(n, 0) + 1
        block["extra_number_values"] = dict(sorted(nums.items(), key=lambda kv: -kv[1]))
        block["gen_s_median"] = round(statistics.median([r["gen_s"] for r in a]), 2) if a else None
        out["arms"][arm] = block
    return out


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    if not rows:
        reasons.append("no_rows")
    infra = sum(1 for r in rows if r.get("infra_error"))
    if rows and infra / len(rows) > 0.10:
        reasons.append(f"infra_rate {infra}/{len(rows)}")
    empty = sum(1 for r in rows if not r.get("infra_error") and not r["text"].strip())
    if empty:
        reasons.append(f"empty_answers {empty}")
    return {"proceed": not reasons, "reasons": reasons, "runs": len(rows), "infra_errors": infra}


def load_rows(run_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    memos = {m["id"]: m for m in cfg["memos"]}
    rows = []
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        inv = json.loads(p.read_text(encoding="utf-8"))
        text = (run_dir / inv["response_file"]).read_text(encoding="utf-8")
        s = score(inv["task"], text, memos[inv["memo_id"]], cfg)
        m = inv.get("api_metrics") or {}
        rows.append({"run_id": inv["run_id"], "key": inv["key"], "mode": inv["mode"], "memo_id": inv["memo_id"],
                     "task": inv["task"], "arm": inv["arm"], "rep": inv["rep"], "infra_error": inv.get("infra_error"),
                     "gen_s": inv.get("gen_s"), "eval_count": m.get("eval_count"), "text": text, **s})
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
        [{**{k: v for k, v in r.items() if k != "text"}, "response_head": r["text"][:200]} for r in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    flat = ["run_id", "key", "mode", "memo_id", "task", "arm", "rep", "all_rules", "format_ok", "gen_s",
            "eval_count", "infra_error"]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        ck = sorted({k for r in rows for k in r["checks"]})
        w.writerow(flat + ck + ["extra_numbers", "hype"])
        for r in rows:
            w.writerow([r[k] for k in flat] + [r["checks"].get(k, "") for k in ck]
                       + [" ".join(r["detail"]["extra_numbers"]), " ".join(r["detail"]["hype"])])
    return rows, agg


def self_test() -> None:
    cfg = {"hype_words": ["최고", "완벽"], "tasks": {"intro": {"min_chars": 20, "max_chars": 200},
                                                  "titles": {"count": 2, "max_chars": 20},
                                                  "meta": {"min_chars": 10, "max_chars": 80}}}
    memo = {"keyword": "회의록 요약 앱", "name": "노트K", "facts": ["월 8,900원입니다", "한 달 300분", "a", "b", "c"]}
    good = "회의록 요약 앱을 찾고 계신가요. 월 8,900원으로 한 달 300분을 씁니다. 어떤 회의부터 정리해 보시겠어요?"
    s = score("intro", good, memo, cfg)
    assert s["all_rules"], s
    s = score("intro", good.replace("8,900원", "9,900원"), memo, cfg)
    assert s["detail"]["extra_numbers"] == ["9900"] and not s["all_rules"] and s["format_ok"], s
    s = score("intro", "회의록 요약 앱은 편하다. 써 보시겠어요?", memo, cfg)
    assert not s["checks"]["polite"], s
    s = score("intro", "회의록 요약 앱은 편합니다. 한번 써 보세요.", memo, cfg)
    assert not s["checks"]["ends_question"], s
    s = score("intro", "회의록 요약 앱은 최고입니다. 써 보시겠어요?", memo, cfg)
    assert s["detail"]["hype"] == ["최고"] and s["all_rules"], s     # 과장어는 참고 표지 — 판정 밖
    s = score("intro", "저 역시 회의록 요약 앱을 찾고 있었습니다. 써 보시겠어요?", memo, cfg)
    assert s["detail"]["first_person"] == ["저 역시"], s
    s = score("meta", "**회의록 요약 앱**으로 한 달 300분을 정리합니다.", memo, cfg)
    assert s["detail"]["chars"] == len("회의록 요약 앱으로 한 달 300분을 정리합니다."), s   # 강조 기호는 글자 수에서 뺀다
    s = score("intro", "회의록 요약 앱의 기능 5가지를 봅니다. 써 보시겠어요?", memo, cfg)
    assert s["checks"]["no_extra_numbers"], s          # 메모 항목 수 5는 허용
    s = score("titles", "회의록 요약 앱 고르는 법\n회의록 요약 앱 무료 300분", memo, cfg)
    assert s["all_rules"], s
    s = score("titles", "1. 회의록 요약 앱 고르는 법\n2. 회의록 요약 앱 무료", memo, cfg)
    assert not s["checks"]["no_marks"], s
    s = score("titles", "회의록 요약 앱 고르는 법\n요약 앱 무료", memo, cfg)
    assert not s["checks"]["keyword"], s
    s = score("meta", "회의록 요약 앱으로 한 달 300분을 정리합니다.", memo, cfg)
    assert s["all_rules"], s
    s = score("meta", "회의록 요약 앱으로 정리합니다.\n두 줄", memo, cfg)
    assert not s["checks"]["single_paragraph"], s
    s = score("meta", "```\n회의록 요약 앱으로 한 달 300분을 정리합니다.\n```", memo, cfg)
    assert s["checks"]["single_paragraph"], s           # 코드펜스 하나는 벗겨서 본다
    # 2026-09-23 T2 적대검증: 원문자·괄호·콜론 번호를 못 잡았다
    for marked in ("① 회의록 요약 앱 소개\n② 회의록 요약 앱 무료", "(1) 회의록 요약 앱 소개\n(2) 회의록 요약 앱 무료",
                   "1: 회의록 요약 앱 소개\n2: 회의록 요약 앱 무료"):
        assert not score("titles", marked, memo, cfg)["checks"]["no_marks"], marked
    print("writing_score selftest OK (17 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("writing_bench_cases.json"))
    args = ap.parse_args()
    if args.selftest:
        self_test()
        return 0
    if not args.run_dir:
        ap.error("--run-dir 또는 --selftest")
    _, agg = score_run_dir(args.run_dir, args.cases)
    print(json.dumps({a: {"all_rules": b["all_rules"], "extra": b["extra_numbers_runs"]} for a, b in agg["arms"].items()},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
