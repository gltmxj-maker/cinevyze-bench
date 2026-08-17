#!/usr/bin/env python3
"""Score the length-instruction compliance benchmark.

The question is not whether the text is good but whether the model landed on the number of
characters it was told to produce. That makes the cleaning rule the load-bearing part of
this scorer: if the raw string were measured as-is, a model that wrapped its answer in a
code fence or added a "물론이죠!" preamble would be judged on characters it was told not to
write. So the measured body is the answer with fences, headings, list bullets, and emphasis
markers removed — the characters a person would actually paste.

Compliance is reported at two tolerances (±10% and ±20%) rather than one, because the
prompt phrasings themselves promise different tolerances, and a single threshold would
quietly favour whichever phrasing it happened to match.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_INLINE_FENCE_RE = re.compile(r"```[^\n]*\n?|```")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
# ★T2 적대검증(2026-08-01): 숫자 목록 패턴(`\d+\.`)은 "1. 2026년은…"이나 "2026. 8. 1. 공지"
# 같은 정상 문장의 앞부분까지 지워 길이를 실제보다 짧게 만든다. 마크다운 목록은 기호 불릿으로
# 충분히 잡히고, 숫자 목록을 놓쳐서 생기는 오차는 몇 글자 수준이라 '본문을 지우는' 위험보다 작다.
# (이번 64회 원자료 확인: 숫자 목록 매칭 0건 / 기호 불릿 41건 = 전부 진짜 목록 → 결과 무영향.)
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"\*{1,3}|_{2,3}|`")
# "다음은 …입니다"는 금지된 머리말이기도 하지만 정상 본문의 첫 문장이기도 하다("다음은 준비물
# 목록입니다. / 텐트와 침낭을…"). 본문을 지우는 쪽이 더 큰 오류라 이 갈래는 제거하고, 오인 여지가
# 없는 대답형 머리말만 남긴다. (이번 64회에서 머리말 제거가 발동한 회차는 0건이었다.)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:물론(?:이죠|입니다)[.!,]?|알겠습니다[.!,]?|네[,.!])\s*\n+",
)


def clean_body(text: str) -> str:
    """Strip wrapper syntax the instruction told the model not to add.

    Only decoration is removed, never words: the goal is to measure the characters a reader
    would receive, so a chatty preamble sentence that the model chose to write is stripped
    (it was explicitly forbidden) while the body itself is left byte-for-byte intact.
    """
    body = text or ""
    fenced = _FENCE_RE.match(body.strip())
    if fenced:
        body = fenced.group(1)
    else:
        body = _INLINE_FENCE_RE.sub("", body)
    body = _PREAMBLE_RE.sub("", body)
    body = _HEADING_RE.sub("", body)
    body = _BULLET_RE.sub("", body)
    body = _EMPHASIS_RE.sub("", body)
    # Collapse blank-line runs so paragraph spacing style does not change the count.
    body = re.sub(r"\n{2,}", "\n", body)
    return body.strip()


def char_count(text: str) -> int:
    """Characters as a Korean writer counts them: spaces in, line breaks out.

    Korean length instructions ('300자') are conventionally understood as including spaces,
    and a model cannot control how many line breaks a renderer keeps, so newlines are not
    counted against it.
    """
    return len(text.replace("\n", "").replace("\r", ""))


def score_one(invocation: dict[str, Any], response_text: str) -> dict[str, Any]:
    if invocation.get("infra_error"):
        return {"chars": None, "target": invocation["target"], "ratio": None, "error_pct": None,
                "within_10": False, "within_20": False, "empty": True,
                "failure_reason": "infra", "failure_detail": invocation["infra_error"]}
    body = clean_body(response_text)
    chars = char_count(body)
    target = invocation["target"]
    if chars == 0:
        return {"chars": 0, "target": target, "ratio": None, "error_pct": None,
                "within_10": False, "within_20": False, "empty": True,
                "failure_reason": "empty", "failure_detail": "정제 후 본문 0자"}
    ratio = chars / target
    error_pct = (chars - target) / target * 100
    return {"chars": chars, "target": target, "ratio": ratio, "error_pct": error_pct,
            "within_10": abs(error_pct) <= 10, "within_20": abs(error_pct) <= 20,
            "empty": False, "failure_reason": None, "failure_detail": None}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if not row["empty"]]
    total = len(rows)
    if not scored:
        return {"total": total, "scored": 0, "within_10": 0, "within_20": 0,
                "within_10_rate": 0.0, "within_20_rate": 0.0,
                "median_ratio": None, "median_error_pct": None, "median_chars": None,
                "over": 0, "under": 0}
    ratios = [row["ratio"] for row in scored]
    errors = [row["error_pct"] for row in scored]
    return {
        "total": total,
        "scored": len(scored),
        "within_10": sum(row["within_10"] for row in rows),
        "within_20": sum(row["within_20"] for row in rows),
        "within_10_rate": sum(row["within_10"] for row in rows) / total if total else 0.0,
        "within_20_rate": sum(row["within_20"] for row in rows) / total if total else 0.0,
        "median_ratio": round(median(ratios), 3),
        "median_error_pct": round(median(errors), 1),
        "median_chars": int(median(row["chars"] for row in scored)),
        "over": sum(1 for row in scored if row["error_pct"] > 0),
        "under": sum(1 for row in scored if row["error_pct"] < 0),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_phrasing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_target[row["target"]].append(row)
        by_phrasing[row["phrasing"]].append(row)
        by_pair[(row["target"], row["phrasing"])].append(row)

    elapsed = [row["elapsed_s"] for row in results if isinstance(row.get("elapsed_s"), (int, float))]
    eval_counts = [row["eval_count"] for row in results if isinstance(row.get("eval_count"), int)]
    # Characters produced per generated token — the bridge between the number a person asks
    # for and the unit the model actually budgets in.
    chars_per_token = [row["chars"] / row["eval_count"] for row in results
                       if row.get("chars") and isinstance(row.get("eval_count"), int) and row["eval_count"]]

    return {
        "runs": len(results),
        "infra_errors": sum(bool(row.get("infra_error")) for row in results),
        "overall": _summary(results),
        "by_target": {str(target): _summary(rows) for target, rows in sorted(by_target.items())},
        "by_phrasing": {phrasing: _summary(rows) for phrasing, rows in sorted(by_phrasing.items())},
        "by_target_phrasing": {f"{target}/{phrasing}": _summary(rows)
                               for (target, phrasing), rows in sorted(by_pair.items())},
        "elapsed_s": {"median": round(median(elapsed), 2) if elapsed else None,
                      "total": round(sum(elapsed), 1) if elapsed else None},
        "eval_count": {"median": int(median(eval_counts)) if eval_counts else None},
        "chars_per_token": {"median": round(median(chars_per_token), 2) if chars_per_token else None},
    }


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Plumbing check before the full run — can the harness produce scoreable text at all."""
    reasons: list[str] = []
    infra = sum(bool(row.get("infra_error")) for row in results)
    if results and infra / len(results) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(results)}")
    if results and all(row["empty"] for row in results):
        reasons.append("정제 후 본문이 전부 0자 — 정제 규칙/프롬프트 점검 필요")
    return {"proceed": not reasons, "reasons": reasons, "runs": len(results), "infra_errors": infra}


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for invocation_path in sorted((run_dir / "raw").glob("*-invocation.json")):
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        scored = score_one(invocation, response)
        metrics = invocation.get("api_metrics") or {}
        rows.append({
            "run_id": invocation["run_id"],
            "prompt_id": invocation["prompt_id"],
            "genre": invocation["genre"],
            "phrasing": invocation["phrasing"],
            "elapsed_s": invocation.get("elapsed_s"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["run_id", "prompt_id", "genre", "phrasing", "target", "chars", "ratio",
                   "error_pct", "within_10", "within_20", "empty", "elapsed_s",
                   "prompt_eval_count", "eval_count", "failure_reason", "failure_detail", "infra_error"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    # 3 prompts x 4 targets x 1 phrasing = 12 invocations is exactly the pilot plan.
    if len(rows) == 12:
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    assert char_count("가나다 라마") == 6
    assert char_count("가나\n다라") == 4

    # A fenced answer is measured on its contents, not on the fence.
    assert clean_body("```\n안녕하세요 반갑습니다\n```") == "안녕하세요 반갑습니다"
    # Markdown decoration is stripped; the words survive intact.
    assert clean_body("## 제목\n- **항목** 하나") == "제목\n항목 하나"
    # A forbidden preamble line is removed, the body kept.
    assert clean_body("물론이죠!\n본문입니다") == "본문입니다"
    # Plain prose is untouched.
    assert clean_body("본문 그대로입니다") == "본문 그대로입니다"
    # ★T2 적대검증 회귀 — 정제는 장식만 지우고 본문은 한 글자도 건드리지 않는다.
    assert clean_body("1. 2026년은 중요한 해였습니다.") == "1. 2026년은 중요한 해였습니다."
    assert clean_body("2026. 8. 1. 공지드립니다") == "2026. 8. 1. 공지드립니다"
    assert clean_body("다음은 준비물 목록입니다.\n텐트와 침낭을 챙기세요.") == \
        "다음은 준비물 목록입니다.\n텐트와 침낭을 챙기세요."
    # …기호 불릿은 종전대로 장식으로 본다.
    assert clean_body("*   1월 1일: 휴무") == "1월 1일: 휴무"

    base = {"target": 100, "infra_error": None}
    exact = score_one(base, "가" * 100)
    assert exact["chars"] == 100 and exact["within_10"] and exact["error_pct"] == 0, exact
    short = score_one(base, "가" * 60)
    assert short["chars"] == 60 and not short["within_10"] and not short["within_20"], short
    assert abs(short["error_pct"] + 40) < 1e-9, short
    edge = score_one(base, "가" * 110)
    assert edge["within_10"] and edge["within_20"], edge
    just_out = score_one(base, "가" * 111)
    assert not just_out["within_10"] and just_out["within_20"], just_out
    empty = score_one(base, "```\n\n```")
    assert empty["empty"] and not empty["within_20"], empty
    infra = score_one({"target": 100, "infra_error": "boom"}, "")
    assert infra["empty"] and infra["failure_reason"] == "infra", infra

    rows = [
        {"target": 100, "chars": 100, "ratio": 1.0, "error_pct": 0.0, "within_10": True, "within_20": True, "empty": False},
        {"target": 100, "chars": 60, "ratio": 0.6, "error_pct": -40.0, "within_10": False, "within_20": False, "empty": False},
    ]
    summary = _summary(rows)
    assert summary["within_10"] == 1 and summary["scored"] == 2 and summary["under"] == 1, summary
    print("length_score self-test OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_dir:
        parser.error("--run-dir 또는 --self-test 필요")
    _, aggregate = score_run_dir(args.run_dir)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
