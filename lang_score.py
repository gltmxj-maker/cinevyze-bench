#!/usr/bin/env python3
"""Score the Korean-vs-English instruction-language benchmark.

Parsing and correctness reuse format_score's JSON path unchanged, so the two benchmarks
cannot drift apart on what counts as a success. What this module adds is the language
axis: paired per-case comparison, output-language contamination (Korean values answered
in English), and prompt/response token counts from the API metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import median
from typing import Any

from format_score import score_response, wilson_interval

LANGS = ("ko", "en")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _expects_hangul(value: Any) -> bool:
    return isinstance(value, str) and bool(_HANGUL_RE.search(value))


def contamination(parsed: Any, expected: dict[str, Any]) -> list[str]:
    """Fields whose gold value is Korean but whose answer carries no Hangul at all.

    Only fields that should be Korean are checked, and only a total absence of Hangul
    combined with Latin letters counts — a partially wrong Korean value is an ordinary
    value error, not a language switch.
    """
    if not isinstance(parsed, dict):
        return []
    switched = []
    for field, gold in expected.items():
        if not _expects_hangul(gold):
            continue
        actual = parsed.get(field)
        if not isinstance(actual, str):
            continue
        if not _HANGUL_RE.search(actual) and _LATIN_RE.search(actual):
            switched.append(field)
    return switched


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    success = sum(bool(row[key]) for row in rows)
    low, high = wilson_interval(success, total)
    return {"success": success, "total": total, "rate": success / total if total else 0.0,
            "wilson_low": low, "wilson_high": high}


def _token_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if isinstance(row.get(key), int)]
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {"n": len(values), "median": median(values),
            "mean": round(sum(values) / len(values), 1), "min": min(values), "max": max(values)}


def paired_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the two arms case by case.

    Aggregate rates alone cannot tell whether the languages disagree on the same inputs
    or merely fail equally often on different ones, so the pairs are counted explicitly.
    """
    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        arms = by_case[(row["task"], row["case_id"])]
        if row["lang"] in arms:
            raise ValueError(
                f"중복 시행: {row['task']}/{row['case_id']}/{row['lang']} — "
                "덮어쓰면 실패 기록이 조용히 사라진다"
            )
        arms[row["lang"]] = row
    both = 0
    ko_only = 0
    en_only = 0
    neither = 0
    incomplete = 0
    disagreements: list[dict[str, Any]] = []
    for (task, case_id), arms in sorted(by_case.items()):
        if set(arms) != set(LANGS):
            incomplete += 1
            continue
        ko_ok = bool(arms["ko"]["success"])
        en_ok = bool(arms["en"]["success"])
        if ko_ok and en_ok:
            both += 1
        elif ko_ok:
            ko_only += 1
        elif en_ok:
            en_only += 1
        else:
            neither += 1
        if ko_ok != en_ok:
            winner = "ko" if ko_ok else "en"
            loser = "en" if ko_ok else "ko"
            disagreements.append({
                "task": task, "case_id": case_id, "winner": winner,
                "loser_failure_reason": arms[loser]["failure_reason"],
                "loser_failure_detail": arms[loser]["failure_detail"],
            })
    return {"pairs": both + ko_only + en_only + neither, "incomplete_pairs": incomplete,
            "both_success": both, "ko_only": ko_only, "en_only": en_only, "neither": neither,
            "mcnemar": mcnemar_exact(ko_only, en_only),
            "disagreements": disagreements}


def mcnemar_exact(ko_only: int, en_only: int) -> dict[str, Any]:
    """Exact McNemar test on the discordant pairs.

    The two arms answer the *same* cases, so the overall rates are not independent samples
    and their Wilson intervals will overlap even when a real difference exists — and, worse
    for honesty here, a raw rate gap can look decisive when only a handful of cases actually
    disagree. Only the discordant pairs carry information about which language is better.
    """
    discordant = ko_only + en_only
    if discordant == 0:
        return {"discordant": 0, "ko_only": 0, "en_only": 0, "p_value": 1.0,
                "significant_at_05": False,
                "interpretation": "불일치 쌍 없음 — 두 언어가 같은 사례에서 같은 결과"}
    smaller = min(ko_only, en_only)
    tail = sum(comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
    p_value = min(1.0, 2 * tail)
    significant = p_value < 0.05
    return {
        "discordant": discordant, "ko_only": ko_only, "en_only": en_only,
        "p_value": round(p_value, 4), "significant_at_05": significant,
        "interpretation": (
            f"불일치 {discordant}쌍(ko만 {ko_only} · en만 {en_only}) → p={p_value:.4f}. "
            + ("유의미한 차이." if significant else
               "이 표본에서는 우연과 구별되지 않음 — 성공률 격차를 '영어가 낫다'로 읽으면 안 된다.")
        ),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_lang: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_lang[row["lang"]].append(row)
        by_task_lang[(row["task"], row["lang"])].append(row)

    aggregate: dict[str, Any] = {"total": len(results), "langs": {}, "task_langs": {}}
    for lang in LANGS:
        rows = by_lang.get(lang, [])
        entry = {metric: _metric_summary(rows, metric) for metric in ("parse", "schema", "semantic", "success")}
        entry["failure_counts"] = dict(Counter(row["failure_reason"] for row in rows if row["failure_reason"]))
        # Only parsed responses can be inspected for output language, so the denominator is
        # the parsed rows. Counting unparsed ones as "clean" would hand a lower contamination
        # score to whichever arm failed to produce JSON more often.
        parsed_rows = [row for row in rows if row.get("parse")]
        entry["contaminated"] = _metric_summary(
            [{**row, "_c": bool(row["contaminated_fields"])} for row in parsed_rows], "_c"
        )
        entry["prompt_tokens"] = _token_summary(rows, "prompt_eval_count")
        entry["response_tokens"] = _token_summary(rows, "eval_count")
        entry["elapsed_s"] = _token_summary(rows, "elapsed_s")
        aggregate["langs"][lang] = entry
    for (task, lang), rows in sorted(by_task_lang.items()):
        aggregate["task_langs"][f"{task}/{lang}"] = {
            metric: _metric_summary(rows, metric) for metric in ("parse", "schema", "semantic", "success")
        }
    aggregate["paired"] = paired_comparison(results)
    aggregate["reading_guide"] = [
        "langs.*.success.wilson_* 는 각 팔을 따로 본 구간이다. 두 팔은 같은 사례를 풀었으므로 "
        "독립 표본이 아니다 — 구간이 겹치는지 여부로 '차이 없음'을 주장하지 마라.",
        "언어 간 우열 판단의 근거는 paired.mcnemar 하나뿐이다(불일치 쌍만 정보를 갖는다).",
        "contaminated 의 분모는 파싱 성공 행이다(파싱 실패는 출력 언어를 볼 수 없다).",
    ]
    return aggregate


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {lang: sum(row["success"] for row in results if row["lang"] == lang) for lang in LANGS}
    totals = {lang: sum(1 for row in results if row["lang"] == lang) for lang in LANGS}
    reasons: list[str] = []
    if all(totals[lang] == 9 and counts[lang] == 9 for lang in LANGS):
        reasons.append("ceiling_all_9_of_9")
    if all(totals[lang] == 9 and counts[lang] == 0 for lang in LANGS):
        reasons.append("floor_all_0_of_9")
    infra = sum(bool(row.get("infra_error")) for row in results)
    if results and infra / len(results) > 0.10:
        reasons.append("infra_errors_over_10_percent")
    return {"proceed": not reasons, "success_counts": counts, "totals": totals,
            "infra_errors": infra, "reasons": reasons}


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for invocation_path in sorted((run_dir / "raw").glob("*-invocation.json")):
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        if not invocation.get("infra_error") and not response_path.exists():
            # A missing response file is a harness failure. Passing "" to the scorer would
            # silently rebrand it as the model producing unparseable output.
            invocation["infra_error"] = f"응답 파일 없음: {invocation['response_file']}"
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        scored = score_response(
            invocation["format"], response, invocation["fields"], invocation["expected"]
        ) if not invocation.get("infra_error") else {
            "parse": False, "schema": False, "semantic": False, "success": False,
            "failure_reason": "infra", "failure_detail": invocation["infra_error"], "parsed": None,
        }
        metrics = invocation.get("api_metrics") or {}
        rows.append({
            "run_id": invocation["run_id"],
            "task": invocation["task"],
            "case_id": invocation["case_id"],
            "lang": invocation["lang"],
            "format": invocation["format"],
            "elapsed_s": invocation.get("elapsed_s"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "infra_error": invocation.get("infra_error"),
            "contaminated_fields": contamination(scored.get("parsed"), invocation["expected"]),
            **scored,
        })

    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["run_id", "task", "case_id", "lang", "elapsed_s", "prompt_eval_count", "eval_count",
                   "parse", "schema", "semantic", "success", "failure_reason", "failure_detail",
                   "contaminated_fields", "infra_error"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out["contaminated_fields"] = ",".join(row["contaminated_fields"])
            writer.writerow(out)

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(rows) == 18:
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    expected = {"id": "O01", "vendor": "한빛상사", "quantity": 120}
    assert contamination({"id": "O01", "vendor": "Hanbit Trading", "quantity": 120}, expected) == ["vendor"]
    assert contamination({"id": "O01", "vendor": "한빛상사", "quantity": 120}, expected) == []
    # A wrong-but-Korean value is a value error, not a language switch.
    assert contamination({"id": "O01", "vendor": "한빛", "quantity": 120}, expected) == []
    # id is Latin in the gold answer, so an all-Latin id must never count as contamination.
    assert contamination({"id": "O01", "vendor": "한빛상사", "quantity": 120}, {"id": "O01"}) == []
    assert contamination(None, expected) == []
    assert contamination({"vendor": 5}, expected) == []
    # Mixed script keeps the Korean anchor, so it is not a switch.
    assert contamination({"vendor": "한빛상사 Co."}, expected) == []

    rows = [
        {"task": "FMT-01", "case_id": "Q01", "lang": "ko", "success": True,
         "failure_reason": None, "failure_detail": None},
        {"task": "FMT-01", "case_id": "Q01", "lang": "en", "success": False,
         "failure_reason": "value", "failure_detail": "오답 필드: category"},
        {"task": "FMT-01", "case_id": "Q02", "lang": "ko", "success": False,
         "failure_reason": "syntax", "failure_detail": "x"},
        {"task": "FMT-01", "case_id": "Q02", "lang": "en", "success": True,
         "failure_reason": None, "failure_detail": None},
        {"task": "FMT-01", "case_id": "Q03", "lang": "ko", "success": True,
         "failure_reason": None, "failure_detail": None},
        {"task": "FMT-01", "case_id": "Q03", "lang": "en", "success": True,
         "failure_reason": None, "failure_detail": None},
        {"task": "FMT-01", "case_id": "Q04", "lang": "ko", "success": False,
         "failure_reason": "value", "failure_detail": "y"},
    ]
    paired = paired_comparison(rows)
    assert paired["pairs"] == 3 and paired["incomplete_pairs"] == 1
    assert paired["both_success"] == 1 and paired["ko_only"] == 1 and paired["en_only"] == 1
    assert paired["neither"] == 0
    assert [d["winner"] for d in paired["disagreements"]] == ["ko", "en"]
    assert paired["disagreements"][0]["loser_failure_reason"] == "value"

    # A 1-vs-5 split is the actual shape of this run: it must NOT read as significant.
    assert mcnemar_exact(1, 5)["p_value"] > 0.05
    assert not mcnemar_exact(1, 5)["significant_at_05"]
    assert mcnemar_exact(0, 12)["significant_at_05"]
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    assert mcnemar_exact(3, 3)["p_value"] == 1.0
    # Symmetric in its arguments: neither language is privileged by the test itself.
    assert mcnemar_exact(2, 9)["p_value"] == mcnemar_exact(9, 2)["p_value"]

    duplicated = [
        {"task": "FMT-01", "case_id": "Q01", "lang": "ko", "success": True,
         "failure_reason": None, "failure_detail": None},
        {"task": "FMT-01", "case_id": "Q01", "lang": "ko", "success": False,
         "failure_reason": "value", "failure_detail": "z"},
    ]
    try:
        paired_comparison(duplicated)
    except ValueError:
        pass
    else:
        raise AssertionError("중복 시행이 조용히 덮어써졌다")

    assert _token_summary([{"t": 3}, {"t": 5}, {"t": None}], "t") == {
        "n": 2, "median": 4, "mean": 4.0, "min": 3, "max": 5}
    assert _token_summary([], "t")["n"] == 0

    ceiling = [{"lang": lang, "success": True, "infra_error": None} for lang in LANGS for _ in range(9)]
    floor = [{"lang": lang, "success": False, "infra_error": None} for lang in LANGS for _ in range(9)]
    mixed = [{"lang": lang, "success": index < 6, "infra_error": None}
             for lang in LANGS for index in range(9)]
    assert "ceiling_all_9_of_9" in pilot_decision(ceiling)["reasons"]
    assert "floor_all_0_of_9" in pilot_decision(floor)["reasons"]
    assert pilot_decision(mixed)["proceed"]
    infra = [dict(row) for row in mixed]
    for row in infra[:3]:
        row["infra_error"] = "timeout"
    assert "infra_errors_over_10_percent" in pilot_decision(infra)["reasons"]
    print("lang_score self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_dir:
        parser.error("--run-dir is required unless --self-test is used")
    rows, aggregate = score_run_dir(args.run_dir)
    print(json.dumps({"runs": len(rows), "langs": aggregate["langs"], "paired": aggregate["paired"],
                      "pilot_decision": aggregate.get("pilot_decision")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
