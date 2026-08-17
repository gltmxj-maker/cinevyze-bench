#!/usr/bin/env python3
"""Score the banned-word (negative instruction) compliance benchmark.

The question is narrow on purpose: the model was told "do not write these words", so the only
thing measured is whether those exact stems appear in the answer. Korean attaches particles and
endings to a stem (최고 → 최고의, 최고입니다), so a substring test on the stem is the correct
rule here rather than a whole-word match.

Two honesty problems shape the rest of this scorer:

1. **A word that would never have appeared is not evidence of compliance.** Every prompt is also
   run with no ban at all (the control arm), and inside a ban arm the words further down the pool
   are left unbanned. Both give a base rate to compare the banned rate against.
2. **A model that answers "I avoided the word '최고'" has technically written it.** That is a
   different failure from using the word in the body, so occurrences that sit inside quotation
   marks in a sentence about the instruction are flagged `quoted_meta` and reported as a second,
   looser number — never silently dropped from the primary one.

`clean_body` is imported from `length_score` on purpose: both benchmarks ask "what characters
would a reader actually receive", and two divergent cleaning rules in one repo would be a bug
waiting to happen.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from length_score import clean_body

# Sentence-ish boundaries. Korean sentences end with 다./요. plus the usual punctuation; the split
# only needs to be good enough to keep an instruction-acknowledgement sentence separate from prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTE_CHARS = "\"'‘’“”「」『』"
# Words that mark a sentence as talking *about* the instruction rather than doing the writing task.
_META_MARKERS = ("쓰지", "사용하지", "쓰지도", "피하", "제외", "금지", "빼고", "않았", "않고", "않습니다", "대신")


def _is_quoted(sentence: str, start: int, end: int) -> bool:
    """True when the occurrence is wrapped in quote marks (allowing one particle after it)."""
    before = sentence[max(0, start - 1):start]
    after = sentence[end:end + 3]
    if before and before[-1] in _QUOTE_CHARS:
        return any(char in _QUOTE_CHARS for char in after)
    return False


def find_occurrences(body: str, word: str) -> list[dict[str, Any]]:
    """Every appearance of the stem, each tagged with the sentence it sits in."""
    found: list[dict[str, Any]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(body):
        stripped = sentence.strip()
        if not stripped:
            continue
        for match in re.finditer(re.escape(word), stripped):
            quoted_meta = _is_quoted(stripped, match.start(), match.end()) and any(
                marker in stripped for marker in _META_MARKERS
            )
            found.append({
                "quoted_meta": quoted_meta,
                "snippet": stripped[max(0, match.start() - 20):match.end() + 20],
            })
    return found


def score_one(invocation: dict[str, Any], response: str) -> dict[str, Any]:
    body = clean_body(response)
    pool: list[str] = invocation["banned_pool"]
    shown: list[str] = invocation["banned_shown"]
    words: list[dict[str, Any]] = []
    for rank, word in enumerate(pool):
        occurrences = find_occurrences(body, word)
        body_occurrences = [item for item in occurrences if not item["quoted_meta"]]
        words.append({
            "word": word,
            "rank": rank,
            "shown": word in shown,
            "count": len(occurrences),
            "count_excl_meta": len(body_occurrences),
            "present": bool(occurrences),
            "present_excl_meta": bool(body_occurrences),
            "snippets": [item["snippet"] for item in occurrences[:2]],
        })
    shown_rows = [row for row in words if row["shown"]]
    return {
        "arm": invocation["arm"],
        "position": invocation["position"],
        "ban_count": len(shown),
        "body_chars": len(body),
        "empty": not body,
        "violated": any(row["present"] for row in shown_rows),
        "violated_excl_meta": any(row["present_excl_meta"] for row in shown_rows),
        "violated_words": [row["word"] for row in shown_rows if row["present"]],
        "violation_count": sum(row["count"] for row in shown_rows),
        "words": words,
    }


def _skip(row: dict[str, Any]) -> bool:
    """분모에서 빼야 하는 회차. 절단(창 초과)은 모델 판단이 아니라 하네스 설정 실패라 오답으로 세지 않는다.
    `truncated` 키는 score_run_dir 가 붙이므로, score_one 만 쓰는 자체검사 경로에서는 없을 수 있다."""
    return (bool(row.get("infra_error")) or bool(row.get("truncated"))
            or bool(row.get("missing_response")) or row["empty"])


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _arm_table(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Violation rate grouped by one field, counting runs (not words) as the denominator."""
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "violated": 0, "excl_meta": 0})
    for row in rows:
        if _skip(row) or row["arm"] == "control":
            continue
        bucket = buckets[str(row[key])]
        bucket["runs"] += 1
        bucket["violated"] += int(row["violated"])
        bucket["excl_meta"] += int(row["violated_excl_meta"])
    return {
        name: {
            "runs": value["runs"],
            "violated": value["violated"],
            "violated_rate": _rate(value["violated"], value["runs"]),
            "violated_excl_meta": value["excl_meta"],
            "violated_rate_excl_meta": _rate(value["excl_meta"], value["runs"]),
        }
        for name, value in sorted(buckets.items())
    }


def _arm_word_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-*word* violation rate by arm — the denominator that makes the arms comparable.

    A run in ban5 carries five chances to slip and a run in ban1 carries one, so run-level rates
    rise with the ban list even if the model's per-word behaviour never changed. This table asks
    the question the run-level one cannot: given a word it was told not to write, how often did
    it write it anyway? `expected_run_rate` is what the run-level rate would be if the words were
    independent, so the gap between it and the observed run rate is the part list length explains.
    """
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"words": 0, "present": 0, "counts": []})
    for row in rows:
        if _skip(row) or row["arm"] == "control":
            continue
        bucket = buckets[row["arm"]]
        shown_here = [word_row for word_row in row["words"] if word_row["shown"]]
        bucket["counts"].append(len(shown_here))
        bucket["words"] += len(shown_here)
        bucket["present"] += sum(int(word_row["present_excl_meta"]) for word_row in shown_here)
    table: dict[str, Any] = {}
    for arm, value in sorted(buckets.items()):
        per_word = _rate(value["present"], value["words"])
        counts = value["counts"]
        # 회차마다 금지어 수가 다를 수 있으므로 평균 개수로 한 번 계산하지 않고 회차별 기대값을 평균낸다.
        expected = (sum(1 - (1 - per_word) ** k for k in counts) / len(counts)
                    if per_word is not None and counts else None)
        table[arm] = {
            "runs": len(counts),
            "banned_words": value["words"],
            "violations": value["present"],
            "per_word_rate": per_word,
            "expected_run_rate": round(expected, 4) if expected is not None else None,
        }
    return table


def _word_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-word appearance rate in three states: no ban at all, not banned this run, banned.

    The three columns are the whole point of the control design — "the model did not write 최고"
    only means something next to "how often it writes 최고 when nobody stopped it".
    """
    buckets: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"runs": 0, "present": 0})
    for row in rows:
        if _skip(row):
            continue
        for word_row in row["words"]:
            if row["arm"] == "control":
                state = "control"
            else:
                state = "banned" if word_row["shown"] else "unbanned_in_ban_arm"
            bucket = buckets[(row["prompt_id"], word_row["word"], state)]
            bucket["runs"] += 1
            bucket["present"] += int(word_row["present_excl_meta"])
    table: dict[str, Any] = {}
    for (prompt_id, word, state), value in buckets.items():
        entry = table.setdefault(f"{prompt_id}/{word}", {})
        entry[state] = {"runs": value["runs"], "present": value["present"],
                        "rate": _rate(value["present"], value["runs"])}
    return dict(sorted(table.items()))


def _state_totals(word_table: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "present": 0})
    for entry in word_table.values():
        for state, value in entry.items():
            totals[state]["runs"] += value["runs"]
            totals[state]["present"] += value["present"]
    return {
        state: {"runs": value["runs"], "present": value["present"],
                "rate": _rate(value["present"], value["runs"])}
        for state, value in sorted(totals.items())
    }


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Plumbing check before the full run — can the harness produce scoreable text at all."""
    reasons: list[str] = []
    if not results:
        # 회차 0건은 '문제 없음'이 아니라 '아무것도 안 돌았음'이다. 통과시키면 빈 런으로 full 이 열린다.
        return {"proceed": False, "reasons": ["회차 0건 — 실행된 시행이 없음"], "runs": 0, "infra_errors": 0}
    infra = sum(bool(row.get("infra_error")) for row in results)
    if results and infra / len(results) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(results)}")
    if results and all(row["empty"] for row in results):
        reasons.append("정제 후 본문이 전부 0자 — 정제 규칙/프롬프트 점검 필요")
    if results and not any(row["arm"] == "control" for row in results):
        reasons.append("통제군 회차 0 — 기저 등장률 없이는 준수율을 해석할 수 없음")
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
            "done_reason": metrics.get("done_reason"),
            # 창이 다 차서 끊긴 회차. 뒤가 잘린 글은 금지어를 쓸 기회 자체가 줄어 위반율을 아래로
            # 끌어내리므로, 본문이 남아 있어도 분모에서 뺀다(하네스 설정 실패이지 모델 판단이 아니다).
            "truncated": metrics.get("done_reason") == "length",
            "missing_response": not response_path.exists(),
            "elapsed_s": invocation.get("elapsed_s"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    scoreable = [row for row in rows if not row.get("infra_error") and not row["empty"]]
    ban_rows = [row for row in scoreable if row["arm"] != "control"]
    word_table = _word_table(rows)
    aggregate: dict[str, Any] = {
        "runs": len(rows),
        "scoreable": len(scoreable),
        "truncated": sum(1 for row in rows if row["truncated"]),
        "missing_response": sum(1 for row in rows if row["missing_response"]),
        "control_runs": sum(1 for row in scoreable if row["arm"] == "control"),
        "ban_runs": len(ban_rows),
        "overall": {
            "violated": sum(int(row["violated"]) for row in ban_rows),
            "violated_rate": _rate(sum(int(row["violated"]) for row in ban_rows), len(ban_rows)),
            "violated_excl_meta": sum(int(row["violated_excl_meta"]) for row in ban_rows),
            "violated_rate_excl_meta": _rate(
                sum(int(row["violated_excl_meta"]) for row in ban_rows), len(ban_rows)),
        },
        "by_arm": _arm_table(rows, "arm"),
        "by_arm_per_word": _arm_word_table(rows),
        "by_position": _arm_table(rows, "position"),
        "by_prompt": _arm_table(rows, "prompt_id"),
        "by_word": word_table,
        "state_totals": _state_totals(word_table),
    }
    decision = pilot_decision(rows)
    aggregate["pilot_decision"] = decision
    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "pilot_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, aggregate


def self_test() -> None:
    # A stem matches through Korean particles and endings — that is the intended rule.
    assert find_occurrences("이건 최고의 제품입니다", "최고")
    assert find_occurrences("정말 최고입니다", "최고")
    assert not find_occurrences("이건 좋은 제품입니다", "최고")

    # An acknowledgement sentence that quotes the banned word is flagged, not dropped.
    meta = find_occurrences("'최고'라는 표현은 쓰지 않았습니다.", "최고")
    assert meta and meta[0]["quoted_meta"] is True
    # Quoting without any instruction talk is ordinary prose, not meta.
    body = find_occurrences("사장님은 '최고'라고 적힌 간판을 걸었다.", "최고")
    assert body and body[0]["quoted_meta"] is False
    # Using the word plainly inside a sentence that also mentions avoidance is still a body use.
    mixed = find_occurrences("최고라는 말을 쓰지 않으려 했지만 최고입니다.", "최고")
    assert mixed and all(item["quoted_meta"] is False for item in mixed)

    invocation = {
        "banned_pool": ["최고", "완벽", "특별"],
        "banned_shown": ["최고"],
        "arm": "ban1",
        "position": "front",
    }
    clean = score_one(invocation, "완벽한 소리를 담았습니다.")
    assert clean["violated"] is False and clean["ban_count"] == 1
    # The unbanned pool word that did appear is still recorded — that is the within-run base rate.
    assert [row["present"] for row in clean["words"]] == [False, True, False]

    dirty = score_one(invocation, "최고의 소리입니다.")
    assert dirty["violated"] is True and dirty["violated_words"] == ["최고"]

    excused = score_one(invocation, "'최고'라는 단어는 쓰지 않았습니다. 맑은 소리를 담았습니다.")
    assert excused["violated"] is True and excused["violated_excl_meta"] is False

    # ★T2 적대검증(2026-08-16) 회귀 고정 ─────────────────────────────────────────
    # 부분일치는 파생형을 잡으라고 고른 규칙이다("어떤 형태로도 쓰지 말라"고 지시했으므로).
    # 뜻이 반대인 합성어까지 잡히는 건 이 규칙의 알려진 대가다. 2026-08-16 런 57회 전 응답을
    # 전수 스캔한 결과 그런 사례는 0건이었고(전부 정상 용법), 그래서 규칙을 그대로 둔다.
    assert find_occurrences("간편식을 챙기세요", "간편")      # 의도된 적중(파생어)
    assert find_occurrences("불완벽한 초안입니다", "완벽")     # 알려진 대가(반대 뜻 합성어)

    # 회차 0건은 통과가 아니다 — 빈 런디렉터리로 full 이 열리던 구멍.
    assert pilot_decision([])["proceed"] is False

    # 절단 회차는 모든 분모에서 빠진다(본문이 남아 있어도).
    assert _skip({"infra_error": None, "truncated": True, "empty": False}) is True
    assert _skip({"infra_error": None, "truncated": False, "empty": False}) is False

    # 팔 안에서 금지어 개수가 갈려도 기대 위반율이 개수별로 계산된다.
    def _row(shown, present):
        words = [{"word": f"w{i}", "rank": i, "shown": True, "count": 0, "count_excl_meta": 0,
                  "present": False, "present_excl_meta": i < present, "snippets": []}
                 for i in range(shown)]
        return {"arm": "banX", "position": "front", "empty": False, "infra_error": None,
                "truncated": False, "words": words}
    table = _arm_word_table([_row(5, 1), _row(1, 0)])["banX"]
    assert table["banned_words"] == 6 and table["violations"] == 1
    per_word = 1 / 6
    expected = ((1 - (1 - round(per_word, 4)) ** 5) + (1 - (1 - round(per_word, 4)) ** 1)) / 2
    assert table["expected_run_rate"] == round(expected, 4), table
    print("banword_score self-test OK")


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
