#!/usr/bin/env python3
"""Score the Korean large-number unit conversion benchmark.

Korean groups digits by four (만·억·조) while the comma convention groups them by three. Every
item here has exactly one right answer, so scoring is string equality after a stated normalisation
— no tolerance, no partial credit.

The interesting part is not the pass rate but the shape of the failures, so a wrong answer is
also classified: did the model land on the right quantity and only write it in the wrong form
(`value_ok_format_off`), slip the whole thing by a factor of ten or ten-thousand (`off_by_unit`),
or produce a different number entirely (`wrong_value`)? Those three tell a reader where to check.

The formatting and parsing helpers live here rather than in the runner because the expected
answer must be produced by the same rule the scorer grades with — a hand-typed gold column is
exactly where a silent mismatch between prompt and grader would hide.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from length_score import clean_body

TASKS = ("digit_to_ko", "ko_to_digit", "comma", "won_round")
_UNIT_VALUE = {None: 1, "만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12, "경": 10 ** 16}
_UNIT_ORDER = ("", "만", "억", "조", "경")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:답|정답|결과)\s*[:：]\s*")
_NUMBER_TOKEN_RE = re.compile(r"(\d+)(경|조|억|만)?")


def format_korean_units(value: int) -> str:
    """1234567890 → '12억 3456만 7890'. Groups of four, zero groups dropped."""
    if value < 0:
        raise ValueError("음수는 이 벤치의 범위 밖")
    if value == 0:
        return "0"
    groups: list[int] = []
    remainder = value
    while remainder:
        groups.append(remainder % 10 ** 4)
        remainder //= 10 ** 4
    if len(groups) > len(_UNIT_ORDER):
        raise ValueError("경 단위를 넘는 값은 이 벤치의 범위 밖")
    parts = [f"{group}{_UNIT_ORDER[index]}" for index, group in reversed(list(enumerate(groups))) if group]
    return " ".join(parts)


def parse_korean_units(text: str) -> int | None:
    """Inverse of format_korean_units, tolerant of spacing and commas but nothing else.

    Anything the grammar does not cover — 천/백/십 spelled as units, stray words, a trailing
    sentence — returns None and is scored as unparsed rather than guessed at.
    """
    compact = re.sub(r"[\s,]", "", text or "")
    if not compact:
        return None
    total = 0
    consumed = 0
    for match in _NUMBER_TOKEN_RE.finditer(compact):
        if match.start() != consumed:
            return None
        total += int(match.group(1)) * _UNIT_VALUE[match.group(2)]
        consumed = match.end()
    if consumed != len(compact):
        return None
    return total


_HANGUL_DIGIT = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
                 "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_SMALL_UNIT = {"십": 10, "백": 100, "천": 1000}
_NUMERAL_TOKEN_RE = re.compile(r"\d+|[영공일이삼사오육륙칠팔구]|[십백천]|[만억조경]")


def parse_korean_numeral(text: str) -> int | None:
    """Read a number written any of the ways a model actually writes one.

    `parse_korean_units` covers the format the prompt asked for. This is the wider grammar —
    Hangul numerals (`십이만 삼천 사백 오십육`), 천/백/십 inside a group, and mixes of the two
    (`1억 2천345만`). It exists so that an answer whose *quantity* is right but whose *form* is
    wrong is scored as a formatting failure rather than disappearing into 'could not read it'.
    Anything with a stray word or symbol still returns None: a guess would be worse than a gap.
    """
    compact = re.sub(r"[\s,]", "", text or "")
    if not compact:
        return None
    strict = parse_korean_units(compact)
    if strict is not None:
        return strict
    total = section = number = 0
    consumed = 0
    seen_unit = False
    for match in _NUMERAL_TOKEN_RE.finditer(compact):
        if match.start() != consumed:
            return None
        consumed = match.end()
        token = match.group()
        if token.isdigit():
            number = number * 10 ** len(token) + int(token)
        elif token in _HANGUL_DIGIT:
            number = number * 10 + _HANGUL_DIGIT[token]
        elif token in _SMALL_UNIT:
            section += (number or 1) * _SMALL_UNIT[token]
            number = 0
            seen_unit = True
        else:
            total += (section + number or 1) * _UNIT_VALUE[token]
            section = number = 0
            seen_unit = True
    if consumed != len(compact) or not seen_unit:
        return None
    return total + section + number


def task_input(task: str, value: int) -> str:
    """The string the prompt shows the model."""
    if task == "digit_to_ko":
        return str(value)
    if task == "ko_to_digit":
        return format_korean_units(value)
    if task == "comma":
        return str(value)
    if task == "won_round":
        return f"{value}원"
    raise ValueError(f"알 수 없는 task: {task}")


def expected_answer(task: str, value: int) -> str:
    if task == "digit_to_ko":
        return format_korean_units(value)
    if task == "ko_to_digit":
        return str(value)
    if task == "comma":
        return f"{value:,}"
    if task == "won_round":
        return f"{value // 10 ** 4}만"
    raise ValueError(f"알 수 없는 task: {task}")


def expected_value(task: str, value: int) -> int:
    """The quantity the answer denotes — for won_round the truncated amount, not the raw one."""
    return value // 10 ** 4 * 10 ** 4 if task == "won_round" else value


def normalise(task: str, text: str) -> str:
    """Strip only what the task treats as insignificant. Commas stay significant in `comma`."""
    cleaned = (text or "").strip()
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.strip().strip(".。")
    if task == "comma":
        return re.sub(r"\s", "", cleaned)
    cleaned = re.sub(r"[\s,]", "", cleaned)
    if task == "won_round":
        cleaned = re.sub(r"원$", "", cleaned)
    return cleaned


def extract_answer(response: str) -> tuple[str, bool]:
    """Pick the answer line under a stated rule, and report whether the model wrote extra lines.

    The prompt asks for the answer alone on one line. When a model ignores that, an explicitly
    labelled line ('답: …') wins; otherwise the first non-empty line does, because that is what a
    reader sees first. The `multiline` flag keeps that disobedience visible instead of hiding it.
    """
    lines = [line.strip() for line in clean_body(response).splitlines() if line.strip()]
    if not lines:
        return "", False
    for line in lines:
        if _ANSWER_PREFIX_RE.match(line):
            return line, len(lines) > 1
    return lines[0], len(lines) > 1


def observed_value(task: str, normalised: str) -> int | None:
    if not normalised:
        return None
    stripped = re.sub(r"[\s,]", "", normalised)
    if task == "won_round":
        stripped = re.sub(r"원$", "", stripped)
    if stripped.isdigit():
        return int(stripped)
    return parse_korean_numeral(stripped)


def classify(task: str, value: int, answer_text: str) -> dict[str, Any]:
    expected_text = expected_answer(task, value)
    normalised = normalise(task, answer_text)
    exact = normalised == normalise(task, expected_text)
    got_value = observed_value(task, normalised)
    want_value = expected_value(task, value)
    if exact:
        verdict = "correct"
    elif got_value is None:
        verdict = "unparsed"
    elif got_value == want_value:
        verdict = "value_ok_format_off"
    elif task == "won_round" and got_value == value != want_value:
        # Echoing the amount without truncating is its own failure, not a slipped unit.
        verdict = "not_truncated"
    elif got_value and want_value and any(
        got_value == want_value * 10 ** power or got_value * 10 ** power == want_value
        # 이 벤치는 6~14자리를 다루므로 한 자리 밀림부터 조 단위 통째 밀림까지 전부 '단위 밀림'이다.
        # 목록이 짧으면 진짜 단위 슬립이 '값 오답'으로 섞여 실패의 모양을 잘못 발표하게 된다.
        for power in range(1, 13)
    ):
        verdict = "off_by_unit"
    else:
        verdict = "wrong_value"
    return {
        "expected": expected_text,
        "answer": answer_text,
        "normalised": normalised,
        "observed_value": got_value,
        "expected_value": want_value,
        "correct": exact,
        "verdict": verdict,
    }


def score_one(invocation: dict[str, Any], response: str) -> dict[str, Any]:
    answer_text, multiline = extract_answer(response)
    scored = classify(invocation["task"], invocation["value"], answer_text)
    return {"multiline": multiline, "empty": not answer_text, **scored}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "correct": 0, "verdicts": defaultdict(int)})
    for row in rows:
        if row.get("infra_error") or row.get("truncated") or row.get("missing_response"):
            continue
        bucket = buckets[str(row[key])]
        bucket["runs"] += 1
        bucket["correct"] += int(row["correct"])
        bucket["verdicts"][row["verdict"]] += 1
    return {
        name: {
            "runs": value["runs"],
            "correct": value["correct"],
            "accuracy": _rate(value["correct"], value["runs"]),
            "verdicts": dict(sorted(value["verdicts"].items())),
        }
        for name, value in sorted(buckets.items())
    }


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Plumbing check before the full run — can the harness produce scoreable answers at all."""
    reasons: list[str] = []
    if not results:
        return {"proceed": False, "reasons": ["회차 0건 — 실행된 시행이 없음"], "runs": 0, "infra_errors": 0}
    infra = sum(bool(row.get("infra_error")) for row in results)
    if results and infra / len(results) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(results)}")
    if results and all(row["empty"] for row in results):
        reasons.append("추출된 답이 전부 빈 문자열 — 추출 규칙/프롬프트 점검 필요")
    if any(row.get("missing_response") for row in results):
        reasons.append("응답 파일이 없는 회차 존재 — 하네스 사고이지 모델의 빈 답이 아님")
    truncated = sum(1 for row in results if row.get("truncated"))
    if results and truncated / len(results) > 0.10:
        reasons.append(f"컨텍스트 창 초과로 답이 잘린 회차 {truncated}/{len(results)} — --num-ctx를 키우세요")
    if results and all(row["verdict"] == "unparsed" for row in results if not row.get("infra_error")):
        reasons.append("전 회차 unparsed — 출력 형식 지시가 통하지 않음")
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
            "item_id": invocation["item_id"],
            "task": invocation["task"],
            "digits": invocation["digits"],
            "value": invocation["value"],
            "elapsed_s": invocation.get("elapsed_s"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "infra_error": invocation.get("infra_error"),
            "done_reason": metrics.get("done_reason"),
            # 창이 다 차서 끊긴 회차. 답은 생성의 마지막에 오므로 잘렸다면 최종 답이 아니고, 본문이
            # 일부 남아 있어도 오답으로 셀 수 없다 — '틀렸다'가 아니라 '못 물어봤다'라서 분모에서 뺀다.
            "truncated": metrics.get("done_reason") == "length",
            # 응답 파일이 없는 것은 모델의 빈 답이 아니라 하네스 사고다. 오답으로 섞이면 안 된다.
            "missing_response": not response_path.exists(),
            **scored,
        })

    scoreable = [row for row in rows if not row.get("infra_error")
                 and not row["truncated"] and not row["missing_response"]]
    verdicts: dict[str, int] = defaultdict(int)
    for row in scoreable:
        verdicts[row["verdict"]] += 1
    aggregate: dict[str, Any] = {
        "runs": len(rows),
        "scoreable": len(scoreable),
        "truncated": sum(1 for row in rows if row["truncated"]),
        "missing_response": sum(1 for row in rows if row["missing_response"]),
        "overall": {
            "correct": sum(int(row["correct"]) for row in scoreable),
            "accuracy": _rate(sum(int(row["correct"]) for row in scoreable), len(scoreable)),
            "verdicts": dict(sorted(verdicts.items())),
            "multiline": sum(int(row["multiline"]) for row in scoreable),
            "multiline_rate": _rate(sum(int(row["multiline"]) for row in scoreable), len(scoreable)),
        },
        "by_task": _group(rows, "task"),
        "by_digits": _group(rows, "digits"),
        "misses": [
            {"item_id": row["item_id"], "task": row["task"], "digits": row["digits"],
             "input": task_input(row["task"], row["value"]), "expected": row["expected"],
             "answer": row["answer"], "verdict": row["verdict"]}
            for row in scoreable if not row["correct"]
        ],
    }
    decision = pilot_decision(rows)
    aggregate["pilot_decision"] = decision
    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "pilot_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, aggregate


def self_test() -> None:
    assert format_korean_units(1234567890) == "12억 3456만 7890"
    assert format_korean_units(100000000) == "1억"
    assert format_korean_units(1000500000) == "10억 50만"
    assert format_korean_units(10000000050000) == "10조 5만"
    assert format_korean_units(123456789012) == "1234억 5678만 9012"
    assert format_korean_units(900000) == "90만"

    # Round-trip: the parser must recover every value the formatter can produce.
    for value in (123456, 900000, 12345678, 10000000, 1234567890, 1000500000,
                  123456789012, 100000000000, 12345678901234, 10000000050000):
        assert parse_korean_units(format_korean_units(value)) == value, value
    assert parse_korean_units("12억3456만7890") == 1234567890   # spacing is insignificant
    assert parse_korean_units("1억 2천만") is None               # 천 is outside the grammar
    assert parse_korean_units("약 1억") is None                  # stray words are not guessed at

    # 넓은 문법 — 한글 수사와 천/백/십, 그리고 둘을 섞은 표기까지 값으로 되읽는다.
    assert parse_korean_numeral("십이만 삼천 사백 오십육") == 123456
    assert parse_korean_numeral("1억 2천345만") == 123450000
    assert parse_korean_numeral("일억 팔천오십만") == 180500000
    assert parse_korean_numeral("12억 3456만 7890") == 1234567890   # 엄격 문법도 그대로 통과
    assert parse_korean_numeral("10억 ㏓만") is None                 # 읽을 수 없으면 추측하지 않는다
    assert parse_korean_numeral("1조 2345만 6789번") is None         # 군더더기 단어도 마찬가지

    assert expected_answer("comma", 1234567890) == "1,234,567,890"
    assert expected_answer("won_round", 12345678) == "1234만"
    assert task_input("ko_to_digit", 1234567890) == "12억 3456만 7890"

    # Exact answers pass on every task.
    for task, value, answer in (
        ("digit_to_ko", 1234567890, "12억 3456만 7890"),
        ("ko_to_digit", 1234567890, "1234567890"),
        ("comma", 1234567890, "1,234,567,890"),
        ("won_round", 12345678, "1234만 원"),
    ):
        assert classify(task, value, answer)["verdict"] == "correct", (task, answer)

    # Right quantity, wrong presentation.
    assert classify("digit_to_ko", 1234567890, "12억3456만7890")["verdict"] == "correct"
    assert classify("comma", 1234567890, "1234567890")["verdict"] == "value_ok_format_off"
    assert classify("won_round", 12345678, "12340000")["verdict"] == "value_ok_format_off"
    # A unit slip is the failure this benchmark exists to name.
    assert classify("ko_to_digit", 1234567890, "123456789")["verdict"] == "off_by_unit"
    # Handing back the amount untouched is 'did not truncate', a separate failure from a unit slip.
    assert classify("won_round", 12345678, "12345678")["verdict"] == "not_truncated"
    assert classify("won_round", 12345678, "1234")["verdict"] == "off_by_unit"      # 단위 표기 누락
    # Everything else.
    assert classify("ko_to_digit", 1234567890, "1234000000")["verdict"] == "wrong_value"
    # 천 표기는 넓은 문법이 읽어내므로 '해독 불가'가 아니라 값 오답으로 잡힌다.
    # (3천456만 = 3456만으로 같은 값이고, 오답인 이유는 끝의 7890이 빠졌기 때문이다.)
    assert classify("digit_to_ko", 1234567890, "12억 3천456만")["verdict"] == "wrong_value"
    assert classify("digit_to_ko", 123456, "십이만 삼천 사백 오십육")["verdict"] == "value_ok_format_off"
    assert classify("digit_to_ko", 1234567890, "12억 3456만 ㏓")["verdict"] == "unparsed"

    # The answer line is taken under the stated rule, and extra lines stay visible.
    assert extract_answer("1,234,567,890") == ("1,234,567,890", False)
    assert extract_answer("계산해 보겠습니다.\n답: 1,234,567,890") == ("답: 1,234,567,890", True)
    assert extract_answer("1,234,567,890\n이렇게 됩니다.") == ("1,234,567,890", True)

    # ★T2 적대검증(2026-08-16) 회귀 고정 — 회차 0건은 통과가 아니다.
    assert pilot_decision([])["proceed"] is False
    # '답:' 접두가 붙은 줄도 정규화 뒤 정답으로 잡힌다(추출 규칙과 채점 규칙이 어긋나지 않는다).
    assert classify("comma", 1234567890, "답: 1,234,567,890")["verdict"] == "correct"
    # 조 단위 통째 밀림도 '값 오답'이 아니라 '단위 밀림'이다.
    assert classify("digit_to_ko", 1000000000000, "1")["verdict"] == "off_by_unit"
    assert classify("ko_to_digit", 1234567890, "1234567890000000")["verdict"] == "off_by_unit"
    # 절단이 10%를 넘으면 full 을 막는다(창 부족을 정확도 오답으로 발표하지 않기 위해).
    truncated_rows = [{"empty": True, "truncated": True, "verdict": "unparsed", "infra_error": None}] + [
        {"empty": False, "truncated": False, "verdict": "correct", "infra_error": None} for _ in range(7)]
    assert pilot_decision([{"empty": False, "truncated": False, "verdict": "correct",
                            "infra_error": None, "missing_response": True}])["proceed"] is False
    assert pilot_decision(truncated_rows)["proceed"] is False
    print("numunit_score self-test OK")


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
