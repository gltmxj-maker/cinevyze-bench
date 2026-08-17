#!/usr/bin/env python3
"""Deterministic scorer for the self-grading benchmark.

Two layers are scored separately and must not be confused:
  1) 답이 실제로 맞았나 — answer_aliases 대조(정답키). 이게 진실값이다.
  2) 모델의 채점이 그 진실값과 일치했나 — 여기서 나오는 오류가 이 실험의 관심사다.

거짓 통과(오답을 정답이라고 판정)와 거짓 탈락(정답을 오답이라고 판정)은 사용자가 입는 피해가
전혀 다르므로 끝까지 분리해서 센다. 사람이 점수를 적어 넣을 자리는 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

_VERDICT_RE = re.compile(r"판정\s*[:：]\s*(정답|오답)")
# "판정: 정답이 아닙니다" 같은 부정형은 앞 단어를 그대로 읽으면 정반대로 집계된다.
_NEGATED_RE = re.compile(r"^\s*(?:이|가)?\s*(?:아니|아님|아닙)")


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text)
    folded = re.sub(r"(?<=\d),(?=\d{3})", "", folded)
    folded = re.sub(r"\s+", "", folded)
    return folded.upper()


def contains_value(value: str, normalized_haystack: str) -> bool:
    """숫자 별칭은 앞뒤에 숫자가 붙으면 안 된다 — '2'가 '12'에 걸리면 오답이 정답이 된다
    (T2 적대검증 2026-08-03·현 데이터 변동 0건이지만 다음 테스트셋에서 터질 자리)."""
    needle = normalize(value)
    if not needle:
        return False
    if needle.isdigit():
        return re.search(r"(?<!\d)" + re.escape(needle) + r"(?!\d)", normalized_haystack) is not None
    return needle in normalized_haystack


def answer_is_correct(aliases: list[str], response_text: str) -> tuple[bool, str | None]:
    normalized = normalize(response_text)
    matched = next((alias for alias in aliases if contains_value(alias, normalized)), None)
    return matched is not None, matched


def parse_verdict(text: str) -> str | None:
    """'판정: 정답' 형식을 먼저 찾고, 없으면 마지막에 등장한 판정 단어를 쓴다."""
    body = text or ""
    verdicts = [m for m in _VERDICT_RE.finditer(body)
                if not _NEGATED_RE.match(body[m.end():m.end() + 6])]
    if verdicts:
        return verdicts[-1].group(1)
    # 폴백도 부정형을 걸러야 한다 — "정답이 아닙니다"에서 '정답'만 집으면 판정이 뒤집힌다.
    squeezed = re.sub(r"\s+", "", unicodedata.normalize("NFKC", body))
    last = None
    for m in re.finditer(r"정답|오답", squeezed):
        if _NEGATED_RE.match(squeezed[m.end():m.end() + 6]):
            continue
        last = m.group(0)
    return last


def score_grade(invocation: dict[str, Any], response_text: str) -> dict[str, Any]:
    verdict = parse_verdict(response_text)
    truth = bool(invocation["answer_correct"])
    # 백지 답안은 '틀린 답'이 아니다. 채점 대상이 비어 있는 회차를 거짓 통과의 분모에 넣으면
    # "오답을 통과시켰다"가 "빈칸을 통과시켰다"로 오염된다(2026-08-03 본런에서 실제 1건 발생).
    graded = invocation.get("graded_answer")
    if invocation.get("infra_error"):
        outcome = "infra"
    elif graded is not None and not str(graded).strip():
        outcome = "blank_answer"
    elif verdict is None:
        outcome = "unparsed"
    elif (verdict == "정답") == truth:
        outcome = "agree"
    elif verdict == "정답":
        outcome = "false_pass"
    else:
        outcome = "false_fail"
    return {
        "verdict": verdict,
        "answer_correct": truth,
        "outcome": outcome,
        "agree": outcome == "agree",
        "false_pass": outcome == "false_pass",
        "false_fail": outcome == "false_fail",
        "scored": outcome in {"agree", "false_pass", "false_fail"},
        "response_chars": len((response_text or "").strip()),
        "response_head": (response_text or "").strip()[:160],
    }


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["scored"]]
    wrong_answers = [row for row in scored if not row["answer_correct"]]
    right_answers = [row for row in scored if row["answer_correct"]]
    agree = sum(row["agree"] for row in scored)
    false_pass = sum(row["false_pass"] for row in scored)
    false_fail = sum(row["false_fail"] for row in scored)
    return {
        "total": len(rows),
        "scored": len(scored),
        "unparsed": sum(1 for row in rows if row["outcome"] == "unparsed"),
        "blank_answer": sum(1 for row in rows if row["outcome"] == "blank_answer"),
        "agree": agree,
        "false_pass": false_pass,
        "false_fail": false_fail,
        "agreement_rate": round(agree / len(scored), 4) if scored else None,
        # 분모가 다르다 — 거짓 통과는 '실제 오답' 중에서, 거짓 탈락은 '실제 정답' 중에서 센다.
        "wrong_answers": len(wrong_answers),
        "right_answers": len(right_answers),
        "false_pass_rate": round(false_pass / len(wrong_answers), 4) if wrong_answers else None,
        "false_fail_rate": round(false_fail / len(right_answers), 4) if right_answers else None,
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: _rate(bucket) for name, bucket in sorted(buckets.items())}


def _answer_summary(answers: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in answers:
        buckets.setdefault(row["author_model"], []).append(row)
    summary = {}
    for model, rows in sorted(buckets.items()):
        correct = sum(row["answer_correct"] for row in rows)
        summary[model] = {
            "answers": len(rows),
            "correct": correct,
            "accuracy": round(correct / len(rows), 4) if rows else None,
        }
    return summary


def aggregate_results(grades: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    self_rows = [row for row in grades if row["is_self"]]
    other_rows = [row for row in grades if not row["is_self"]]
    return {
        "answer_accuracy": _answer_summary(answers),
        "overall": _rate(grades),
        "by_arm": _group(grades, "arm"),
        "by_author": _group(grades, "author_model"),
        "self_vs_other": {"자기 답": _rate(self_rows), "다른 모델 답": _rate(other_rows)},
        "by_arm_self": {
            "자기 답": _group(self_rows, "arm"),
            "다른 모델 답": _group(other_rows, "arm"),
        },
        "by_category": _group(grades, "category"),
        "infra_errors": sum(1 for row in grades if row.get("infra_error")),
    }


def pilot_decision(grades: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    infra = sum(bool(row.get("infra_error")) for row in grades)
    blanks = sum(1 for row in grades if row["outcome"] == "blank_answer")
    if grades and blanks / len(grades) > 0.10:
        reasons.append(f"백지 답안 {blanks}/{len(grades)} — 채점이 아니라 빈칸을 재게 된다")
    if grades and infra / len(grades) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(grades)}")
    unparsed = sum(1 for row in grades if row["outcome"] == "unparsed")
    if grades and unparsed / len(grades) > 0.20:
        # 판정을 못 읽어내면 채점 정확도가 아니라 형식 준수를 재게 된다.
        reasons.append(f"판정 파싱 실패 {unparsed}/{len(grades)} — 채점 프롬프트 형식 점검 필요")
    if answers and all(row["answer_correct"] for row in answers):
        reasons.append("답이 전부 정답 — 거짓 통과를 잴 분모(오답)가 없다")
    if answers and not any(row["answer_correct"] for row in answers):
        reasons.append("답이 전부 오답 — 거짓 탈락을 잴 분모(정답)가 없다")
    return {"proceed": not reasons, "reasons": reasons, "grades": len(grades),
            "answers": len(answers), "infra_errors": infra}


_ANSWER_COLUMNS = ["run_id", "question_id", "category", "author_model", "answer_correct",
                   "matched_alias", "elapsed_s", "infra_error", "response_head"]
_GRADE_COLUMNS = ["run_id", "question_id", "category", "author_model", "grader_model", "is_self",
                  "arm", "answer_correct", "verdict", "outcome", "agree", "false_pass",
                  "false_fail", "elapsed_s", "infra_error", "response_head"]


def _load(run_dir: Path, kind: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "raw").glob(f"*-{kind}.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answer_rows: list[dict[str, Any]] = []
    for invocation in _load(run_dir, "answer"):
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        correct, matched = answer_is_correct(invocation["answer_aliases"], response)
        answer_rows.append({
            "run_id": invocation["run_id"],
            "question_id": invocation["question_id"],
            "category": invocation["category"],
            "author_model": invocation["author_model"],
            "answer_correct": correct and not invocation.get("infra_error"),
            "matched_alias": matched,
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            "response_head": response.strip()[:160],
        })

    grade_rows: list[dict[str, Any]] = []
    for invocation in _load(run_dir, "grade"):
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        scored = score_grade(invocation, response)
        grade_rows.append({
            "run_id": invocation["run_id"],
            "question_id": invocation["question_id"],
            "category": invocation["category"],
            "author_model": invocation["author_model"],
            "grader_model": invocation["grader_model"],
            "is_self": invocation["author_model"] == invocation["grader_model"],
            "arm": invocation["arm"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    (run_dir / "answers.json").write_text(json.dumps(answer_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(grade_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows, columns in (("answers.csv", answer_rows, _ANSWER_COLUMNS),
                                ("results.csv", grade_rows, _GRADE_COLUMNS)):
        with (run_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})

    aggregate = aggregate_results(grade_rows, answer_rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    # 채점이 한 건도 없는 상태에서 판정을 굳히면 파싱 실패율 검사가 영영 안 돈다.
    if (grade_rows and (run_dir / "pilot_marker.json").exists()
            and not (run_dir / "pilot_decision.json").exists()):
        decision = pilot_decision(grade_rows, answer_rows)
        (run_dir / "pilot_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return grade_rows, aggregate


def self_test() -> None:
    assert parse_verdict("판정: 정답") == "정답"
    assert parse_verdict("판정 : 오답") == "오답"
    assert parse_verdict("이 답은 틀렸습니다. 판정: 오답") == "오답"
    # 문제 문장에 '정답'이 먼저 나와도 마지막 판정이 이긴다.
    assert parse_verdict("정답을 확인해 보면 391이 맞지 않습니다. 판정: 오답") == "오답"
    assert parse_verdict("제 계산으로는 391이므로 정답입니다") == "정답"
    assert parse_verdict("모르겠습니다") is None
    # 부정형 — 앞 단어만 읽으면 정반대로 집계된다.
    assert parse_verdict("판정: 정답이 아닙니다") is None
    assert parse_verdict("판정: 정답이 아닙니다. 판정: 오답") == "오답"
    assert not answer_is_correct(["3"], "정답은 30입니다.")[0]
    assert answer_is_correct(["3"], "정답은 3입니다.")[0]
    assert not answer_is_correct(["2"], "12개입니다")[0]
    infra_first = score_grade({"answer_correct": True, "graded_answer": "", "infra_error": "boom"}, "")
    assert infra_first["outcome"] == "infra", infra_first

    correct, matched = answer_is_correct(["391"], "17 × 23 = 391 입니다")
    assert correct and matched == "391"
    assert not answer_is_correct(["391"], "17 × 23 = 380 입니다")[0]
    assert answer_is_correct(["8월15일"], "8월 15일입니다")[0]

    agree = score_grade({"answer_correct": True}, "판정: 정답")
    assert agree["outcome"] == "agree", agree
    blank = score_grade({"answer_correct": False, "graded_answer": "  "}, "판정: 정답")
    assert blank["outcome"] == "blank_answer" and not blank["scored"], blank
    fp = score_grade({"answer_correct": False}, "판정: 정답")
    assert fp["outcome"] == "false_pass" and fp["false_pass"], fp
    ff = score_grade({"answer_correct": True}, "판정: 오답")
    assert ff["outcome"] == "false_fail", ff
    unparsed = score_grade({"answer_correct": True}, "잘 모르겠습니다만")
    assert unparsed["outcome"] == "unparsed" and not unparsed["scored"], unparsed
    infra = score_grade({"answer_correct": True, "infra_error": "boom"}, "")
    assert infra["outcome"] == "infra" and not infra["scored"], infra

    rows = [
        {"arm": "blind", "author_model": "a", "grader_model": "a", "is_self": True,
         "category": "계산", "answer_correct": False, "outcome": "false_pass", "agree": False,
         "false_pass": True, "false_fail": False, "scored": True},
        {"arm": "blind", "author_model": "a", "grader_model": "a", "is_self": True,
         "category": "계산", "answer_correct": True, "outcome": "agree", "agree": True,
         "false_pass": False, "false_fail": False, "scored": True},
        {"arm": "with_key", "author_model": "b", "grader_model": "a", "is_self": False,
         "category": "상식", "answer_correct": False, "outcome": "agree", "agree": True,
         "false_pass": False, "false_fail": False, "scored": True},
    ]
    answers = [
        {"author_model": "a", "answer_correct": True},
        {"author_model": "a", "answer_correct": False},
        {"author_model": "b", "answer_correct": False},
    ]
    agg = aggregate_results(rows, answers)
    assert agg["overall"]["agreement_rate"] == round(2 / 3, 4), agg["overall"]
    # 거짓 통과율의 분모는 전체가 아니라 '실제 오답' 2건이다.
    assert agg["overall"]["false_pass_rate"] == 0.5, agg["overall"]
    assert agg["self_vs_other"]["자기 답"]["false_pass_rate"] == 1.0, agg["self_vs_other"]
    assert agg["self_vs_other"]["다른 모델 답"]["false_pass_rate"] == 0.0, agg["self_vs_other"]
    assert agg["answer_accuracy"]["a"]["accuracy"] == 0.5, agg["answer_accuracy"]
    assert agg["by_arm"]["with_key"]["scored"] == 1, agg["by_arm"]

    all_right = pilot_decision([], [{"author_model": "a", "answer_correct": True}])
    assert not all_right["proceed"], all_right
    ok = pilot_decision(
        [dict(rows[0], outcome="agree")],
        [{"author_model": "a", "answer_correct": True}, {"author_model": "a", "answer_correct": False}])
    assert ok["proceed"], ok
    print("selfgrade_score self-test OK")


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
