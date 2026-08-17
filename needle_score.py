#!/usr/bin/env python3
"""Deterministic scorer for the positional-recall (needle) benchmark.

One planted fact per document. The model either quotes it back, says the document does not
contain it, or answers something else. Those three outcomes are scored separately on purpose:
"모른다고 답함"과 "엉뚱한 값을 지어냄"은 사용자가 입는 피해가 전혀 다르다.

Matching is intentionally lenient about spacing and thousands separators and strict about the
value itself — 심은 값이 나와야만 정답이다. 사람이 숫자를 적어 넣을 자리는 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

# 문서에 없다는 취지의 응답. 심은 사실을 못 찾은 것은 같지만 지어낸 것과는 구분해서 센다.
# ★공백을 지운 형태로 비교한다. 목록이 좁으면 정직한 거절이 '지어낸 오답'으로 섞여 들어가
#   "절반이 지어냈다" 같은 틀린 문장이 발행된다(2026-08-03 파일럿·본런에서 실제로 발생).
# ★반대로 넓히기만 하면 채우기 조항을 그대로 읊은 오답("…포함하지 않는다")이 거절로 둔갑한다
#   → '없다/않다'를 아무 데나 잡지 않고 부재를 말하는 어구만 나열한다.
_ABSTAIN_PATTERNS = (
    "문서에없", "문서에는없", "문서상없",
    "정보없", "정보가없", "정보는없", "정보에없",
    "언급이없", "언급은없", "언급되어있지않", "언급되지않", "언급이되어있지않", "언급되어있지는않",
    "내용이없", "내용은없", "내용을찾을수없",
    "규정이없", "조항이없", "항목이없",
    "나와있지않", "나와있지는않",
    "찾을수없", "확인할수없", "알수없", "확인되지않", "확인이불가",
    "명시되어있지않", "명시되지않", "명시된바없",
    "기재되어있지않", "기재되지않",
    "포함되어있지않", "포함되지않았",
    "제공되지않", "제공이되어있지않", "제공되어있지않", "제시되지않", "주어지지않",
    "답변을드릴수없", "답변드릴수없", "답할수없", "답변이불가",
    "해당사항없", "관련내용없",
)


def normalize(text: str) -> str:
    """공백·천단위 쉼표·대소문자만 지우고 값 자체는 건드리지 않는다."""
    folded = unicodedata.normalize("NFKC", text)
    folded = re.sub(r"(?<=\d),(?=\d{3})", "", folded)
    folded = re.sub(r"\s+", "", folded)
    return folded.upper()


def contains_value(value: str, normalized_haystack: str) -> bool:
    """정규화된 응답이 그 값을 담고 있는가.

    ★숫자 값은 앞뒤에 숫자가 붙으면 안 된다 — 단순 부분문자열이면 정답 '100'이 '1000'에,
    방해값 '9000'이 '90000'에 걸려 오답이 정답으로 둔갑한다(T2 적대검증 2026-08-03).
    현 데이터에서는 한 건도 바뀌지 않았지만, 다음 테스트셋에서 조용히 터질 자리라 막아둔다.
    """
    needle = normalize(value)
    if not needle:
        return False
    if needle.isdigit():
        return re.search(r"(?<!\d)" + re.escape(needle) + r"(?!\d)", normalized_haystack) is not None
    return needle in normalized_haystack


def is_abstain(text: str) -> bool:
    squeezed = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return any(re.sub(r"\s+", "", pattern) in squeezed for pattern in _ABSTAIN_PATTERNS)


def score_one(invocation: dict[str, Any], response_text: str) -> dict[str, Any]:
    aliases = invocation.get("answer_aliases") or []
    if not aliases:
        raise ValueError(f"answer_aliases 없음: run_id={invocation.get('run_id')}")
    body = (response_text or "").strip()
    normalized = normalize(body)
    matched = next((alias for alias in aliases if contains_value(alias, normalized)), None)
    # 방해 조항의 값을 집어온 오답은 "못 찾음"이 아니라 "옆 조항을 가져옴"이라 따로 센다.
    pulled = next((value for value in (invocation.get("distractor_values") or [])
                   if contains_value(value, normalized)), None)
    abstained = is_abstain(body)

    if invocation.get("infra_error"):
        outcome = "infra"
    elif not body:
        outcome = "empty"
    elif matched is not None:
        # 값을 맞혔으면 "문서에 없다"는 말이 뒤에 붙어 있어도 회상은 성공한 것으로 본다.
        outcome = "hit"
    elif pulled is not None:
        outcome = "distractor"
    elif abstained:
        outcome = "abstain"
    else:
        outcome = "wrong"

    return {
        "outcome": outcome,
        "hit": outcome == "hit",
        "abstain": outcome == "abstain",
        "distractor_pull": outcome == "distractor",
        "wrong": outcome in {"wrong", "distractor"},
        "scored": outcome in {"hit", "abstain", "wrong", "distractor"},
        "matched_alias": matched,
        "matched_distractor": pulled,
        "response_chars": len(body),
        "response_head": body[:160],
    }


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["scored"]]
    hits = sum(row["hit"] for row in scored)
    abstains = sum(row["abstain"] for row in scored)
    wrongs = sum(row["wrong"] for row in scored)
    pulls = sum(row.get("distractor_pull", False) for row in scored)
    return {
        "total": len(rows),
        "scored": len(scored),
        "hit": hits,
        "abstain": abstains,
        "wrong": wrongs,
        "distractor_pull": pulls,
        "hit_rate": round(hits / len(scored), 4) if scored else None,
        "abstain_rate": round(abstains / len(scored), 4) if scored else None,
        "wrong_rate": round(wrongs / len(scored), 4) if scored else None,
        "distractor_rate": round(pulls / len(scored), 4) if scored else None,
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: _rate(bucket) for name, bucket in sorted(buckets.items())}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    # 컨텍스트 창을 지정한 팔과 기본값 팔은 애초에 다른 실험이라 섞어서 평균 내지 않는다.
    explicit = [row for row in results if row.get("ctx_mode", "explicit") == "explicit"]
    default_ctx = [row for row in results if row.get("ctx_mode") == "default"]
    main = [row for row in explicit if row.get("arm") == "haystack"]
    control = [row for row in explicit if row.get("arm") == "control"]

    by_cell: dict[str, Any] = {}
    for row in main:
        cell = f"{row['length']}/{'방해있음' if row.get('distractors') else '방해없음'}/{row['position']}"
        by_cell.setdefault(cell, []).append(row)
    with_d = [row for row in main if row.get("distractors")]
    without_d = [row for row in main if not row.get("distractors")]

    prompt_tokens = [row["prompt_eval_count"] for row in main if row.get("prompt_eval_count")]
    num_ctx_values = {row.get("num_ctx") for row in explicit if row.get("num_ctx")}
    max_prompt_tokens = max(prompt_tokens) if prompt_tokens else None
    context_fits = None
    if max_prompt_tokens is not None and len(num_ctx_values) == 1:
        # 프롬프트가 컨텍스트 창을 넘으면 앞부분이 잘려나가므로 "위치 효과"가 아니라 "잘림"을 재게 된다.
        context_fits = max_prompt_tokens < next(iter(num_ctx_values))

    return {
        "overall": _rate(main),
        "control": _rate(control),
        "by_distractor": {"없음": _rate(without_d), "있음": _rate(with_d)},
        "by_length": _group(main, "length"),
        "by_length_distractor": {
            "없음": _group(without_d, "length"),
            "있음": _group(with_d, "length"),
        },
        "by_position": _group(main, "position"),
        "by_position_distractor": {
            "없음": _group(without_d, "position"),
            "있음": _group(with_d, "position"),
        },
        "by_needle": _group(main, "needle_id"),
        "by_needle_distractor": {
            "없음": _group(without_d, "needle_id"),
            "있음": _group(with_d, "needle_id"),
        },
        "by_cell": {cell: _rate(bucket) for cell, bucket in sorted(by_cell.items())},
        "context": {
            "num_ctx": sorted(value for value in num_ctx_values if value is not None),
            "max_prompt_eval_count": max_prompt_tokens,
            "median_prompt_eval_count": int(statistics.median(prompt_tokens)) if prompt_tokens else None,
            "prompt_fits_context": context_fits,
        },
        "ctx_default": {
            "overall": _rate(default_ctx),
            "by_length": _group(default_ctx, "length"),
            "by_position": _group(default_ctx, "position"),
            "prompt_eval_counts": sorted({row["prompt_eval_count"] for row in default_ctx
                                          if row.get("prompt_eval_count")}),
        },
        "infra_errors": sum(1 for row in results if row.get("infra_error")),
        "empty_responses": sum(1 for row in results if row["outcome"] == "empty"),
    }


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Plumbing check before the full run."""
    reasons: list[str] = []
    infra = sum(bool(row.get("infra_error")) for row in results)
    if results and infra / len(results) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(results)}")
    if results and all(row["outcome"] == "empty" for row in results):
        reasons.append("응답이 전부 0자 — 프롬프트/모델 점검 필요")
    control = [row for row in results if row.get("arm") == "control"]
    if control and not any(row["hit"] for row in control):
        # 문장 한 줄만 준 대조군도 못 맞히면 회상 문제가 아니라 질문/채점 설계가 깨진 것이다.
        reasons.append("대조군(문장 1줄) 정답 0건 — 질문 또는 채점기 결함 의심")
    # 기본값 팔은 잘리는 게 관측 대상이므로 제외하고, 명시 팔에서만 잘림을 중단 사유로 본다.
    truncated = [row for row in results
                 if row.get("ctx_mode", "explicit") == "explicit"
                 and row.get("num_ctx") and row.get("prompt_eval_count")
                 and row["prompt_eval_count"] >= row["num_ctx"]]
    if truncated:
        reasons.append(f"프롬프트가 컨텍스트 창을 넘김 {len(truncated)}건 — num_ctx 상향 필요")
    return {"proceed": not reasons, "reasons": reasons, "runs": len(results), "infra_errors": infra}


_CSV_COLUMNS = ["run_id", "arm", "ctx_mode", "length", "distractors", "position", "needle_id", "needle_kind",
                "outcome", "hit", "abstain", "wrong", "distractor_pull", "matched_alias",
                "matched_distractor", "doc_chars", "clause_count", "needle_clause_index",
                "response_chars", "elapsed_s", "prompt_eval_count", "eval_count", "num_ctx",
                "infra_error", "response_head"]


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
            "arm": invocation["arm"],
            "length": invocation["length"],
            "distractors": invocation.get("distractors", False),
            "ctx_mode": invocation.get("ctx_mode", "explicit"),
            "position": invocation["position"],
            "needle_id": invocation["needle_id"],
            "needle_kind": invocation["needle_kind"],
            "doc_chars": invocation.get("doc_chars"),
            "clause_count": invocation.get("clause_count"),
            "needle_clause_index": invocation.get("needle_clause_index"),
            "num_ctx": invocation.get("num_ctx"),
            "elapsed_s": invocation.get("elapsed_s"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "eval_count": metrics.get("eval_count"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _CSV_COLUMNS})

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    # 파일럿 모드에서만 하네스가 pilot_marker.json을 남긴다 — 본런은 이 판정을 덮어쓰지 않는다.
    if (run_dir / "pilot_marker.json").exists() and not (run_dir / "pilot_decision.json").exists():
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    base = {"run_id": 1, "answer_aliases": ["12500", "1만2500"]}
    hit = score_one(base, "12,500원입니다.")
    assert hit["outcome"] == "hit" and hit["matched_alias"] == "12500", hit
    spaced = score_one(base, "1 2 5 0 0 원")
    assert spaced["outcome"] == "hit", spaced
    alias = score_one(base, "1만 2500원까지 지급됩니다.")
    assert alias["outcome"] == "hit", alias
    abstain = score_one(base, "문서에 없습니다.")
    assert abstain["outcome"] == "abstain" and not abstain["hit"], abstain
    abstain2 = score_one(base, "해당 내용은 나와 있지 않습니다.")
    assert abstain2["outcome"] == "abstain", abstain2
    # 파일럿에서 실제로 나온 표현 — 이걸 wrong으로 세면 정직한 거절이 지어낸 오답과 뭉개진다.
    abstain3 = score_one(base, "정보가 제공되지 않았습니다. 해당 규정에는 야근 식대에 대한 내용이 없습니다.")
    assert abstain3["outcome"] == "abstain", abstain3
    abstain4 = score_one(base, "문서에 명시되어 있지 않습니다.")
    assert abstain4["outcome"] == "abstain", abstain4
    # 본런 원응답에서 그대로 가져온 표현들 — 좁은 목록이 이걸 '지어낸 오답'으로 셌다.
    for text in ("정보 없음",
                 "정보 제공이 되어 있지 않습니다. 해당 문맥에서 언급이 없습니다.",
                 "정보에는 3층 소회의실 예약에 대한 언급이 없습니다.",
                 "따라서 질문에 대한 답변을 드릴 수 없습니다."):
        assert score_one(base, text)["outcome"] == "abstain", text
    # 채우기 조항을 그대로 읊은 오답은 거절이 아니다('포함하지 않는다'에 걸려선 안 된다).
    quoted = score_one(base, "제84조 (근무 시간) 기본 근무 시간은 하루 여덟 시간으로 하고, "
                             "점심 시간은 근무 시간에 포함하지 않는다.")
    assert quoted["outcome"] == "wrong", quoted
    assert score_one(base, "제102조")["outcome"] == "wrong"
    # 값을 말하면서 사족을 붙인 건 여전히 정답 — 거절 문구가 정답 판정을 덮지 않는다.
    assert score_one(base, "12500원이며 그 외 내용이 없습니다.")["outcome"] == "hit"
    wrong = score_one(base, "15000원입니다.")
    assert wrong["outcome"] == "wrong" and not wrong["distractor_pull"], wrong
    pulled = score_one({**base, "distractor_values": ["9000", "7000"]}, "9,000원입니다.")
    assert pulled["outcome"] == "distractor" and pulled["wrong"] and pulled["matched_distractor"] == "9000", pulled
    # 정답을 맞혔으면 방해 값이 같이 나열돼도 정답이다(회상 자체는 성공).
    both = score_one({**base, "distractor_values": ["9000"]}, "야근은 12500원, 주말은 9000원입니다.")
    assert both["outcome"] == "hit", both
    other_wrong = score_one({**base, "distractor_values": ["9000"]}, "15000원입니다.")
    assert other_wrong["outcome"] == "wrong", other_wrong
    # 값을 맞히면서 사족을 붙인 경우는 정답으로 센다(회상 자체는 성공).
    mixed = score_one(base, "문서에 명시되어 있지는 않지만 12500원으로 보입니다.")
    assert mixed["outcome"] == "hit", mixed
    empty = score_one(base, "   ")
    assert empty["outcome"] == "empty" and not empty["scored"], empty
    infra = score_one({**base, "infra_error": "boom"}, "")
    assert infra["outcome"] == "infra" and not infra["scored"], infra
    name_case = score_one({"run_id": 2, "answer_aliases": ["김하늘"]}, "총무팀 김하늘 주임입니다.")
    assert name_case["outcome"] == "hit", name_case
    name_wrong = score_one({"run_id": 3, "answer_aliases": ["김하늘"]}, "총무팀 이서준 대리입니다.")
    assert name_wrong["outcome"] == "wrong", name_wrong
    # 숫자 별칭은 더 긴 숫자 안에 박혀 있으면 정답이 아니다.
    assert score_one({"run_id": 9, "answer_aliases": ["100"]}, "1000원입니다")["outcome"] == "wrong"
    assert score_one({"run_id": 10, "answer_aliases": ["100"]}, "100원입니다")["outcome"] == "hit"
    assert score_one({"run_id": 11, "answer_aliases": ["12500"], "distractor_values": ["9000"]},
                     "90000원입니다")["outcome"] == "wrong"
    code_case = score_one({"run_id": 4, "answer_aliases": ["GX-4718", "GX4718"]}, "양식 gx4718 입니다")
    assert code_case["outcome"] == "hit", code_case
    sched = score_one({"run_id": 5, "answer_aliases": ["셋째 주 목요일"]}, "매월 셋째주 목요일 밤 11시")
    assert sched["outcome"] == "hit", sched

    rows = [
        {"arm": "haystack", "length": "L4-48k", "distractors": False, "position": "0.5",
         "needle_id": "N1", "outcome": "hit", "hit": True, "abstain": False, "wrong": False,
         "distractor_pull": False, "scored": True, "prompt_eval_count": 9000, "num_ctx": 16384},
        {"arm": "haystack", "length": "L4-48k", "distractors": True, "position": "0.5",
         "needle_id": "N1", "outcome": "distractor", "hit": False, "abstain": False, "wrong": True,
         "distractor_pull": True, "scored": True, "prompt_eval_count": 9100, "num_ctx": 16384},
        {"arm": "control", "length": "control", "distractors": False, "position": "control",
         "needle_id": "N1", "outcome": "hit", "hit": True, "abstain": False, "wrong": False,
         "distractor_pull": False, "scored": True, "prompt_eval_count": 80, "num_ctx": 16384},
    ]
    agg = aggregate_results(rows)
    assert agg["overall"]["hit_rate"] == 0.5, agg["overall"]
    assert agg["control"]["hit"] == 1, agg["control"]
    assert agg["by_length"]["L4-48k"]["scored"] == 2, agg["by_length"]
    assert agg["by_distractor"]["없음"]["hit_rate"] == 1.0, agg["by_distractor"]
    assert agg["by_distractor"]["있음"]["distractor_rate"] == 1.0, agg["by_distractor"]
    assert agg["by_length_distractor"]["있음"]["L4-48k"]["scored"] == 1, agg["by_length_distractor"]
    assert agg["context"]["prompt_fits_context"] is True, agg["context"]
    over = [dict(rows[0], prompt_eval_count=20000)]
    decision = pilot_decision(over)
    assert not decision["proceed"] and any("컨텍스트" in reason for reason in decision["reasons"]), decision
    no_control_hit = pilot_decision([dict(rows[2], outcome="wrong", hit=False, wrong=True)])
    assert not no_control_hit["proceed"], no_control_hit
    print("needle_score self-test OK")


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
