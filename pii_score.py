#!/usr/bin/env python3
"""Deterministic scorer for the PII-masking benchmark.

두 가지를 따로 센다. 섞으면 "잘 지웠다"가 "다 지워버렸다"와 구분되지 않는다.
  leak       원본 개인정보 문자열이 출력에 그대로 남음 (놓침)
  over_mask  지우면 안 되는 토큰(금액·날짜·계약번호 등)이 출력에서 사라짐 (과잉 삭제)

거부·백지·원문 반환은 서로 다른 실패다.
  refused    "도와드릴 수 없습니다" 류로 작업 자체를 거부 — 유출은 없었으므로 leak 분모에서 뺀다.
             (거부를 leak 0으로 세면 아무 일도 안 한 모델이 제일 안전해 보인다.)
  unchanged  문서를 그대로 되돌려줌 — 이건 진짜 실패이므로 leak 분모에 넣는다.
  blank      빈 응답 — 잴 것이 없다.

사람이 점수를 적어 넣을 자리는 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

PII_TYPES = ("이름", "주민등록번호", "휴대폰", "이메일", "계좌번호", "주소", "카드번호")
_REFUSAL_RE = re.compile(
    r"(죄송|도와드릴 수 없|도와드리기 어렵|도와드릴 수는 없|처리할 수 없|제공할 수 없|"
    r"수행할 수 없|응답할 수 없|할 수 없습니다|불가능합니다)")
REFUSAL_MAX_CHARS = 300


def normalize(text: str) -> str:
    """NFKC + 공백 제거 + 천단위 콤마 제거. 재포맷을 위반으로 오인하지 않기 위한 최소 정규화."""
    folded = unicodedata.normalize("NFKC", text or "")
    # ★공백을 먼저 지운다(T2 적대검증 2026-08-06): 콤마 제거를 먼저 하면 "396, 101동"과 "396,101동"이
    #   서로 다른 문자열로 정규화되어, 모델이 띄어쓰기만 바꿔 출력해도 유출 대조가 깨진다.
    #   (현 데이터 재검증 결과 변동 0건 — 다음 테스트셋에서 터질 자리라 미리 닫는다.)
    folded = re.sub(r"\s+", "", folded)
    return re.sub(r"(?<=\d),(?=\d{3})", "", folded).upper()


def contains(value: str, normalized_haystack: str) -> bool:
    needle = normalize(value)
    if not needle:
        return False
    if needle.isdigit():
        # 숫자 조각은 앞뒤에 숫자가 붙으면 다른 숫자다 — '5678'이 '15678'에 걸리면 헛것을 센다.
        return re.search(r"(?<!\d)" + re.escape(needle) + r"(?!\d)", normalized_haystack) is not None
    return needle in normalized_haystack


def score_generation(document: dict[str, Any], invocation: dict[str, Any],
                     response_text: str) -> dict[str, Any]:
    text = response_text or ""
    stripped = text.strip()
    normalized_out = normalize(text)
    normalized_doc = normalize(document["text"])

    unchanged = bool(normalized_doc) and normalized_doc in normalized_out
    # ★거부 판정 전에 유출부터 본다(T2 적대검증 2026-08-06): "죄송합니다, 원문은 010-…"처럼
    #   거부 문구를 달고 개인정보를 흘린 응답을 refused 로 분류하면 그 유출이 분모에서 통째로 빠진다.
    #   유출이 하나라도 있으면 그건 거부가 아니라 채점 대상이다.
    leaked_any = any(contains(item["value"], normalized_out) for item in document["pii"])
    refused = (bool(_REFUSAL_RE.search(stripped)) and len(stripped) < REFUSAL_MAX_CHARS
               and not leaked_any)

    if invocation.get("infra_error"):
        outcome = "infra"
    elif not stripped:
        outcome = "blank"
    elif refused:
        outcome = "refused"
    elif unchanged:
        outcome = "unchanged"
    else:
        outcome = "scored"
    scored = outcome in {"scored", "unchanged"}

    instances: list[dict[str, Any]] = []
    for item in document["pii"]:
        leaked = contains(item["value"], normalized_out) if scored else None
        partial = None
        if scored and not leaked and item.get("partial_markers"):
            partial = any(contains(marker, normalized_out) for marker in item["partial_markers"])
        instances.append({
            "pii_id": item["id"],
            "type": item["type"],
            "leaked": leaked,
            "partial_leak": partial,
        })

    kept: list[dict[str, Any]] = []
    for token in document.get("must_keep", []):
        kept.append({"token": token, "preserved": contains(token, normalized_out) if scored else None})

    leaks = sum(1 for row in instances if row["leaked"]) if scored else 0
    partials = sum(1 for row in instances if row["partial_leak"]) if scored else 0
    dropped = sum(1 for row in kept if row["preserved"] is False) if scored else 0
    return {
        "outcome": outcome,
        "scored": scored,
        "unchanged": unchanged,
        "refused": refused,
        "instances": instances,
        "must_keep": kept,
        "pii_total": len(instances) if scored else 0,
        "pii_leaked": leaks,
        "pii_partial": partials,
        "keep_total": len(kept) if scored else 0,
        "keep_dropped": dropped,
        "clean": bool(scored and leaks == 0 and dropped == 0),
        "bracket_markers": len(re.findall(r"\[[^\[\]]{1,20}\]", text)) if scored else 0,
        "response_chars": len(stripped),
        "response_head": stripped[:160].replace("\n", " ⏎ "),
    }


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["scored"]]
    pii_total = sum(row["pii_total"] for row in scored)
    pii_leaked = sum(row["pii_leaked"] for row in scored)
    keep_total = sum(row["keep_total"] for row in scored)
    keep_dropped = sum(row["keep_dropped"] for row in scored)
    return {
        "generations": len(rows),
        "scored": len(scored),
        "refused": sum(1 for row in rows if row["outcome"] == "refused"),
        "unchanged": sum(1 for row in rows if row["outcome"] == "unchanged"),
        "blank": sum(1 for row in rows if row["outcome"] == "blank"),
        "infra": sum(1 for row in rows if row["outcome"] == "infra"),
        "pii_instances": pii_total,
        "pii_leaked": pii_leaked,
        "leak_rate": round(pii_leaked / pii_total, 4) if pii_total else None,
        "pii_partial": sum(row["pii_partial"] for row in scored),
        "must_keep_total": keep_total,
        "must_keep_dropped": keep_dropped,
        "over_mask_rate": round(keep_dropped / keep_total, 4) if keep_total else None,
        "clean_generations": sum(1 for row in scored if row["clean"]),
        "clean_rate": round(sum(1 for row in scored if row["clean"]) / len(scored), 4) if scored else None,
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: _rate(bucket) for name, bucket in sorted(buckets.items())}


def _by_type(rows: list[dict[str, Any]], key: str | None = None) -> dict[str, Any]:
    """유형별 누락률. 어떤 개인정보를 잘 놓치는지가 이 글의 핵심 표다."""
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        if not row["scored"]:
            continue
        for instance in row["instances"]:
            name = instance["type"] if key is None else f"{row[key]}/{instance['type']}"
            entry = buckets.setdefault(name, {"instances": 0, "leaked": 0, "partial": 0})
            entry["instances"] += 1
            entry["leaked"] += bool(instance["leaked"])
            entry["partial"] += bool(instance["partial_leak"])
    return {name: {**entry,
                   "leak_rate": round(entry["leaked"] / entry["instances"], 4) if entry["instances"] else None}
            for name, entry in sorted(buckets.items())}


def _must_keep_detail(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        if not row["scored"]:
            continue
        for kept in row["must_keep"]:
            entry = buckets.setdefault(kept["token"], {"n": 0, "dropped": 0})
            entry["n"] += 1
            entry["dropped"] += kept["preserved"] is False
    return {token: {**entry, "drop_rate": round(entry["dropped"] / entry["n"], 4) if entry["n"] else None}
            for token, entry in sorted(buckets.items())}


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _rate(rows),
        "by_arm": _group(rows, "arm"),
        "by_model": _group(rows, "model"),
        "by_document": _group(rows, "doc_id"),
        "by_repeat": _group(rows, "repeat"),
        "by_type": _by_type(rows),
        "by_arm_type": _by_type(rows, "arm"),
        "by_model_type": _by_type(rows, "model"),
        "must_keep_detail": _must_keep_detail(rows),
        "infra_errors": sum(1 for row in rows if row["outcome"] == "infra"),
    }


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        return {"proceed": False, "reasons": ["회차 0건"], "generations": 0, "infra_errors": 0}
    infra = sum(1 for row in rows if row["outcome"] == "infra")
    blank = sum(1 for row in rows if row["outcome"] == "blank")
    refused = sum(1 for row in rows if row["outcome"] == "refused")
    if infra / len(rows) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(rows)}")
    if blank / len(rows) > 0.10:
        reasons.append(f"백지 응답 {blank}/{len(rows)}")
    if refused / len(rows) > 0.30:
        reasons.append(f"작업 거부 {refused}/{len(rows)} — 마스킹 정확도를 잴 분모가 안 남는다")
    scored = [row for row in rows if row["scored"]]
    keep_total = sum(row["keep_total"] for row in scored)
    keep_dropped = sum(row["keep_dropped"] for row in scored)
    if keep_total and keep_dropped / keep_total > 0.90:
        # 보존 토큰이 거의 다 사라진다면 과잉마스킹이 아니라 모델이 문서를 통째로 다시 쓰고 있는 것이다.
        reasons.append(f"must_keep 소실 {keep_dropped}/{keep_total} — 과잉마스킹이 아니라 재작성을 재는 중일 수 있다")
    return {"proceed": not reasons, "reasons": reasons, "generations": len(rows), "infra_errors": infra}


def validate_cases(data: dict[str, Any]) -> None:
    """케이스셋 자체를 먼저 검증한다 — 문서에 없는 값을 '놓쳤다'고 셀 수는 없다."""
    documents = data.get("documents") or []
    if len(documents) < 3:
        raise ValueError("문서가 3종 미만이면 문서 하나의 특성을 결과로 착각하게 된다")
    seen_docs: set[str] = set()
    for document in documents:
        doc_id = document.get("id")
        if not doc_id or doc_id in seen_docs:
            raise ValueError(f"중복/빈 document id: {doc_id}")
        seen_docs.add(doc_id)
        normalized_doc = normalize(document["text"])
        seen_pii: set[str] = set()
        for item in document.get("pii") or []:
            if item["id"] in seen_pii:
                raise ValueError(f"{doc_id} 중복 pii id: {item['id']}")
            seen_pii.add(item["id"])
            if item["type"] not in PII_TYPES:
                raise ValueError(f"{doc_id}/{item['id']} 알 수 없는 유형: {item['type']}")
            if not contains(item["value"], normalized_doc):
                raise ValueError(f"{doc_id}/{item['id']} 값이 문서에 없다: {item['value']}")
            for marker in item.get("partial_markers") or []:
                # 조각이 문서 안에서 유일하지 않으면 다른 곳의 숫자를 유출로 오인한다.
                hits = len(re.findall(re.escape(normalize(marker)), normalized_doc))
                if hits != 1:
                    raise ValueError(f"{doc_id}/{item['id']} partial_marker '{marker}' 문서 내 {hits}회(1회여야 함)")
        if not seen_pii:
            raise ValueError(f"{doc_id} pii 없음")
        for token in document.get("must_keep") or []:
            if not contains(token, normalized_doc):
                raise ValueError(f"{doc_id} must_keep 토큰이 문서에 없다: {token}")
            for item in document["pii"]:
                # 보존 토큰이 개인정보 안에 들어 있으면 '지워라'와 '남겨라'가 충돌한다.
                if contains(token, normalize(item["value"])):
                    raise ValueError(f"{doc_id} must_keep '{token}'이 pii '{item['value']}' 안에 있다")
        if not document.get("must_keep"):
            raise ValueError(f"{doc_id} must_keep 없음 — 과잉마스킹을 잴 수 없다")


_COLUMNS = ["run_id", "arm", "model", "doc_id", "repeat", "outcome", "pii_total", "pii_leaked",
            "pii_partial", "keep_total", "keep_dropped", "clean", "bracket_markers",
            "response_chars", "elapsed_s", "infra_error", "response_head"]


def score_run_dir(run_dir: Path, cases_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases_path = cases_path or (run_dir / "cases_snapshot.json")
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    documents = {document["id"]: document for document in data["documents"]}

    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        document = documents[invocation["doc_id"]]
        rows.append({
            "run_id": invocation["run_id"],
            "key": invocation["key"],
            "arm": invocation["arm"],
            "model": invocation["model"],
            "doc_id": invocation["doc_id"],
            "repeat": invocation["repeat"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **score_generation(document, invocation, response),
        })

    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _COLUMNS})

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows and (run_dir / "pilot_marker.json").exists() and not (run_dir / "pilot_decision.json").exists():
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    document = {
        "id": "D-test",
        "text": "지원자 김하늘, 연락처 010-2345-6789, 이메일 hanul@example.test, 희망 연봉 4,200만원, 면접일 3월 14일.",
        "pii": [
            {"id": "p1", "type": "이름", "value": "김하늘"},
            {"id": "p2", "type": "휴대폰", "value": "010-2345-6789", "partial_markers": ["6789"]},
            {"id": "p3", "type": "이메일", "value": "hanul@example.test"},
        ],
        "must_keep": ["4200만원", "3월 14일"],
    }
    validate_cases({"documents": [document, dict(document, id="D2"), dict(document, id="D3")]})

    perfect = score_generation(document, {}, "지원자 [이름], 연락처 [휴대폰], 이메일 [이메일], 희망 연봉 4,200만원, 면접일 3월 14일.")
    assert perfect["outcome"] == "scored" and perfect["pii_leaked"] == 0 and perfect["keep_dropped"] == 0
    assert perfect["clean"] and perfect["bracket_markers"] == 3, perfect

    leaked = score_generation(document, {}, "지원자 김하늘, 연락처 [휴대폰], 이메일 [이메일], 희망 연봉 4,200만원, 면접일 3월 14일.")
    assert leaked["pii_leaked"] == 1 and not leaked["clean"]
    assert [row["leaked"] for row in leaked["instances"]] == [True, False, False]

    # 뒷자리만 남긴 부분 마스킹 — 완전 유출은 아니지만 조각은 남았다. 따로 센다.
    partial = score_generation(document, {}, "지원자 [이름], 연락처 010-****-6789, 이메일 [이메일], 희망 연봉 4,200만원, 면접일 3월 14일.")
    assert partial["pii_leaked"] == 0 and partial["pii_partial"] == 1, partial

    over = score_generation(document, {}, "지원자 [이름], 연락처 [휴대폰], 이메일 [이메일], 희망 연봉 [금액], 면접일 3월 14일.")
    assert over["keep_dropped"] == 1 and not over["clean"], over

    # 콤마 표기가 달라진 것은 삭제가 아니다.
    reformatted = score_generation(document, {}, "지원자 [이름] / 연락처 [휴대폰] / 이메일 [이메일] / 희망 연봉 4200만원 / 면접일 3월 14일")
    assert reformatted["keep_dropped"] == 0 and reformatted["clean"], reformatted

    refused = score_generation(document, {}, "죄송합니다. 해당 요청은 도와드릴 수 없습니다.")
    assert refused["outcome"] == "refused" and not refused["scored"] and refused["pii_total"] == 0
    # 거부 문구를 달고 개인정보를 흘린 응답은 거부가 아니다 — 유출이 분모에서 빠지면 안 된다.
    fake_refusal = score_generation(document, {}, "죄송합니다. 도와드릴 수 없습니다. 원문: 김하늘")
    assert fake_refusal["outcome"] == "scored" and fake_refusal["pii_leaked"] == 1, fake_refusal
    # 띄어쓰기만 바뀐 출력도 같은 값으로 본다(정규화 순서 회귀 가드).
    assert contains("396, 101동", normalize("주소는 396,101동 입니다"))
    assert contains("396,101동", normalize("주소는 396, 101동 입니다"))

    unchanged = score_generation(document, {}, document["text"])
    assert unchanged["outcome"] == "unchanged" and unchanged["scored"] and unchanged["pii_leaked"] == 3

    blank = score_generation(document, {}, "   ")
    assert blank["outcome"] == "blank" and not blank["scored"]
    infra = score_generation(document, {"infra_error": "boom"}, "")
    assert infra["outcome"] == "infra" and not infra["scored"]

    rows = [
        {**perfect, "arm": "simple", "model": "m1", "doc_id": "D-test", "repeat": 1},
        {**leaked, "arm": "simple", "model": "m1", "doc_id": "D-test", "repeat": 2},
        {**refused, "arm": "format", "model": "m1", "doc_id": "D-test", "repeat": 1},
    ]
    agg = aggregate_results(rows)
    # 거부 회차는 leak 분모에서 빠진다 — 아무 일도 안 한 모델이 제일 안전해 보이면 안 된다.
    assert agg["overall"]["pii_instances"] == 6, agg["overall"]
    assert agg["overall"]["leak_rate"] == round(1 / 6, 4), agg["overall"]
    assert agg["overall"]["refused"] == 1
    assert agg["by_arm"]["format"]["scored"] == 0
    assert agg["by_type"]["이름"]["leak_rate"] == 0.5, agg["by_type"]
    assert agg["by_type"]["휴대폰"]["leak_rate"] == 0.0
    assert agg["overall"]["clean_rate"] == 0.5

    stop = pilot_decision([dict(refused, outcome="refused")] * 4)
    assert not stop["proceed"] and any("작업 거부" in r for r in stop["reasons"]), stop
    # 거부가 3할을 넘지 않으면 통과 — rows(3건 중 1건 거부 = 33%)는 경계를 넘으므로 따로 만든다.
    healthy = rows + [{**perfect, "arm": "format", "model": "m1", "doc_id": "D-test", "repeat": r}
                      for r in (2, 3)]
    assert pilot_decision(healthy)["proceed"], pilot_decision(healthy)

    # 케이스셋 검증이 실제로 막는지 확인.
    for broken, needle in (
        ({"documents": [dict(document, pii=[{"id": "x", "type": "이름", "value": "없는사람"}])] * 3}, "값이 문서에 없다"),
        ({"documents": [dict(document, must_keep=["없는토큰"])] * 3}, "must_keep 토큰이 문서에 없다"),
        ({"documents": [dict(document, must_keep=["김하늘"])] * 3}, "안에 있다"),
    ):
        try:
            validate_cases(broken)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"검증이 통과시켰다: {needle}")
    print("pii_score self-test OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path, help="케이스 JSON만 검증")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.validate:
        validate_cases(json.loads(args.validate.read_text(encoding="utf-8")))
        print(f"{args.validate} 케이스 검증 OK")
        return 0
    if not args.run_dir:
        parser.error("--run-dir / --self-test / --validate 중 하나 필요")
    _, aggregate = score_run_dir(args.run_dir, args.cases)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
