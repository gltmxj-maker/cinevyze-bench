#!/usr/bin/env python3
"""Deterministic scorer for the option-order bias benchmark.

두 안을 주고 고르게 한 뒤 **순서만 뒤집어** 다시 묻는다. 내용이 같은데 답이 바뀌면
그건 판단이 아니라 자리다.

  flip             AB에서 고른 실질 안과 BA에서 고른 실질 안이 다름(라벨이 아니라 내용으로 되돌려 비교)
  first_pick_rate  먼저 제시된 쪽을 고른 비율 — 무편향이면 50%
  label_pick_rate  라벨 ①을 고른 비율 — 자리 편향인지 라벨 편향인지 가르는 축
  accuracy         정답이 있는 케이스만
  unparseable      마지막 줄에서 답을 못 뽑음 — 분모에서 분리한다

★답 추출은 보수적으로 한다. 마지막 줄이 답처럼 안 생겼으면 '못 뽑음'으로 두고 분모에서 뺀다.
  장단점을 나열한 줄에서 마지막에 언급된 선택지를 답으로 세면, 나열 순서가 그대로 '편향'으로
  둔갑한다 — 그건 모델의 편향이 아니라 채점기가 만든 편향이다.

사람이 점수를 적어 넣을 자리는 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ARMS = ("plain", "reason_first")
ORDERS = ("AB", "BA")
LABEL_SCHEMES = ("normal", "reversed")
KINDS = ("objective", "subjective")

_MARKER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile("①"), 1),
    (re.compile("②"), 2),
    (re.compile(r"(?<![0-9])1\s*번(?!호)"), 1),
    (re.compile(r"(?<![0-9])2\s*번(?!호)"), 2),
)
# ★T2 적대검증 2026-08-08 — "첫 번째/두 번째"는 라벨(①/②)이 아니라 **자리**를 가리킨다.
#   라벨로 매핑하면 라벨 대조군(위쪽=②)에서 `3 - label` 을 한 번 더 태워 정반대 자리가 된다.
#   그래서 별도 채널로 뽑아 자리로 바로 해석한다. (현 데이터 등장 0건 — 다음 라운드에서 터질 자리라 미리 닫는다.)
_POSITION_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"첫\s*번째"), 1),
    (re.compile(r"두\s*번째"), 2),
)
# 장단점을 늘어놓은 줄은 답이 아니다.
_REASON_RE = re.compile(r"(장점|단점|반면|비교|이유|근거)")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(정답|답변|답|선택|결론|최종)\s*[:：]?")
_SHORT_LINE = 12
_SCAN_LINES = 3


def markers_in(line: str) -> set[int]:
    found: set[int] = set()
    for pattern, choice in _MARKER_PATTERNS:
        if pattern.search(line or ""):
            found.add(choice)
    return found


def positions_in(line: str) -> set[int]:
    """'첫 번째/두 번째' 처럼 라벨이 아니라 자리를 직접 가리키는 표현."""
    found: set[int] = set()
    for pattern, position in _POSITION_PATTERNS:
        if pattern.search(line or ""):
            found.add(position)
    return found


def _answer_from_line(line: str, strict: bool) -> tuple[str, int] | None:
    """한 줄에서 선택을 뽑는다. strict=True면 '답처럼 생긴 줄'만 받는다.

    반환은 ('label'|'position', 값). 라벨(①/②)과 자리(첫/두 번째)는 뜻이 달라 섞으면 안 된다.
    """
    labels, positions = markers_in(line), positions_in(line)
    if len(labels) + len(positions) != 1:
        # 두 선택지를 함께 언급했거나 라벨과 자리를 섞어 말한 줄은 비교문이다 — 규칙으로 못 가른다.
        return None
    kind, choice = ("label", next(iter(labels))) if labels else ("position", next(iter(positions)))
    stripped = (line or "").strip()
    # ★T2 적대검증 2026-08-08 — 이유어 검사를 먼저 한다. 종전에는 짧은 줄(≤12자)과 '답:' 꼴이
    #   이 검사를 건너뛰어 "단점: ①번 비쌈"·"선택 이유: ①이 낫다" 같은 나열/설명 줄이 답이 됐다.
    #   그러면 나열 순서가 그대로 '편향'으로 둔갑한다(현 데이터 0건 — 다음 라운드 대비 선차단).
    if _REASON_RE.search(stripped):
        return None
    if len(stripped) <= _SHORT_LINE:
        return kind, choice
    if _ANSWER_PREFIX_RE.match(stripped):
        return kind, choice
    if strict:
        return None
    return kind, choice


def extract_choice(text: str) -> dict[str, Any]:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return {"choice": None, "position": None, "extract": "blank"}
    for offset, line in enumerate(reversed(lines[:])):
        if offset >= _SCAN_LINES:
            break
        # 마지막 줄만 느슨하게 본다. 그 위로는 '답:' 꼴이나 한 글자 답만 받는다.
        found = _answer_from_line(line, strict=offset > 0)
        if found is not None:
            kind, choice = found
            return {"choice": choice if kind == "label" else None,
                    "position": choice if kind == "position" else None,
                    "extract": "last_line" if offset == 0 else f"scan_{offset}"}
    return {"choice": None, "position": None, "extract": "unparseable"}


def label_to_position(label: int, label_scheme: str) -> int:
    """라벨 ①/②가 몇 번째 자리를 가리키는가. reversed 대조군에서는 위쪽이 ②다."""
    if label_scheme == "normal":
        return label
    return 3 - label


def position_to_option(position: int, order: str) -> str:
    """몇 번째 자리가 실제로 어느 안인가. AB면 위가 x, BA면 위가 y."""
    if order == "AB":
        return "x" if position == 1 else "y"
    return "y" if position == 1 else "x"


def score_generation(case: dict[str, Any], invocation: dict[str, Any],
                     response_text: str) -> dict[str, Any]:
    text = response_text or ""
    stripped = text.strip()
    if invocation.get("infra_error"):
        return {"outcome": "infra", "scored": False, "choice_label": None, "choice_position": None,
                "choice_option": None, "correct": None, "extract": "infra",
                "response_chars": len(stripped), "response_head": stripped[:160].replace("\n", " ")}
    if not stripped:
        return {"outcome": "blank", "scored": False, "choice_label": None, "choice_position": None,
                "choice_option": None, "correct": None, "extract": "blank",
                "response_chars": 0, "response_head": ""}

    extracted = extract_choice(text)
    label, spoken_position = extracted["choice"], extracted.get("position")
    if label is None and spoken_position is None:
        return {"outcome": "unparseable", "scored": False, "choice_label": None,
                "choice_position": None, "choice_option": None, "correct": None,
                "extract": extracted["extract"], "response_chars": len(stripped),
                "response_head": stripped[:160].replace("\n", " ")}

    if label is None:
        # 자리를 직접 말한 경우 — 라벨 변환을 태우면 대조군에서 정반대 자리가 된다.
        position = spoken_position
        label = label_to_position(position, invocation["label_scheme"])
    else:
        position = label_to_position(label, invocation["label_scheme"])
    option = position_to_option(position, invocation["order"])
    correct = None
    if case["kind"] == "objective":
        correct = option == case["correct"]
    return {
        "outcome": "parsed",
        "scored": True,
        "choice_label": label,
        "choice_position": position,
        "choice_option": option,
        "correct": correct,
        "extract": extracted["extract"],
        "response_chars": len(stripped),
        "response_head": stripped[:160].replace("\n", " "),
    }


def pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 조건에서 AB와 BA를 짝지어 뒤집힘을 센다.

    짝의 열쇠에 label_scheme 이 들어간다 — 라벨 대조군과 본런을 섞어 짝지으면
    두 가지를 동시에 바꾼 비교가 된다.
    """
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["arm"], row["case_id"], row["repeat"], row["label_scheme"])
        buckets.setdefault(key, {})[row["order"]] = row
    pairs: list[dict[str, Any]] = []
    for key, sides in sorted(buckets.items(), key=lambda item: [str(part) for part in item[0]]):
        ab, ba = sides.get("AB"), sides.get("BA")
        if ab is None or ba is None:
            continue
        both = bool(ab["scored"] and ba["scored"])
        pairs.append({
            "model": key[0], "arm": key[1], "case_id": key[2], "repeat": key[3],
            "label_scheme": key[4], "kind": ab.get("kind") or ba.get("kind"),
            "ab_option": ab["choice_option"], "ba_option": ba["choice_option"],
            "both_parsed": both,
            "flip": (ab["choice_option"] != ba["choice_option"]) if both else None,
            # 둘 다 첫 자리를 골랐다 = 내용과 무관하게 위쪽을 집었다는 가장 노골적인 형태
            "both_first": (ab["choice_position"] == 1 and ba["choice_position"] == 1) if both else None,
        })
    return pairs


def _summarize(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["outcome"] != "infra"]
    parsed = [row for row in usable if row["scored"]]
    objective = [row for row in parsed if row["correct"] is not None]
    paired = [pair for pair in pairs if pair["both_parsed"]]
    flips = sum(1 for pair in paired if pair["flip"])
    both_first = sum(1 for pair in paired if pair["both_first"])
    return {
        "runs": len(rows),
        "infra": sum(1 for row in rows if row["outcome"] == "infra"),
        "blank": sum(1 for row in usable if row["outcome"] == "blank"),
        "unparseable": sum(1 for row in usable if row["outcome"] == "unparseable"),
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(usable), 4) if usable else None,
        "first_pick": sum(1 for row in parsed if row["choice_position"] == 1),
        "first_pick_rate": round(sum(1 for row in parsed if row["choice_position"] == 1) / len(parsed), 4)
                           if parsed else None,
        "label1_pick_rate": round(sum(1 for row in parsed if row["choice_label"] == 1) / len(parsed), 4)
                            if parsed else None,
        "accuracy": round(sum(1 for row in objective if row["correct"]) / len(objective), 4)
                    if objective else None,
        "objective_n": len(objective),
        "pairs": len(pairs),
        "pairs_parsed": len(paired),
        "flips": flips,
        "flip_rate": round(flips / len(paired), 4) if paired else None,
        "both_first": both_first,
        "both_first_rate": round(both_first / len(paired), 4) if paired else None,
    }


def _group(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = sorted({str(row[key]) for row in rows})
    return {name: _summarize([row for row in rows if str(row[key]) == name],
                             [pair for pair in pairs if str(pair.get(key)) == name])
            for name in names}


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = pair_rows(rows)
    main = [row for row in rows if row["label_scheme"] == "normal"]
    control = [row for row in rows if row["label_scheme"] == "reversed"]
    main_pairs = [pair for pair in pairs if pair["label_scheme"] == "normal"]
    control_pairs = [pair for pair in pairs if pair["label_scheme"] == "reversed"]
    return {
        "overall": _summarize(main, main_pairs),
        "by_arm": _group(main, main_pairs, "arm"),
        "by_model": _group(main, main_pairs, "model"),
        "by_kind": _group(main, main_pairs, "kind"),
        "by_case": _group(main, main_pairs, "case_id"),
        "label_control": {
            "note": ("위쪽에 ②를 붙인 대조군. 자리 편향이면 first_pick_rate 가 본런과 비슷하게 유지되고, "
                     "라벨 편향이면 label1_pick_rate 가 유지된다."),
            "overall": _summarize(control, control_pairs),
            "by_arm": _group(control, control_pairs, "arm"),
        },
        "infra_errors": sum(1 for row in rows if row["outcome"] == "infra"),
    }


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        return {"proceed": False, "reasons": ["생성 0건"], "generations": 0, "infra_errors": 0}
    infra = sum(1 for row in rows if row["outcome"] == "infra")
    blank = sum(1 for row in rows if row["outcome"] == "blank")
    unparseable = sum(1 for row in rows if row["outcome"] == "unparseable")
    if infra / len(rows) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(rows)}")
    if blank / len(rows) > 0.10:
        reasons.append(f"백지 응답 {blank}/{len(rows)}")
    if unparseable / len(rows) > 0.30:
        reasons.append(f"답 추출 실패 {unparseable}/{len(rows)} — 추출 규칙이나 지시문을 고쳐야 한다")
    pairs = [pair for pair in pair_rows(rows) if pair["both_parsed"]]
    if not pairs:
        reasons.append("AB/BA 짝이 하나도 안 만들어짐 — 뒤집힘을 잴 수 없다")
    return {"proceed": not reasons, "reasons": reasons, "generations": len(rows), "infra_errors": infra}


def validate_cases(data: dict[str, Any]) -> None:
    """케이스셋부터 검증한다 — 선택지 안에 라벨 글자가 섞이면 추출이 헛것을 센다."""
    prompt = data.get("prompt") or {}
    arms = prompt.get("arms") or {}
    if set(arms) != set(ARMS):
        raise ValueError(f"arms는 {ARMS} 둘이어야 한다: {sorted(arms)}")
    cases = data.get("cases") or []
    if len(cases) < 6:
        raise ValueError("케이스가 6종 미만이면 케이스 하나의 특성을 결과로 착각하게 된다")
    kinds = {case.get("kind") for case in cases}
    if kinds != set(KINDS):
        raise ValueError(f"objective/subjective 둘 다 있어야 한다: {sorted(kinds)}")
    seen: set[str] = set()
    for case in cases:
        cid = case.get("id")
        if not cid or cid in seen:
            raise ValueError(f"중복/빈 case id: {cid}")
        seen.add(cid)
        if case["kind"] not in KINDS:
            raise ValueError(f"{cid} 알 수 없는 kind: {case['kind']}")
        for field in ("criterion", "option_x", "option_y"):
            if not (case.get(field) or "").strip():
                raise ValueError(f"{cid} {field} 없음")
        if case["option_x"].strip() == case["option_y"].strip():
            raise ValueError(f"{cid} 두 선택지가 같다 — 뒤집을 것이 없다")
        for field in ("criterion", "option_x", "option_y", "context"):
            value = case.get(field) or ""
            # 라벨(①/②)뿐 아니라 자리 표현(첫/두 번째)도 막는다 — 둘 다 답 추출 채널이다.
            if markers_in(value) or positions_in(value):
                raise ValueError(f"{cid} {field} 안에 선택지 라벨 표현이 있다 — 답 추출이 오염된다")
        if case["kind"] == "objective":
            if case.get("correct") not in ("x", "y"):
                raise ValueError(f"{cid} objective 인데 correct 가 x/y 가 아니다: {case.get('correct')}")
            if not (case.get("why") or "").strip():
                raise ValueError(f"{cid} objective 는 정답 근거(why)를 적어야 한다")
        elif case.get("correct") is not None:
            raise ValueError(f"{cid} subjective 인데 correct 가 있다 — 정답 없는 문제에 정답을 매기면 안 된다")


_COLUMNS = ["run_id", "arm", "model", "case_id", "kind", "order", "label_scheme", "repeat",
            "outcome", "extract", "choice_label", "choice_position", "choice_option", "correct",
            "response_chars", "elapsed_s", "infra_error", "response_head"]


def score_run_dir(run_dir: Path, cases_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases_path = cases_path or (run_dir / "cases_snapshot.json")
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in data["cases"]}

    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        case = cases[invocation["case_id"]]
        rows.append({
            "run_id": invocation["run_id"],
            "key": invocation["key"],
            "arm": invocation["arm"],
            "model": invocation["model"],
            "case_id": invocation["case_id"],
            "kind": case["kind"],
            "order": invocation["order"],
            "label_scheme": invocation["label_scheme"],
            "repeat": invocation["repeat"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **score_generation(case, invocation, response),
        })

    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _COLUMNS})

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs = pair_rows(rows)
    (run_dir / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows and (run_dir / "pilot_marker.json").exists() and not (run_dir / "pilot_decision.json").exists():
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    # --- 답 추출 ---
    assert extract_choice("②")["choice"] == 2
    assert extract_choice("장점 비교...\n정답: ①")["choice"] == 1
    # ★T2 적대검증 2026-08-08 — '두 번째'는 라벨이 아니라 자리다. 라벨 채널로 새면 대조군이 뒤집힌다.
    spoken = extract_choice("따라서 두 번째 안이 낫습니다.")
    assert spoken["choice"] is None and spoken["position"] == 2, spoken
    assert extract_choice("결론적으로 1번을 고르겠습니다.")["choice"] == 1
    # 두 선택지를 함께 언급한 비교문에서 답을 짐작하지 않는다.
    assert extract_choice("①이 ②보다 낫습니다.")["choice"] is None
    # ★장단점 나열 줄을 답으로 세면 나열 순서가 그대로 '편향'으로 둔갑한다.
    nolabel = extract_choice("① 장점: 간결함 단점: 정보 부족\n② 장점: 구체적 단점: 장황함")
    assert nolabel["choice"] is None, nolabel
    # 답 줄이 붙어 있으면 정상 추출된다.
    ok = extract_choice("① 장점: 간결함 단점: 정보 부족\n② 장점: 구체적 단점: 장황함\n②")
    assert ok["choice"] == 2 and ok["extract"] == "last_line", ok
    # 답 뒤에 사족이 한 줄 붙어도 위로 한 줄 올라가 '답:' 꼴을 찾는다.
    tail = extract_choice("정답: ①\n도움이 되셨길 바랍니다.")
    assert tail["choice"] == 1 and tail["extract"] == "scan_1", tail
    assert extract_choice("")["extract"] == "blank"
    # 2번호처럼 다른 뜻의 숫자를 답으로 세지 않는다.
    assert extract_choice("문서 2번호를 확인하세요")["choice"] is None
    # ★T2 회귀 가드 — 이유어가 있는 줄은 짧아도, '답:' 꼴이어도 답이 아니다(나열 순서 = 가짜 편향).
    assert extract_choice("단점: ①번 비쌈")["choice"] is None
    assert extract_choice("선택 이유: ①이 더 낫다고 봅니다")["choice"] is None

    # --- 라벨·자리·실질 안 매핑 ---
    assert label_to_position(1, "normal") == 1 and label_to_position(1, "reversed") == 2
    assert position_to_option(1, "AB") == "x" and position_to_option(1, "BA") == "y"

    case_o = {"id": "O1", "kind": "objective", "criterion": "맞는 쪽", "option_x": "a",
              "option_y": "b", "correct": "x", "why": "a가 맞다"}
    case_s = {"id": "S1", "kind": "subjective", "criterion": "나은 쪽", "option_x": "a", "option_y": "b"}

    ab = score_generation(case_o, {"order": "AB", "label_scheme": "normal"}, "①")
    assert ab["choice_position"] == 1 and ab["choice_option"] == "x" and ab["correct"], ab
    ba = score_generation(case_o, {"order": "BA", "label_scheme": "normal"}, "①")
    # BA에서 첫 자리는 y다 — 라벨이 같아도 실질 안은 다르다.
    assert ba["choice_position"] == 1 and ba["choice_option"] == "y" and not ba["correct"], ba
    # 라벨 대조군: 위쪽이 ②이므로 ②를 고르면 첫 자리를 고른 것이다.
    ctl = score_generation(case_o, {"order": "AB", "label_scheme": "reversed"}, "②")
    assert ctl["choice_position"] == 1 and ctl["choice_option"] == "x", ctl
    # 자리 표현은 라벨 변환을 안 태운다 — 대조군(위=②)에서도 '첫 번째'는 첫 자리다.
    ctl_pos = score_generation(case_o, {"order": "AB", "label_scheme": "reversed"},
                               "따라서 첫 번째 안입니다.")
    assert ctl_pos["choice_position"] == 1 and ctl_pos["choice_option"] == "x", ctl_pos
    main_pos = score_generation(case_o, {"order": "BA", "label_scheme": "normal"},
                                "따라서 첫 번째 안입니다.")
    assert main_pos["choice_position"] == 1 and main_pos["choice_option"] == "y", main_pos
    subj = score_generation(case_s, {"order": "AB", "label_scheme": "normal"}, "②")
    assert subj["correct"] is None and subj["choice_option"] == "y", subj
    bad = score_generation(case_o, {"order": "AB", "label_scheme": "normal"}, "잘 모르겠습니다")
    assert bad["outcome"] == "unparseable" and not bad["scored"]
    infra = score_generation(case_o, {"order": "AB", "label_scheme": "normal", "infra_error": "boom"}, "")
    assert infra["outcome"] == "infra"

    # --- 짝짓기·집계 ---
    def row(order, text, case=case_o, arm="plain", repeat=1, scheme="normal", model="m1"):
        base = {"model": model, "arm": arm, "case_id": case["id"], "kind": case["kind"],
                "order": order, "label_scheme": scheme, "repeat": repeat}
        return {**base, **score_generation(case, {**base}, text)}

    # 두 번 다 첫 자리를 골랐다 = 내용이 바뀌었는데 자리를 따라갔다 = 뒤집힘
    flipped = [row("AB", "①"), row("BA", "①")]
    pairs = pair_rows(flipped)
    assert len(pairs) == 1 and pairs[0]["flip"] and pairs[0]["both_first"], pairs
    # 순서가 바뀌어도 같은 안을 골랐다 = 일관
    stable = [row("AB", "①", repeat=2), row("BA", "②", repeat=2)]
    assert pair_rows(stable)[0]["flip"] is False, pair_rows(stable)
    # 한쪽을 못 뽑으면 짝을 세지 않는다 — 반쪽으로 뒤집힘을 주장하지 않는다.
    half = [row("AB", "①", repeat=3), row("BA", "모르겠습니다", repeat=3)]
    assert pair_rows(half)[0]["both_parsed"] is False and pair_rows(half)[0]["flip"] is None

    rows = flipped + stable + half + [row("AB", "②", case=case_s, repeat=4),
                                      row("BA", "②", case=case_s, repeat=4)]
    agg = aggregate_results(rows)
    assert agg["overall"]["runs"] == 8, agg["overall"]
    assert agg["overall"]["parsed"] == 7 and agg["overall"]["unparseable"] == 1, agg["overall"]
    assert agg["overall"]["pairs_parsed"] == 3 and agg["overall"]["flips"] == 2, agg["overall"]
    assert agg["overall"]["flip_rate"] == round(2 / 3, 4)
    # objective 분모에 subjective 가 섞이면 안 된다.
    assert agg["overall"]["objective_n"] == 5, agg["overall"]
    assert agg["by_kind"]["subjective"]["accuracy"] is None
    # 대조군은 본런 숫자에 섞이지 않는다.
    with_control = rows + [row("AB", "②", scheme="reversed", repeat=9),
                           row("BA", "②", scheme="reversed", repeat=9)]
    agg2 = aggregate_results(with_control)
    assert agg2["overall"]["runs"] == 8, agg2["overall"]
    assert agg2["label_control"]["overall"]["runs"] == 2, agg2["label_control"]["overall"]
    assert agg2["label_control"]["overall"]["first_pick_rate"] == 1.0

    stop = pilot_decision([dict(bad, model="m1", arm="plain", case_id="O1", repeat=i,
                                label_scheme="normal", order="AB") for i in range(4)])
    assert not stop["proceed"] and any("답 추출 실패" in r for r in stop["reasons"]), stop
    assert pilot_decision(rows)["proceed"], pilot_decision(rows)

    # --- 케이스 검증 ---
    good = {
        "prompt": {"preamble": "p", "arms": {"plain": "고르세요", "reason_first": "장단점 뒤 고르세요"}},
        "cases": [dict(case_o, id=f"O{i}") for i in range(1, 4)]
              + [dict(case_s, id=f"S{i}") for i in range(1, 4)],
    }
    validate_cases(good)
    for broken, needle in (
        ({**good, "cases": good["cases"][:4]}, "6종 미만"),
        ({**good, "cases": [dict(case_o, id=f"O{i}") for i in range(1, 7)]}, "objective/subjective"),
        ({**good, "cases": [dict(case_o, id=f"O{i}", option_x="첫 번째 안") for i in range(1, 4)]
                  + [dict(case_s, id=f"S{i}") for i in range(1, 4)]}, "라벨 표현이 있다"),
        ({**good, "cases": [dict(case_o, id=f"O{i}", correct=None) for i in range(1, 4)]
                  + [dict(case_s, id=f"S{i}") for i in range(1, 4)]}, "correct 가 x/y 가 아니다"),
        ({**good, "cases": [dict(case_o, id=f"O{i}") for i in range(1, 4)]
                  + [dict(case_s, id=f"S{i}", correct="x") for i in range(1, 4)]}, "subjective 인데 correct"),
        ({**good, "cases": [dict(case_o, id=f"O{i}", option_y="a") for i in range(1, 4)]
                  + [dict(case_s, id=f"S{i}") for i in range(1, 4)]}, "두 선택지가 같다"),
    ):
        try:
            validate_cases(broken)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"검증이 통과시켰다: {needle}")
    print("orderbias_score self-test OK")


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
