#!/usr/bin/env python3
"""Deterministic scorer for the multi-turn instruction-decay benchmark.

대화 첫 턴에 준 규칙 3개를 매 턴 지키는지 기계로만 판정한다. 사람이 점수를 적어 넣을 자리는 없다.

세 규칙은 서로 독립이라 끝까지 분리해서 센다 — 하나로 합치면 "지시를 잊었다"가
"셋 중 뭘 잊었는지 모르겠다"가 된다.
  prefix     답변을 '요약:' 으로 시작
  no_latin   답변에 영문 알파벳 없음
  signature  마지막 줄에 '— 가온다인 안내봇'

★잘림과 망각을 구분한다. 누적 대화가 컨텍스트 창을 넘으면 모델이 규칙을 '잊은' 게 아니라
프롬프트에서 규칙이 '사라진' 것이다. prompt_eval_count를 num_ctx와 대조해 overflow 턴을
따로 세고, 하나라도 있으면 파일럿을 세운다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

RULES = ("prefix", "no_latin", "signature")
SIGNATURE_CORE = "가온다인안내봇"

# 턴 구간 — 셀당 n이 얇아서 곡선은 구간으로 묶어 읽는다(원자료는 턴 단위로 남는다).
TURN_BUCKETS = (("t01", 1, 1), ("t02-05", 2, 5), ("t06-10", 6, 10),
                ("t11-15", 11, 15), ("t16-21", 16, 21))

_LATIN_RE = re.compile(r"[A-Za-z]")
# 굵게/머리표 같은 장식은 규칙 위반이 아니다 — 접두 판정 전에 벗겨낸다.
_LEAD_DECOR_RE = re.compile(r"^[\s*_#>\-••`\"'\[]+")
_PREFIX_RE = re.compile(r"^요약\s*[:：]")
_DASH_CHARS = "—–―‒−-"


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def check_prefix(text: str) -> bool:
    body = (text or "").lstrip()
    body = _LEAD_DECOR_RE.sub("", body)
    return bool(_PREFIX_RE.match(unicodedata.normalize("NFKC", body)))


def check_no_latin(text: str) -> bool:
    """NFKC 정규화 후 판정 — 전각 영문(Ａ-Ｚ)이나 호환 문자로 쓰면 규칙을 우회하게 된다
    (T2 적대검증 2026-08-06 지적. 현 데이터 해당 0건이지만 다음 테스트셋에서 터질 자리)."""
    return not _LATIN_RE.search(unicodedata.normalize("NFKC", text or ""))


def check_signature(text: str) -> bool:
    """마지막 비어 있지 않은 줄이 서명으로 끝나야 한다.

    대시 종류(—/–/-)나 앞뒤 장식은 따지지 않는다. 규칙의 취지는 '서명을 붙였는가'이고,
    유니코드 대시를 어느 걸 썼는지는 이 실험의 관심사가 아니다. 다만 서명이 본문 중간에만
    있고 끝에는 없는 경우는 위반으로 센다 — '마지막 줄에'가 규칙 문면이다.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    for dash in _DASH_CHARS:
        last = last.replace(dash, "")
    return _squeeze(last).endswith(SIGNATURE_CORE)


CHECKS = {"prefix": check_prefix, "no_latin": check_no_latin, "signature": check_signature}


def score_turn(invocation: dict[str, Any], response_text: str) -> dict[str, Any]:
    text = response_text or ""
    metrics = invocation.get("api_metrics") or {}
    num_ctx = invocation.get("num_ctx")
    prompt_tokens = metrics.get("prompt_eval_count")
    eval_tokens = metrics.get("eval_count") or 0
    # ★두 가지를 따로 본다(T2 적대검증 2026-08-06): 프롬프트가 창을 넘으면 규칙 문장이 잘려 나가고,
    #   프롬프트+생성이 창을 채우면 답변 끝(=서명 자리)이 잘린다. 어느 쪽이든 '망각'이 아니라 '잘림'이다.
    prompt_overflow = bool(num_ctx and prompt_tokens and prompt_tokens >= num_ctx)
    total_overflow = bool(num_ctx and prompt_tokens and prompt_tokens + eval_tokens >= num_ctx)
    # ollama가 한도에 걸려 생성을 끊으면 done_reason이 'stop'이 아니라 'length'다 — 가장 직접적인 증거.
    truncated = metrics.get("done_reason") == "length"
    overflow = prompt_overflow or total_overflow or truncated

    if invocation.get("infra_error"):
        outcome = "infra"
    elif overflow:
        # ★백지보다 먼저 판정한다 — 잘려서 아무것도 못 뱉은 회차를 'blank'로 묻으면 잘림이 은폐된다.
        outcome = "overflow"
    elif not text.strip():
        # 백지 응답은 '규칙을 어긴 답'이 아니다. 분모에 넣으면 준수율이 오염된다.
        outcome = "blank"
    else:
        outcome = "scored"

    row: dict[str, Any] = {
        "outcome": outcome,
        "scored": outcome == "scored",
        "prompt_tokens": prompt_tokens,
        "eval_tokens": eval_tokens,
        "total_tokens": (prompt_tokens + eval_tokens) if prompt_tokens else None,
        "done_reason": metrics.get("done_reason"),
        "num_ctx": num_ctx,
        "context_overflow": overflow,
        "prompt_overflow": prompt_overflow,
        "truncated": truncated,
        "response_chars": len(text.strip()),
        "response_head": text.strip()[:160].replace("\n", " ⏎ "),
    }
    for rule in RULES:
        row[rule] = CHECKS[rule](text) if outcome == "scored" else None
    row["all_rules"] = (all(row[rule] for rule in RULES) if outcome == "scored" else None)
    return row


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["scored"]]
    entry: dict[str, Any] = {
        "total": len(rows),
        "scored": len(scored),
        "blank": sum(1 for row in rows if row["outcome"] == "blank"),
        "overflow": sum(1 for row in rows if row["outcome"] == "overflow"),
        "infra": sum(1 for row in rows if row["outcome"] == "infra"),
    }
    for rule in RULES + ("all_rules",):
        kept = sum(1 for row in scored if row[rule])
        entry[rule] = {
            "kept": kept,
            "n": len(scored),
            "rate": round(kept / len(scored), 4) if scored else None,
        }
    return entry


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: _rate(bucket) for name, bucket in sorted(buckets.items())}


def _bucket_of(turn: int) -> str:
    for label, low, high in TURN_BUCKETS:
        if low <= turn <= high:
            return label
    return f"t{turn:02d}+"


def _by_arm_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        out[arm] = {label: _rate([row for row in arm_rows if _bucket_of(row["turn"]) == label])
                    for label, _, _ in TURN_BUCKETS}
    return out


def _first_violation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """대화별로 각 규칙이 처음 깨진 턴. 준수율 평균이 감추는 '언제부터'를 드러낸다."""
    convs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        convs.setdefault((row["arm"], row["model"], row["conv_id"]), []).append(row)
    out: dict[str, Any] = {}
    for (arm, model, conv_id), conv_rows in sorted(convs.items()):
        conv_rows.sort(key=lambda item: item["turn"])
        entry = {}
        for rule in RULES:
            broken = [row["turn"] for row in conv_rows if row["scored"] and not row[rule]]
            entry[rule] = broken[0] if broken else None
        entry["turns_scored"] = sum(1 for row in conv_rows if row["scored"])
        out[f"{arm}/{model}/{conv_id}"] = entry
    return out


def _survival_by_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """팔별로 '끝까지 한 번도 안 깬 대화' 수 — 평균 준수율과 다른 이야기를 한다."""
    first = _first_violation(rows)
    out: dict[str, Any] = {}
    for label, entry in first.items():
        arm = label.split("/", 1)[0]
        bucket = out.setdefault(arm, {rule: {"clean": 0, "conversations": 0, "first_violation_turns": []}
                                      for rule in RULES})
        for rule in RULES:
            bucket[rule]["conversations"] += 1
            if entry[rule] is None:
                bucket[rule]["clean"] += 1
            else:
                bucket[rule]["first_violation_turns"].append(entry[rule])
    for arm_entry in out.values():
        for rule_entry in arm_entry.values():
            turns = sorted(rule_entry["first_violation_turns"])
            rule_entry["first_violation_turns"] = turns
            rule_entry["median_first_violation"] = turns[len(turns) // 2] if turns else None
    return out


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _rate(rows),
        "by_arm": _group(rows, "arm"),
        "by_model": _group(rows, "model"),
        "by_turn_bucket": {label: _rate([row for row in rows if _bucket_of(row["turn"]) == label])
                           for label, _, _ in TURN_BUCKETS},
        "by_arm_turn_bucket": _by_arm_bucket(rows),
        "by_conversation": _group(rows, "conv_id"),
        "survival_by_arm": _survival_by_arm(rows),
        "first_violation": _first_violation(rows),
        "context": {
            "max_prompt_tokens": max((row["prompt_tokens"] or 0 for row in rows), default=0),
            # 생성분까지 합친 창 점유 최대치 — 프롬프트만 보면 답변 끝이 잘리는 경우를 놓친다.
            "max_total_tokens": max((row.get("total_tokens") or 0 for row in rows), default=0),
            "num_ctx": next((row["num_ctx"] for row in rows if row.get("num_ctx")), None),
            "num_ctx_values": sorted({row["num_ctx"] for row in rows if row.get("num_ctx")}),
            "overflow_turns": sum(1 for row in rows if row["context_overflow"]),
            "prompt_overflow_turns": sum(1 for row in rows if row.get("prompt_overflow")),
            "truncated_turns": sum(1 for row in rows if row.get("truncated")),
            "done_reason_counts": {r: sum(1 for row in rows if row.get("done_reason") == r)
                                   for r in sorted({row.get("done_reason") for row in rows} - {None})},
        },
        "infra_errors": sum(1 for row in rows if row["outcome"] == "infra"),
    }


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        return {"proceed": False, "reasons": ["회차 0건"], "turns": 0, "infra_errors": 0}
    infra = sum(1 for row in rows if row["outcome"] == "infra")
    blank = sum(1 for row in rows if row["outcome"] == "blank")
    overflow = sum(1 for row in rows if row["context_overflow"])
    if infra / len(rows) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(rows)}")
    if blank / len(rows) > 0.10:
        reasons.append(f"백지 응답 {blank}/{len(rows)} — 규칙 준수가 아니라 빈칸을 재게 된다")
    if overflow:
        reasons.append(f"컨텍스트 초과 {overflow}턴 — 망각이 아니라 잘림을 재게 된다. num_ctx를 올려라")
    # 첫 턴부터 못 지키는 규칙은 '망각'이 아니라 '애초에 못 따르는 지시'다 — 축이 성립하지 않는다.
    turn_one = [row for row in rows if row["turn"] == 1 and row["scored"]]
    for rule in RULES:
        if turn_one:
            kept = sum(1 for row in turn_one if row[rule])
            if kept / len(turn_one) < 0.5:
                reasons.append(f"1턴 {rule} 준수 {kept}/{len(turn_one)} — 지시 자체가 안 먹힌다(망각 축 불성립)")
    return {"proceed": not reasons, "reasons": reasons, "turns": len(rows), "infra_errors": infra}


_COLUMNS = ["run_id", "key", "arm", "model", "conv_id", "turn", "question", "outcome",
            "prefix", "no_latin", "signature", "all_rules", "prompt_tokens", "eval_tokens",
            "total_tokens", "done_reason", "num_ctx", "context_overflow", "truncated",
            "elapsed_s", "infra_error", "response_head"]


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "raw").glob("*-turn.json")):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        rows.append({
            "run_id": invocation["run_id"],
            "key": invocation["key"],
            "arm": invocation["arm"],
            "model": invocation["model"],
            "conv_id": invocation["conv_id"],
            "turn": invocation["turn"],
            "question": invocation["question"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **score_turn(invocation, response),
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
    good = "요약: 3층 소회의실은 총무팀에서 관리합니다.\n— 가온다인 안내봇"
    assert check_prefix(good) and check_no_latin(good) and check_signature(good)
    # 장식은 위반이 아니다.
    assert check_prefix("**요약:** 네")
    assert check_prefix("  요약 : 네")
    assert check_prefix("요약： 네")          # 전각 콜론
    assert not check_prefix("답변: 네")
    assert not check_prefix("네, 요약: 이렇습니다")   # 접두가 아니라 중간
    # 영문 한 글자라도 있으면 위반.
    assert not check_no_latin("요약: AI가 처리합니다")
    assert check_no_latin("요약: 인공지능이 처리합니다. 3층입니다.")
    # 서명은 마지막 줄에 있어야 한다.
    assert check_signature("본문\n- 가온다인 안내봇")
    assert check_signature("본문\n가온다인 안내봇")
    assert not check_signature("— 가온다인 안내봇\n본문이 뒤에 왔다")
    assert not check_signature("본문\n— 가온다인 안내")
    assert not check_signature("")
    # 서명 뒤에 군말이 붙으면 '마지막 줄에 붙인다'를 지킨 게 아니다.
    assert not check_signature("본문\n— 가온다인 안내봇 였습니다. 더 궁금하면 물어보세요")

    infra = score_turn({"infra_error": "boom"}, "")
    assert infra["outcome"] == "infra" and not infra["scored"] and infra["prefix"] is None
    blank = score_turn({}, "   ")
    assert blank["outcome"] == "blank" and not blank["scored"]
    over = score_turn({"num_ctx": 100, "api_metrics": {"prompt_eval_count": 100}}, good)
    assert over["outcome"] == "overflow" and not over["scored"] and over["context_overflow"]
    # 생성분까지 합치면 창을 넘는 회차 = 답변 끝(서명 자리)이 잘린 것 → 망각으로 세면 안 된다.
    tot = score_turn({"num_ctx": 1000, "api_metrics": {"prompt_eval_count": 900, "eval_count": 200}}, good)
    assert tot["outcome"] == "overflow" and not tot["prompt_overflow"], tot
    # done_reason=length = ollama가 한도에 걸려 끊은 것 — 가장 직접적인 잘림 증거.
    cut = score_turn({"num_ctx": 8192, "api_metrics": {"prompt_eval_count": 900, "done_reason": "length"}}, good)
    assert cut["outcome"] == "overflow" and cut["truncated"], cut
    # 잘려서 백지가 된 회차를 blank 로 묻으면 잘림이 은폐된다 — overflow 가 먼저다.
    cut_blank = score_turn({"num_ctx": 100, "api_metrics": {"prompt_eval_count": 100}}, "")
    assert cut_blank["outcome"] == "overflow", cut_blank
    # 전각 영문으로 규칙을 우회할 수 없어야 한다.
    assert not check_no_latin("요약: Ａ등급입니다\n— 가온다인 안내봇")
    assert check_no_latin("요약: 가등급입니다\n— 가온다인 안내봇")
    under = score_turn({"num_ctx": 8192, "api_metrics": {"prompt_eval_count": 900, "eval_count": 50}}, good)
    assert under["scored"] and under["all_rules"] is True
    partial = score_turn({"num_ctx": 8192, "api_metrics": {"prompt_eval_count": 900}},
                         "요약: AI 담당입니다.\n— 가온다인 안내봇")
    assert partial["prefix"] and not partial["no_latin"] and partial["signature"]
    assert partial["all_rules"] is False

    def row(arm, model, conv, turn, prefix=True, no_latin=True, signature=True, outcome="scored"):
        return {"arm": arm, "model": model, "conv_id": conv, "turn": turn, "outcome": outcome,
                "scored": outcome == "scored", "prefix": prefix, "no_latin": no_latin,
                "signature": signature, "all_rules": prefix and no_latin and signature,
                "prompt_tokens": 900, "num_ctx": 8192, "context_overflow": False}

    rows = [
        row("user_once", "m1", "cA", 1),
        row("user_once", "m1", "cA", 7, signature=False),
        row("user_once", "m1", "cA", 18, signature=False, prefix=False),
        row("system", "m1", "cA", 1),
        row("system", "m1", "cA", 7),
        row("system", "m1", "cA", 18),
    ]
    agg = aggregate_results(rows)
    assert agg["overall"]["scored"] == 6
    assert agg["by_arm"]["user_once"]["signature"]["rate"] == round(1 / 3, 4), agg["by_arm"]
    assert agg["by_arm"]["system"]["all_rules"]["rate"] == 1.0
    assert agg["by_arm_turn_bucket"]["user_once"]["t06-10"]["signature"]["rate"] == 0.0
    assert agg["first_violation"]["user_once/m1/cA"]["signature"] == 7
    assert agg["first_violation"]["user_once/m1/cA"]["prefix"] == 18
    assert agg["first_violation"]["system/m1/cA"]["prefix"] is None
    assert agg["survival_by_arm"]["system"]["signature"]["clean"] == 1
    assert agg["survival_by_arm"]["user_once"]["signature"]["median_first_violation"] == 7

    # overflow가 하나라도 있으면 파일럿을 세운다 — 잘림을 망각으로 발표하면 안 된다.
    bad_ctx = dict(row("system", "m1", "cB", 3), context_overflow=True, outcome="overflow", scored=False)
    stopped = pilot_decision(rows + [bad_ctx])
    assert not stopped["proceed"] and any("컨텍스트 초과" in r for r in stopped["reasons"]), stopped
    # 1턴부터 안 지키는 규칙 = 지시가 안 먹히는 것 → 망각 축 불성립.
    unfollowable = [row("system", "m1", f"c{i}", 1, prefix=False) for i in range(4)]
    stopped2 = pilot_decision(unfollowable)
    assert not stopped2["proceed"] and any("망각 축 불성립" in r for r in stopped2["reasons"]), stopped2
    assert pilot_decision(rows)["proceed"], pilot_decision(rows)
    print("multiturn_score self-test OK")


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
