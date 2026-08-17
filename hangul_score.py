#!/usr/bin/env python3
"""Score the Hangul-vs-Latin in-image text rendering benchmark.

The character metrics reuse ocr_score's CER definition unchanged, so this benchmark and
the STT/OCR ones cannot drift apart on what counts as a character error.

Two things matter for honesty here:

1. OCR is a measuring instrument, not the truth. If Tesseract cannot read clean rendered
   Hangul, a "Hangul fails" conclusion is confounded. `calibration.json`, produced by
   run_hangul_img.py --calibrate, records the instrument's own floor on the identical
   strings, and the aggregate carries it so the reader can subtract it.
2. "No Hangul at all" and "wrong Hangul" are different failures. A model that draws Latin
   scribbles when asked for 사랑 has failed differently from one that draws malformed
   Hangul, so the script of the OCR output is recorded separately from its accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from format_score import wilson_interval
from lang_score import mcnemar_exact
from ocr_score import _lev, _norm, cer

SCRIPTS = ("ko", "en")
# "Almost right" is counted in characters, not as a CER ratio. The Korean words are 2
# characters and the English ones 4-6, so any ratio threshold would mark a single Hangul
# slip (0.5 CER) as a total failure while forgiving the same slip in English (0.17) —
# the metric itself would manufacture the gap this benchmark is trying to measure.
NEAR_MAX_CHAR_ERRORS = 1
_HANGUL_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Anime-trained SDXL checkpoints answer a Hangul request with kana or kanji often enough
# that "wrong script" and "no script" have to be told apart — collapsing them into "none"
# would hide the single most informative failure mode.
_KANA_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")


def script_of(text: str) -> str:
    """Which writing system the OCR actually saw."""
    found = [name for name, pattern in
             (("ko", _HANGUL_RE), ("en", _LATIN_RE), ("ja", _KANA_RE))
             if pattern.search(text or "")]
    if len(found) > 1:
        return "mixed"
    return found[0] if found else "none"


def score_one(requested: str, recognised: str) -> dict[str, Any]:
    """Compare one requested string against what OCR read back."""
    error = cer(requested, recognised)
    normalised_ref = _norm(requested, False)
    normalised_hyp = _norm(recognised, False)
    edits = _lev(list(normalised_ref), list(normalised_hyp))
    exact = bool(normalised_ref) and normalised_ref == normalised_hyp
    observed = script_of(recognised)
    return {
        "requested_norm": normalised_ref,
        "ocr_norm": normalised_hyp,
        "cer": round(error, 4),
        "char_errors": edits,
        "exact": exact,
        "near": bool(normalised_hyp) and edits <= NEAR_MAX_CHAR_ERRORS,
        "ocr_script": observed,
        "script_match": observed == script_of(requested),
        "empty_ocr": not normalised_hyp,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    hits = sum(bool(row.get(key)) for row in rows)
    low, high = wilson_interval(hits, total)
    return {"hits": hits, "total": total, "rate": hits / total if total else 0.0,
            "wilson_low": low, "wilson_high": high}


def _cer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["cer"] for row in rows if isinstance(row.get("cer"), (int, float))]
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {"n": len(values), "median": round(median(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4)}


def _pair_cells(results: list[dict[str, Any]]) -> tuple[list[dict[str, dict[str, Any]]], int]:
    cells: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        cells[(row["checkpoint"], row["pair_id"], row["repeat"])][row["script"]] = row
    complete, incomplete = [], 0
    for _, arms in sorted(cells.items()):
        if set(arms) == set(SCRIPTS):
            complete.append(arms)
        else:
            incomplete += 1
    return complete, incomplete


def _paired_on(cells: list[dict[str, dict[str, Any]]], key: str) -> dict[str, Any]:
    both = ko_only = en_only = neither = 0
    for arms in cells:
        ko_ok, en_ok = bool(arms["ko"][key]), bool(arms["en"][key])
        if ko_ok and en_ok:
            both += 1
        elif ko_ok:
            ko_only += 1
        elif en_ok:
            en_only += 1
        else:
            neither += 1
    return {"both": both, "ko_only": ko_only, "en_only": en_only, "neither": neither,
            "mcnemar": mcnemar_exact(ko_only, en_only)}


def paired_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the two scripts within each checkpoint/pair/repeat cell.

    Aggregate rates alone cannot say whether the scripts diverge on the same prompt or
    merely fail at similar rates on different ones, so the cells are counted explicitly.

    Two keys are paired, not one. `exact` is the headline metric but it can bottom out at
    zero on both arms, and a table of "neither, 36 times" carries no information about
    which script did better. `script_match` — did any readable glyph of the requested
    writing system appear at all — still separates the arms when `exact` cannot, so a
    floor result stays interpretable instead of silently collapsing.
    """
    cells, incomplete = _pair_cells(results)
    exact = _paired_on(cells, "exact")
    return {"cells": len(cells), "incomplete_cells": incomplete,
            "both_exact": exact["both"], "ko_only": exact["ko_only"],
            "en_only": exact["en_only"], "neither": exact["neither"],
            "mcnemar": exact["mcnemar"],
            "on_script_match": _paired_on(cells, "script_match"),
            "floor": exact["neither"] == len(cells) and bool(cells)}


def load_calibration(run_dir: Path) -> dict[str, Any] | None:
    """The OCR instrument's own floor, if it has been measured for this run."""
    path = run_dir / "calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results(results: list[dict[str, Any]], calibration: dict[str, Any] | None) -> dict[str, Any]:
    by_script: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ckpt_script: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_script[row["script"]].append(row)
        by_ckpt_script[(row["checkpoint"], row["script"])].append(row)

    aggregate: dict[str, Any] = {"total": len(results), "scripts": {}, "checkpoint_scripts": {}}
    for script in SCRIPTS:
        rows = by_script.get(script, [])
        aggregate["scripts"][script] = {
            "exact": _rate(rows, "exact"),
            "near": _rate(rows, "near"),
            "script_match": _rate(rows, "script_match"),
            "empty_ocr": _rate(rows, "empty_ocr"),
            "cer": _cer_summary(rows),
            "ocr_script_counts": dict(Counter(row["ocr_script"] for row in rows)),
            "infra_errors": sum(bool(row.get("infra_error")) for row in rows),
        }
    for (checkpoint, script), rows in sorted(by_ckpt_script.items()):
        aggregate["checkpoint_scripts"][f"{checkpoint}/{script}"] = {
            "exact": _rate(rows, "exact"), "near": _rate(rows, "near"), "cer": _cer_summary(rows),
        }
    aggregate["paired"] = paired_comparison(results)
    aggregate["ocr_calibration"] = calibration or {
        "status": "미측정 — OCR 자체 한계를 뺄 수 없으므로 절대 수치는 상한이 아님"
    }
    aggregate["reading_guide"] = [
        "scripts.*.*.wilson_* 는 각 팔을 따로 본 구간이다. 두 팔은 같은 단어쌍·같은 시드를 "
        "풀었으므로 독립 표본이 아니다 — 구간 겹침으로 '차이 없음'을 주장하지 마라.",
        "문자 체계 간 우열 판단의 근거는 paired 블록이다. exact 가 양팔 0이면(floor=true) "
        "exact 쌍비교는 정보가 없고, paired.on_script_match 가 유일하게 남는 비교축이다.",
        "repeat 는 같은 프롬프트의 재생성이므로 완전한 독립 시행이 아니다 — n=36 의 Wilson "
        "구간은 실제보다 좁다(쌍 6개 × 체크포인트 2 × 반복 3).",
    ]
    aggregate["known_limits"] = [
        "tesseract에 jpn traineddata가 없어 가나/한자 출력은 ocr_script=ja로 잡히지 않고 "
        "자모 잡음으로 흘러간다 — '한글 아님'의 내역은 이미지 육안 확인이 근거이지 이 수치가 아니다.",
        "exact/near는 OCR 판독 기준이므로, 사람이 읽을 수 있는데 OCR이 놓친 경우를 실패로 셀 수 있다 "
        "(calibration이 그 하한을 보여준다).",
    ]
    return aggregate


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for invocation_path in sorted((run_dir / "raw").glob("*-invocation.json")):
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        if invocation.get("infra_error"):
            scored = {"requested_norm": _norm(invocation["requested_text"], False), "ocr_norm": "",
                      "cer": 1.0, "char_errors": None, "exact": False, "near": False,
                      "ocr_script": "none", "script_match": False, "empty_ocr": True}
        else:
            scored = score_one(invocation["requested_text"], invocation.get("ocr_text", ""))
        rows.append({
            "run_id": invocation["run_id"],
            "checkpoint": invocation["checkpoint"],
            "pair_id": invocation["pair_id"],
            "script": invocation["script"],
            "repeat": invocation["repeat"],
            "requested_text": invocation["requested_text"],
            "ocr_text": invocation.get("ocr_text", ""),
            "seed": (invocation.get("generation") or {}).get("seed"),
            "elapsed_s": (invocation.get("generation") or {}).get("elapsed_s"),
            "image_file": invocation.get("image_file"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    (run_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["run_id", "checkpoint", "pair_id", "script", "repeat", "requested_text",
                   "ocr_text", "cer", "char_errors", "exact", "near", "ocr_script",
                   "script_match", "seed", "elapsed_s", "infra_error"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})

    aggregate = aggregate_results(rows, load_calibration(run_dir))
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows, aggregate


def self_test() -> None:
    assert script_of("사랑") == "ko"
    assert script_of("LOVE") == "en"
    assert script_of("사랑 LOVE") == "mixed"
    assert script_of("") == "none"
    assert script_of("123 !!") == "none"
    # A Hangul request answered in kana is a distinct failure from a blank image.
    assert script_of("ありがとう") == "ja"
    assert script_of("東京") == "ja"

    perfect = score_one("사랑", "사랑")
    assert perfect["exact"] and perfect["cer"] == 0.0 and perfect["script_match"]
    # OCR case and punctuation noise must not count as a rendering failure.
    assert score_one("LOVE", " love. ")["exact"]
    # One slipped character is "near" in either script — a CER threshold would have called
    # the 2-char Korean slip a total failure and the 6-char English slip a near miss.
    ko_slip, en_slip = score_one("사랑", "사탕"), score_one("SCHOOL", "SCH0OL")
    assert ko_slip["near"] and en_slip["near"] and not (ko_slip["exact"] or en_slip["exact"])
    assert ko_slip["char_errors"] == en_slip["char_errors"] == 1
    assert not score_one("사랑", "미소")["near"]
    blank = score_one("사랑", "")
    assert blank["empty_ocr"] and blank["cer"] == 1.0 and not blank["near"]
    # Asked for Hangul, got Latin: a script failure, recorded distinctly from a spelling one.
    switched = score_one("사랑", "SARANG")
    assert not switched["script_match"] and switched["ocr_script"] == "en"

    rows = [
        {"checkpoint": "A", "pair_id": "W1", "repeat": 1, "script": "ko", "exact": True},
        {"checkpoint": "A", "pair_id": "W1", "repeat": 1, "script": "en", "exact": True},
        {"checkpoint": "A", "pair_id": "W2", "repeat": 1, "script": "ko", "exact": False},
        {"checkpoint": "A", "pair_id": "W2", "repeat": 1, "script": "en", "exact": True},
        {"checkpoint": "A", "pair_id": "W3", "repeat": 1, "script": "ko", "exact": False},
        {"checkpoint": "A", "pair_id": "W3", "repeat": 1, "script": "en", "exact": False},
        {"checkpoint": "A", "pair_id": "W4", "repeat": 1, "script": "ko", "exact": True},
    ]
    for row in rows:
        row["script_match"] = row["exact"]
    paired = paired_comparison(rows)
    assert paired["cells"] == 3 and paired["incomplete_cells"] == 1
    assert paired["both_exact"] == 1 and paired["ko_only"] == 0
    assert paired["en_only"] == 1 and paired["neither"] == 1
    assert not paired["floor"]

    # The actual shape of this run: exact bottoms out on both arms. The exact pairing then
    # carries no information at all, and script_match has to be the axis that still speaks.
    floored = [
        {"checkpoint": "A", "pair_id": f"W{i}", "repeat": 1, "script": script,
         "exact": False, "script_match": script == "en"}
        for i in range(1, 4) for script in SCRIPTS
    ]
    floor_paired = paired_comparison(floored)
    assert floor_paired["floor"] and floor_paired["neither"] == 3
    assert floor_paired["mcnemar"]["discordant"] == 0
    assert floor_paired["on_script_match"]["en_only"] == 3
    assert floor_paired["on_script_match"]["mcnemar"]["p_value"] == 0.25

    rate = _rate([{"x": True}, {"x": False}], "x")
    assert rate["hits"] == 1 and rate["total"] == 2 and rate["wilson_low"] < 0.5 < rate["wilson_high"]
    assert _cer_summary([])["n"] == 0
    assert _cer_summary([{"cer": 0.0}, {"cer": 1.0}])["median"] == 0.5
    print("hangul_score self-test: PASS")


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
    print(json.dumps({"runs": len(rows), "scripts": aggregate["scripts"],
                      "paired": aggregate["paired"],
                      "ocr_calibration": aggregate["ocr_calibration"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
