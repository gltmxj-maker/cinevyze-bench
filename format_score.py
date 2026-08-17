#!/usr/bin/env python3
"""Score the JSON/CSV/Markdown output-format benchmark.

The scorer intentionally does not repair model output. It permits only one optional
code fence around the entire response, then parses the remaining text as a whole.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FORMATS = ("json", "csv", "markdown")
_FENCE_RE = re.compile(r"\A```(?:json|csv|markdown|md)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*\Z", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


class ParseFailure(ValueError):
    """Expected benchmark parse failure with a stable reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


def strip_optional_fence(text: str) -> str:
    """Strip one whole-response code fence; reject partial or multiple fences."""
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    match = _FENCE_RE.fullmatch(stripped)
    if not match or "```" in match.group(1):
        raise ParseFailure("wrapper_noise", "응답 전체를 감싼 단일 코드펜스가 아님")
    return match.group(1).strip()


def _split_markdown_row(line: str) -> list[str]:
    if not line.startswith("|") or not line.endswith("|"):
        raise ParseFailure("syntax", "마크다운 표 행은 양쪽 파이프가 필요함")
    cells = [cell.strip() for cell in line[1:-1].split("|")]
    if not cells or any(cell == "" for cell in cells):
        raise ParseFailure("syntax", "빈 마크다운 표 셀")
    return cells


def parse_json(text: str) -> tuple[dict[str, Any], list[str]]:
    body = strip_optional_fence(text)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ParseFailure("schema", f"JSON 중복 키: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(body, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ParseFailure("syntax", f"JSON 파싱 실패: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ParseFailure("schema", "JSON 최상위 값이 객체가 아님")
    return value, list(value.keys())


def parse_csv(text: str) -> tuple[dict[str, str], list[str]]:
    body = strip_optional_fence(text)
    try:
        rows = list(csv.reader(io.StringIO(body, newline=""), strict=True))
    except (csv.Error, UnicodeError) as exc:
        raise ParseFailure("syntax", f"CSV 파싱 실패: {exc}") from exc
    if len(rows) != 2:
        raise ParseFailure("row_count", f"CSV는 헤더+데이터 2행이어야 하나 {len(rows)}행")
    header, values = rows
    clean_header = [name.strip() for name in header]
    if (not clean_header or len(clean_header) != len(values)
            or len(set(clean_header)) != len(clean_header)):
        raise ParseFailure("schema", "CSV 열 수 불일치 또는 중복 헤더")
    if any(not name for name in clean_header):
        raise ParseFailure("schema", "CSV 빈 헤더")
    return dict(zip(clean_header, (value.strip() for value in values))), clean_header


def parse_markdown(text: str) -> tuple[dict[str, str], list[str]]:
    body = strip_optional_fence(text)
    lines = body.splitlines()
    if len(lines) != 3 or any(not line.strip() for line in lines):
        raise ParseFailure("row_count", f"마크다운 표는 정확히 3행이어야 하나 {len(lines)}행")
    header = _split_markdown_row(lines[0].strip())
    separator = _split_markdown_row(lines[1].strip())
    values = _split_markdown_row(lines[2].strip())
    if len(header) != len(separator) or len(header) != len(values):
        raise ParseFailure("schema", "마크다운 표 열 수 불일치")
    if len(set(header)) != len(header):
        raise ParseFailure("schema", "마크다운 표 중복 헤더")
    if any(not _SEPARATOR_RE.fullmatch(cell) for cell in separator):
        raise ParseFailure("syntax", "마크다운 구분선 형식 오류")
    return dict(zip(header, values)), header


def _normalise_value(actual: Any, expected: Any) -> Any:
    if isinstance(expected, bool):
        return actual if isinstance(actual, bool) else None
    if isinstance(expected, int):
        if isinstance(actual, bool):
            return None
        if isinstance(actual, int):
            return actual
        if isinstance(actual, float) and actual.is_integer():
            return int(actual)
        if isinstance(actual, str):
            raw = actual.strip()
            if re.fullmatch(r"-?\d+", raw):
                return int(raw)
            if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", raw):
                return int(raw.replace(",", ""))
        return actual
    if isinstance(expected, str):
        return actual.strip() if isinstance(actual, str) else actual
    return actual


def score_response(output_format: str, text: str, fields: list[str], expected: dict[str, Any]) -> dict[str, Any]:
    """Return parse/schema/semantic axes and a single final success bit."""
    parsers = {"json": parse_json, "csv": parse_csv, "markdown": parse_markdown}
    if output_format not in parsers:
        raise ValueError(f"알 수 없는 형식: {output_format}")

    result: dict[str, Any] = {
        "parse": False,
        "schema": False,
        "semantic": False,
        "success": False,
        "failure_reason": None,
        "failure_detail": None,
        "parsed": None,
    }
    try:
        parsed, observed_fields = parsers[output_format](text)
        result["parse"] = True
    except ParseFailure as exc:
        result["failure_reason"] = exc.reason
        result["failure_detail"] = exc.detail
        return result

    if output_format == "json":
        schema_ok = set(observed_fields) == set(fields) and len(observed_fields) == len(fields)
    else:
        schema_ok = observed_fields == fields
    if not schema_ok:
        result["failure_reason"] = "schema"
        result["failure_detail"] = f"기대 필드 {fields}, 관측 필드 {observed_fields}"
        result["parsed"] = parsed
        return result
    result["schema"] = True

    normalised = {field: _normalise_value(parsed.get(field), expected[field]) for field in fields}
    result["parsed"] = normalised
    if normalised != expected:
        result["failure_reason"] = "value"
        wrong = [field for field in fields if normalised.get(field) != expected.get(field)]
        result["failure_detail"] = f"오답 필드: {', '.join(wrong)}"
        return result

    result["semantic"] = True
    result["success"] = True
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    success = sum(bool(row[key]) for row in rows)
    low, high = wilson_interval(success, total)
    return {"success": success, "total": total, "rate": success / total if total else 0.0,
            "wilson_low": low, "wilson_high": high}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_format: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_format[row["format"]].append(row)
        by_task_format[(row["task"], row["format"])].append(row)

    aggregate: dict[str, Any] = {"total": len(results), "formats": {}, "task_formats": {}}
    for output_format in FORMATS:
        rows = by_format.get(output_format, [])
        aggregate["formats"][output_format] = {
            metric: _metric_summary(rows, metric) for metric in ("parse", "schema", "semantic", "success")
        }
        aggregate["formats"][output_format]["failure_counts"] = dict(
            Counter(row["failure_reason"] for row in rows if row["failure_reason"])
        )
    for (task, output_format), rows in sorted(by_task_format.items()):
        aggregate["task_formats"][f"{task}/{output_format}"] = {
            metric: _metric_summary(rows, metric) for metric in ("parse", "schema", "semantic", "success")
        }
    return aggregate


def pilot_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {output_format: sum(row["success"] for row in results if row["format"] == output_format)
              for output_format in FORMATS}
    totals = {output_format: sum(1 for row in results if row["format"] == output_format)
              for output_format in FORMATS}
    reasons: list[str] = []
    if all(totals[fmt] == 9 and counts[fmt] == 9 for fmt in FORMATS):
        reasons.append("ceiling_all_9_of_9")
    if all(totals[fmt] == 9 and counts[fmt] == 0 for fmt in FORMATS):
        reasons.append("floor_all_0_of_9")
    if all(totals[fmt] == 9 for fmt in FORMATS) and max(counts.values()) - min(counts.values()) <= 1:
        reasons.append("format_spread_at_most_1")

    failures = [row for row in results if not row["success"]]
    if failures and all(row["parse"] and row["schema"] and not row["semantic"] for row in failures):
        reasons.append("only_semantic_failures")
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
        response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        scored = score_response(
            invocation["format"], response, invocation["fields"], invocation["expected"]
        ) if not invocation.get("infra_error") else {
            "parse": False, "schema": False, "semantic": False, "success": False,
            "failure_reason": "infra", "failure_detail": invocation["infra_error"], "parsed": None,
        }
        rows.append({
            "run_id": invocation["run_id"],
            "task": invocation["task"],
            "case_id": invocation["case_id"],
            "format": invocation["format"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["run_id", "task", "case_id", "format", "elapsed_s", "parse", "schema",
                   "semantic", "success", "failure_reason", "failure_detail", "infra_error"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(rows) == 27:
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    fields = ["id", "quantity"]
    expected = {"id": "A1", "quantity": 1200}
    good = {
        "json": '{"id":"A1","quantity":1200}',
        "csv": "id,quantity\nA1,1200",
        "markdown": "| id | quantity |\n| --- | --- |\n| A1 | 1200 |",
    }
    for output_format, text in good.items():
        assert score_response(output_format, text, fields, expected)["success"]
        fenced = f"```{output_format if output_format != 'markdown' else 'md'}\n{text}\n```"
        assert score_response(output_format, fenced, fields, expected)["success"]
        assert not score_response(output_format, "결과입니다\n" + text, fields, expected)["success"]
        assert not score_response(output_format, text + "\n설명 끝", fields, expected)["success"]

    assert not score_response("json", "{'id':'A1','quantity':1200}", fields, expected)["parse"]
    assert not score_response("json", '{"id":"A1","quantity":1200,}', fields, expected)["parse"]
    assert score_response("json", '{"id":"A1","quantity":999}', fields, expected)["parse"]
    assert not score_response("json", '{"id":"A1","quantity":999}', fields, expected)["semantic"]
    assert not score_response("json", '{"id":"A1"}', fields, expected)["schema"]
    assert not score_response("json", '[{"id":"A1","quantity":1200}]', fields, expected)["schema"]
    assert not score_response("json", '{"id":"A1","id":"A2","quantity":1200}', fields, expected)["schema"]
    assert not score_response("json", '{"id":"A1","quantity":true}', fields, expected)["semantic"]

    assert not score_response("csv", "id,quantity\nA1,1200\nA2,3", fields, expected)["success"]
    assert not score_response("csv", "quantity,id\n1200,A1", fields, expected)["schema"]
    assert not score_response("csv", "id,quantity,extra\nA1,1200,x", fields, expected)["schema"]
    assert not score_response("csv", "id,quantity", fields, expected)["success"]
    assert not score_response("csv", "id ,id\nA1,A1", ["id", "quantity"], expected)["schema"]

    assert not score_response("markdown", "| id | quantity |\n| A1 | 1200 |", fields, expected)["parse"]
    assert not score_response("markdown", "| id | quantity |\n| -- | --- |\n| A1 | 1200 |", fields, expected)["parse"]
    assert not score_response("markdown", "| quantity | id |\n| --- | --- |\n| 1200 | A1 |", fields, expected)["schema"]
    assert score_response("markdown", "| id | quantity |\n| --- | --- |\n| A1 | 1200 |\n", fields, expected)["success"]

    assert not score_response("csv", "id,quantity\nA1,1,200", fields, expected)["semantic"]
    assert not score_response("csv", 'id,quantity\nA1,"1,2"', fields, expected)["semantic"]
    assert score_response("markdown", "| id | quantity |\n| --- | --- |\n| A1 | 1,200 |", fields, expected)["success"]
    assert score_response("json", '{"quantity":1200,"id":"A1"}', fields, expected)["success"]

    low0, high0 = wilson_interval(0, 20)
    low1, high1 = wilson_interval(20, 20)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert 0 <= low0 <= high0 <= 1
    assert 0 <= low1 <= high1 <= 1
    assert high0 > 0 and low1 < 1

    ceiling = []
    passing = []
    floor = []
    for output_format, pass_count in (("json", 9), ("csv", 8), ("markdown", 6)):
        for index in range(9):
            base = {"format": output_format, "parse": True, "schema": True,
                    "semantic": index < pass_count, "success": index < pass_count,
                    "infra_error": None}
            passing.append(base)
            ceiling.append({**base, "semantic": True, "success": True})
            floor.append({**base, "semantic": False, "success": False})
    passing[-1]["parse"] = False
    passing[-1]["schema"] = False
    assert pilot_decision(passing)["proceed"]
    assert "ceiling_all_9_of_9" in pilot_decision(ceiling)["reasons"]
    assert "floor_all_0_of_9" in pilot_decision(floor)["reasons"]
    infra = [dict(row) for row in passing]
    for row in infra[:3]:
        row["infra_error"] = "timeout"
    assert "infra_errors_over_10_percent" in pilot_decision(infra)["reasons"]
    print("format_score self-test: PASS")


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
    print(json.dumps({"runs": len(rows), "formats": aggregate["formats"],
                      "pilot_decision": aggregate.get("pilot_decision")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
