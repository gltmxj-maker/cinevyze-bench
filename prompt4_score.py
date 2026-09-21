#!/usr/bin/env python3
"""prompt4 벤치 채점기 — V(막연) 대 F(4칸) 지시 준수 (2026-09-21 재제작 10번).

두 팔은 분모와 난도가 다르다(설계 2026-09-21):
- V: 최소 형태만 — 요청한 종류의 산출물이 나왔는가. 품질은 잡지 않는다. x/9 출력.
- F: F 프롬프트 [출력형식]에 적힌 명시 요구만. y/42 요구 + z/9 완전준수.
두 비율을 하나의 향상률로 빼거나 나누지 않는다. 프롬프트에 없는 축(설득력·정확성·창의성)은
채점하지 않고, 검증할 수 없는 판단은 채점기에 들어오지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCORE_VERSION = "1.0"

_SPACE_RE = re.compile(r"\s+")
_TAIL_RE = re.compile(r'[.!?]+["\')\]]*[.!?]*$')
_SENT_RE = re.compile(r'[^.!?\n]*[.!?]+["\')\]]*')


def _sentences(text: str) -> list[str]:
    """완결 문장만 = 종결부로 끝나는 연속체. 제목·목록 조각은 대상 밖."""
    return [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]


def _char_count_nospace(text: str) -> int:
    return len(_SPACE_RE.sub("", text))


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            paras.append("\n".join(current))
            current = []
    if current:
        paras.append("\n".join(current))
    return paras


def _honorific_ok(text: str) -> bool:
    # 설계 계약: 종결부의 마침표·느낌표·물음표와 닫는 따옴표/괄호를 떼고 '요'/'니다'로 끝나는지 본다.
    for sent in _sentences(text):
        tail = _TAIL_RE.sub("", sent.rstrip()).rstrip()
        if not (tail.endswith("요") or tail.endswith("니다")):
            return False
    return True


def _split_row(line: str) -> list[str]:
    # 이스케이프된 \|는 열 구분자로 세지 않는다.
    parts = re.split(r"(?<!\\)\|", line.strip())
    return [p.strip() for p in parts]


def _md_tables(text: str) -> list[list[list[str]]]:
    """마크다운 표 블록(헤더+구분행+데이터행)을 찾아 셀 격자로 반환."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if "|" in line:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    tables: list[list[list[str]]] = []
    for block in blocks:
        if len(block) >= 2 and re.fullmatch(r"\s*:?\|?[\s:|-]*\|?\s*", block[1]) and "-" in block[1]:
            tables.append([_split_row(l) for l in block])
    return tables


def _outside_table_text(text: str) -> str:
    outside: list[str] = []
    in_table = False
    for line in text.splitlines():
        if "|" in line:
            in_table = True
        else:
            if in_table and not line.strip():
                continue
            outside.append(line)
            in_table = False
    return "\n".join(outside)


# ── V 팔: 최소 형태 ──

_REFUSAL_RE = re.compile(r"죄송하지만|죄송합니다만|할 수 없|하지 못하겠|거절|도와드릴 수 없")
_EMAIL_SIGNALS = ("안녕", "제목", "님께", "보내", "메일", "드립니", "드리겠", "납품", "지연")
_TOOL_TOKENS = (
    "chatgpt", "챗gpt", "챗지피티", "gpt", "클로드", "claude", "제미나이", "gemini",
    "노션", "notion", "코파일럿", "copilot", "퍼플렉시티", "perplexity", "뤼튼", "wrtn",
    "jasper", "sudowrite", "quillbot", "퀼봇", "copy.ai", "rytr", "writesonic",
    "오픈ai", "openai", "미스트랄", "mistral", "챗봇", "글쓰기 도구", "작성 도구",
)


def _score_v(task: str, text: str) -> dict[str, Any]:
    body = (text or "").strip()
    observed = f"{len(body)}자 출력"
    if not body:
        passed, observed = False, "빈 응답"
    elif _REFUSAL_RE.search(body) and len(body) < 120:
        passed, observed = False, f"짧은 거부 응답({len(body)}자)"
    elif task == "PRM-01":
        passed = True
    elif task == "PRM-02":
        hit = [s for s in _EMAIL_SIGNALS if s in body]
        passed = bool(hit)
        observed = f"이메일 신호 {len(hit)}개" if hit else "이메일 신호 없음"
    elif task == "PRM-03":
        lower = body.lower()
        found = sorted({t for t in _TOOL_TOKENS if t in lower})
        passed = len(found) >= 2
        observed = f"도구 {len(found)}종 언급({', '.join(found[:4])})" if found else "도구 언급 없음"
    else:
        raise ValueError(f"모르는 과업: {task}")
    return {
        "requirements": [{"id": f"V{task[-2:]}_ARTIFACT", "pass": passed, "observed": observed}],
        "failed": [] if passed else [f"V{task[-2:]}_ARTIFACT"],
        "complete": passed,
    }


# ── F 팔: [출력형식] 명시 요구만 ──

def _score_f01(text: str) -> list[dict[str, Any]]:
    body = (text or "").strip()
    n_chars = _char_count_nospace(body)
    paras = _paragraphs(body)
    lines = _nonempty_lines(body)
    last = lines[-1] if lines else ""
    return [
        {"id": "F01_CHAR_RANGE", "pass": 250 <= n_chars <= 400, "observed": f"공백 제외 {n_chars}자"},
        {"id": "F01_PARAGRAPHS", "pass": len(paras) <= 3, "observed": f"문단 {len(paras)}개"},
        {"id": "F01_HONORIFIC", "pass": _honorific_ok(body), "observed": "완결 문장 존댓말 검사"},
        {"id": "F01_REQUIRED_WORD", "pass": "자영업자" in body, "observed": "'자영업자' 포함 여부"},
        {"id": "F01_LAST_LINE", "pass": last.startswith("한 줄 요약: ") and len(last.strip()) > len("한 줄 요약: "),
         "observed": f"마지막 줄: {last[:30]!r}"},
    ]


def _score_f02(text: str) -> list[dict[str, Any]]:
    body = (text or "").strip()
    lines = _nonempty_lines(body)
    title = lines[0] if lines else ""
    rest = "\n".join(lines[1:]) if len(lines) > 1 else ""
    paras = _paragraphs(rest)
    n_chars = _char_count_nospace(rest)
    return [
        {"id": "F02_TITLE_PREFIX", "pass": title.startswith("제목: "), "observed": f"첫 줄: {title[:30]!r}"},
        {"id": "F02_PARAGRAPHS", "pass": len(paras) <= 3, "observed": f"본문 문단 {len(paras)}개"},
        {"id": "F02_CHAR_MAX", "pass": n_chars <= 300, "observed": f"본문 공백 제외 {n_chars}자"},
        {"id": "F02_DATE_EXACT", "pass": "2026년 8월 10일" in body, "observed": "'2026년 8월 10일' 포함 여부"},
        {"id": "F02_HONORIFIC", "pass": _honorific_ok(rest), "observed": "본문 완결 문장 존댓말 검사"},
    ]


def _score_f03(text: str) -> list[dict[str, Any]]:
    body = (text or "").strip()
    tables = _md_tables(body)
    outside = _outside_table_text(body).strip()
    if tables:
        header = [c for c in tables[0][0] if c]
        data_rows = len([r for r in tables[0][2:] if any(c for c in r)])
        cols_obs = f"헤더 {header} · 데이터 행 {data_rows}개"
        cols_ok = header == ["도구", "장점", "단점"]
        rows_ok = data_rows == 4
    else:
        cols_obs, cols_ok, rows_ok = "표 없음", False, False
    return [
        {"id": "F03_ONE_TABLE_ONLY", "pass": len(tables) == 1, "observed": f"표 {len(tables)}개"},
        {"id": "F03_NO_PROSE", "pass": not outside, "observed": f"표 밖 비공백 {len(outside)}자"},
        {"id": "F03_COLUMNS", "pass": cols_ok, "observed": cols_obs},
        {"id": "F03_DATA_ROWS", "pass": rows_ok, "observed": cols_obs},
    ]


_F_SCORERS = {"PRM-01": _score_f01, "PRM-02": _score_f02, "PRM-03": _score_f03}


def score_one(task: str, arm: str, text: str) -> dict[str, Any]:
    """순수 함수 — 파일·마커를 건드리지 않는다."""
    if arm == "V":
        result = _score_v(task, text)
    elif arm == "F":
        checks = _F_SCORERS[task](text or "")
        result = {
            "requirements": checks,
            "failed": [c["id"] for c in checks if not c["pass"]],
            "complete": all(c["pass"] for c in checks),
        }
    else:
        raise ValueError(f"모르는 팔: {arm}")
    result["task"], result["arm"], result["score_version"] = task, arm, SCORE_VERSION
    return result


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    v_pass = sum(1 for s in scores if s["arm"] == "V" and s.get("valid", True) and s["complete"])
    v_valid = sum(1 for s in scores if s["arm"] == "V" and s.get("valid", True))
    f_reqs = [r for s in scores if s["arm"] == "F" and s.get("valid", True) for r in s["requirements"]]
    f_pass = sum(1 for r in f_reqs if r["pass"])
    f_complete = sum(1 for s in scores if s["arm"] == "F" and s.get("valid", True) and s["complete"])
    f_out = sum(1 for s in scores if s["arm"] == "F" and s.get("valid", True))
    infra = sum(1 for s in scores if not s.get("valid", True))
    by_task: dict[str, dict[str, int]] = {}
    failed_by_req: dict[str, int] = {}
    for s in scores:
        if s["arm"] != "F" or not s.get("valid", True):
            continue
        t = by_task.setdefault(s["task"], {"pass": 0, "total": 0, "complete": 0, "outputs": 0})
        for r in s["requirements"]:
            t["total"] += 1
            if r["pass"]:
                t["pass"] += 1
            else:
                failed_by_req[r["id"]] = failed_by_req.get(r["id"], 0) + 1
        t["outputs"] += 1
        if s["complete"]:
            t["complete"] += 1
    return {
        "v_artifact": f"{v_pass}/{v_valid}",
        "v_pass": v_pass, "v_valid": v_valid,
        "f_requirements": f"{f_pass}/{len(f_reqs)}",
        "f_pass": f_pass, "f_total": len(f_reqs),
        "f_complete": f"{f_complete}/{f_out}",
        "f_complete_n": f_complete, "f_outputs": f_out,
        "infra": infra,
        "by_task": by_task,
        "failed_by_requirement": failed_by_req,
    }


def score_run_dir(run_dir: Path) -> dict[str, Any]:
    scores = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((run_dir / "raw").glob("*-score.json"))]
    return aggregate(scores)


def score_legacy_dir(legacy_dir: Path) -> dict[str, Any]:
    """07-26 원시 폴더(평평한 NN-output.txt)를 같은 채점기로 다시 읽는다."""
    import yaml
    run_yaml = yaml.safe_load((legacy_dir / "run.yaml").read_text(encoding="utf-8"))
    scores = []
    for i, run in enumerate(run_yaml["runs"], start=1):
        task, arm = run["task"][:-1], run["task"][-1]
        text = (legacy_dir / f"{i:02d}-output.txt").read_text(encoding="utf-8")
        s = score_one(task, arm, text)
        s["run_id"] = f"{task}{arm}-legacy{i:02d}"
        s["valid"] = True
        scores.append(s)
    return aggregate(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="prompt4 채점기")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--legacy-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir:
        print(json.dumps(score_run_dir(args.run_dir), ensure_ascii=False, indent=2))
    elif args.legacy_dir:
        print(json.dumps(score_legacy_dir(args.legacy_dir), ensure_ascii=False, indent=2))
    else:
        parser.error("--run-dir 또는 --legacy-dir 지정")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
