#!/usr/bin/env python3
"""localcloud 벤치 채점기 — TXT-02/TXT-01/TXT-05 결정론 채점(집필 서브 설계 계약).

모든 판정은 고정 규칙표에서 나온다. 사람이 수치를 적어 넣을 자리가 없다.
ANSI/제어문자는 이미 제거된 clean_text를 받는다(원응답은 하네스가 따로 보존).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = chr(96) * 3

# ── TXT-02 코딩 ──────────────────────────────────────────────────────────────
CODING_CASES = [
    ([3, 1, 2, 3, 1], [3, 2, 1]),
    ([], []),
    ([5, 5, 5], [5]),
    ([-1, 2, -1, 0, 2], [2, 0, -1]),
    ([4, -2, 4, 7, 0, -2], [7, 4, 0, -2]),
]


def extract_code(response: str):
    labeled = re.findall(FENCE + r"(?:python|py)\n(.*?)" + FENCE, response, re.DOTALL)
    unlabeled = re.findall(FENCE + r"\n(.*?)" + FENCE, response, re.DOTALL)
    if labeled:
        return "\n".join(labeled), "labeled_blocks"
    if unlabeled:
        return "\n".join(unlabeled), "unlabeled_blocks"
    try:
        ast.parse(response)
        return response, "whole_response"
    except SyntaxError:
        return None, "unparseable"


def score_coding(response: str) -> dict:
    code, mode = extract_code(response)
    result = {
        "code_extractable": code is not None,
        "extraction_mode": mode,
        "syntax_valid": False,
        "defines_dedup_sort": False,
        "functional_cases_passed": 0,
        "functional_cases_total": len(CODING_CASES),
        "functional_pass": False,
        "example_output_shown": False,
        "strategy_class": "other",
        "code_line_count": 0,
        "explanation_present": False,
        "input_mutated": None,
        "coding_contract_pass": False,
    }
    if code is None:
        return result
    try:
        ast.parse(code)
        compile(code, "<candidate>", "exec")
        result["syntax_valid"] = True
    except SyntaxError:
        return result
    result["code_line_count"] = sum(1 for line in code.splitlines() if line.strip())
    outside = re.sub(FENCE + r".*?" + FENCE, "", response, flags=re.DOTALL)
    result["explanation_present"] = bool(re.search(r"[가-힣]", outside))
    result["example_output_shown"] = bool(re.search(r"\[\s*3\s*,\s*2\s*,\s*1\s*\]", response))

    tree = ast.parse(code)
    func = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "dedup_sort"), None)
    if func is None:
        return result
    result["defines_dedup_sort"] = True
    src = ast.unparse(func)
    if re.search(r"\bset\s*\(", src) and re.search(r"\.sort\s*\(|sorted\s*\(", src):
        result["strategy_class"] = "set_then_sort"
    elif re.search(r"\bdict\b|\.keys\s*\(", src):
        result["strategy_class"] = "dict_then_sort"
    elif re.search(r"for\s+\w+\s+in\b", src):
        result["strategy_class"] = "loop_unique_then_sort"

    passed = 0
    mutated = False
    for inp, expected in CODING_CASES:
        harness = (
            "import json\n" + code + "\n"
            "_inp = " + repr(inp) + "\n"
            "_out = dedup_sort(_inp)\n"
            "print(json.dumps({'ok': isinstance(_out, list) and _out == " + repr(expected) + ", "
            "'mutated': _inp != " + repr(inp) + "}, ensure_ascii=False))\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as tmp:
            tmp.write(harness)
            tmp_path = tmp.name
        try:
            proc = subprocess.run([sys.executable, tmp_path], capture_output=True,
                                   text=True, timeout=3)
            if proc.returncode == 0:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
                if payload.get("ok"):
                    passed += 1
                if payload.get("mutated"):
                    mutated = True
        except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError, OSError):
            pass
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    result["functional_cases_passed"] = passed
    result["functional_pass"] = passed == len(CODING_CASES)
    result["input_mutated"] = mutated
    result["coding_contract_pass"] = bool(
        result["code_extractable"] and result["syntax_valid"]
        and result["defines_dedup_sort"] and result["functional_pass"]
        and result["example_output_shown"])
    return result


# ── TXT-01 요약 ─────────────────────────────────────────────────────────────
K_SLOTS = {
    "K1_METHOD_AND_TYPES": [r"추출\s*방식", r"드립", r"에스프레소", r"침지"],
    "K2_DRIP": [r"중력", r"깔끔|산뜻|깨끗"],
    "K3_ESPRESSO": [r"고압|9\s*기압|압력", r"진하|농도|크레마"],
    "K4_IMMERSION_AND_COLDBREW": [r"침지|프렌치프레스", r"콜드브루"],
    "K5_CONTROL_VARIABLES": [r"분쇄도|분쇄", r"온도", r"시간", r"취향|조절"],
}
ALLOWED_NUMBERS_TXT01 = {"3", "4", "8", "9", "12", "25", "30", "90", "96"}

# 고정 사신표 C01~C08 키워드 버킷 — 각 문장은 하나 이상의 버킷에 연결돼야 한다.
FACT_BUCKETS = [
    r"추출|드립|에스프레소|침지|콜드브루|프렌치프레스|커피|원두",
    r"맛|산뜻|쓴|신|바디|크레마|농도|향",
    r"온도|물|중력|압력|기압|분쇄|시간|변수",
]

# 원문에 없는 방향 반전 표현(고정 모순 패턴).
CONTRADICTION_PATTERNS = [
    r"높은\s*온도.*(신맛|산미)",
    r"낮은\s*온도.*(쓴맛|쓴\s*맛)",
    r"짧은\s*시간.*곱게",
    r"오래.*(에스프레소|곱게)",
]


def _split_sentences(response: str):
    numbered = re.findall(r"^\s*\d\.\s*(.+)$", response, re.MULTILINE)
    if numbered and len(numbered) >= 3:
        return numbered, True
    parts = re.split(r"(?<=[.!?])\s+|\n+", response.strip())
    return [p for p in parts if p.strip()], False


def score_summary(response: str) -> dict:
    sentences, numbered = _split_sentences(response)
    preamble = bool(sentences and re.match(
        r"^\s*(다음은|요약입니다|핵심\s*5문장|5문장\s*요약)", sentences[0]))
    if preamble and len(sentences) > 1:
        sentences = sentences[1:]
    core = {k: all(re.search(p, response) for p in pats) for k, pats in K_SLOTS.items()}
    # 목록 번호(1. 2. …)와 머리말의 지시 에코("핵심 5문장 요약입니다")는 숫자 위반이
    # 아니다 — 설계 계약 "목록 번호를 뺀 숫자" + 머리말은 내용 문장에서 제외.
    number_source = re.sub(r"^\s*\d+[.)]\s*", "", response, flags=re.MULTILINE)
    number_source = re.sub(r"[^\n]{0,40}\d+\s*문장[^\n]{0,40}", " ", number_source)
    numbers = re.findall(r"\d+", number_source)
    unsupported_nums = sorted({n for n in numbers if n not in ALLOWED_NUMBERS_TXT01})
    contradiction_flags = [i for i, p in enumerate(CONTRADICTION_PATTERNS)
                           if re.search(p, response)]
    unmapped = [i + 1 for i, s in enumerate(sentences)
                if not any(re.search(b, s) for b in FACT_BUCKETS)]
    result = {
        "numbered_list": numbered,
        "preamble_present": preamble,
        "sentence_count": len(sentences),
        "sentence_count_pass": len(sentences) == 5,
        "core_coverage": core,
        "core_coverage_count": sum(core.values()),
        "core_coverage_pass": all(core.values()),
        "unsupported_numeric_tokens": unsupported_nums,
        "contradiction_flags": contradiction_flags,
        "unmapped_sentences": unmapped,
        "unsupported_content_tokens": [],
        "grounding_rule_pass": not unsupported_nums and not contradiction_flags and not unmapped,
    }
    result["summary_contract_pass"] = bool(
        result["sentence_count_pass"] and result["core_coverage_pass"]
        and result["grounding_rule_pass"])
    return result


# ── TXT-05 번역 ─────────────────────────────────────────────────────────────
TERMS = {
    "T_LLM": ["대규모 언어 모델"],
    "T_LATENCY": ["지연 시간", "대기 시간", "응답 시간"],
    "T_THROUGHPUT": ["처리량"],
    "T_BATCHING": ["배치 처리", "배칭", "일괄 처리"],
    "T_QUANTIZATION": ["양자화"],
    "T_WEIGHTS": ["가중치", "모델 가중치"],
    "T_INFERENCE": ["추론"],
    "T_CONTEXT_WINDOW": ["컨텍스트 윈도우", "문맥 창"],
    "T_TOKEN": ["토큰"],
}
CLAUSE_PATTERNS = {
    "P1_BALANCE": [r"균형"],
    "P2_DEFINITIONS": [r"지연\s*시간|대기\s*시간|응답\s*시간", r"처리량|초당"],
    "P3_BATCH_BENEFIT": [r"묶어|일괄|배치|여러\s*요청"],
    "P4_BATCH_COST": [r"과도|지연\s*시간.*(증가|늘)|늘어날|증가시킬"],
    "P5_QUANTIZATION": [r"양자화", r"메모리"],
    "P6_QUALITY_COST": [r"품질.*(손실|저하|손해)|감수"],
    "P7_CONTEXT_COST": [r"프롬프트|컨텍스트", r"GPU|메모리|시간"],
}


def score_translation(response: str) -> dict:
    term_occurrences = {}
    term_choices = {}
    mixed = []
    missing = []
    for term_id, variants in TERMS.items():
        # 포함 관계 보정: "모델 가중치"는 "가중치"를 포함한다 — 긴 표기의 범위를 먼저
        # 덜어낸 뒤 짧은 표기를 세야 같은 단어를 두 표기로 섞은 것으로 오탐하지 않는다.
        text = response
        found = {}
        for v in sorted(variants, key=len, reverse=True):
            hits = [m for m in re.finditer(re.escape(v), text)]
            if hits:
                found[v] = len(hits)
                text = re.sub(re.escape(v), " ", text)
        term_occurrences[term_id] = sum(found.values())
        term_choices[term_id] = list(found)
        if not found:
            missing.append(term_id)
        elif len(found) > 1:
            mixed.append(term_id)
    clauses = {}
    omitted = []
    for clause_id, pats in CLAUSE_PATTERNS.items():
        ok = all(re.search(p, response) for p in pats)
        clauses[clause_id] = ok
        if not ok:
            omitted.append(clause_id)
    numbers = re.findall(r"\d+", response)
    unsupported_nums = sorted(set(numbers))
    result = {
        "term_choice": term_choices,
        "term_occurrences": term_occurrences,
        "term_coverage_count": 9 - len(missing),
        "missing_term_ids": missing,
        "mixed_term_ids": mixed,
        "term_coverage_pass": not missing,
        "term_consistency_pass": not mixed,
        "clause_coverage": clauses,
        "clause_coverage_count": sum(clauses.values()),
        "omitted_clause_ids": omitted,
        "clause_coverage_pass": all(clauses.values()),
        "unsupported_numeric_tokens": unsupported_nums,
        "unsupported_numeric_pass": not unsupported_nums,
    }
    result["translation_contract_pass"] = bool(
        result["term_coverage_pass"] and result["term_consistency_pass"]
        and result["clause_coverage_pass"] and result["unsupported_numeric_pass"])
    return result


def score_task(task_id: str, clean_text: str) -> dict:
    if task_id == "TXT-02":
        return score_coding(clean_text)
    if task_id == "TXT-01":
        return score_summary(clean_text)
    if task_id == "TXT-05":
        return score_translation(clean_text)
    raise ValueError("모르는 과업: " + task_id)
