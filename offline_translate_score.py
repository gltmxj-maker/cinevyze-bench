#!/usr/bin/env python3
"""오프라인 번역 지시 2x2 벤치 채점기 — 2026-09-22 재제작 13번.

★수치의 write-origin = harness(run_offline_translate_bench.py) 출력. 이 스크립트는 그
출력을 '채점'만 한다. 채점 규칙은 하네스가 실행 전에 고정하는 score-spec.json과 같은
값을 하드코딩해 결정론적으로 판정한다 — 사람이 통과 여부를 적어 넣을 자리가 없다.

두 축:
- 내용 보존 계약(모든 팔): clause_coverage · term_coverage · numeric_fidelity ·
  unsupported_number (polarity 는 조항 커버리지로 갈음 — score-spec 참조)
- 지시 준수 계약(팔별): translation_only · glossary_exact(G/GS) · register(S/GS)

self-test: python3 offline_translate_score.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ARMS = ("B", "G", "S", "GS")
TASKS = ("T", "C", "E")
TASK_NAMES = {"T": "기술 문서", "C": "캐주얼 후기", "E": "비즈니스 메일"}
ARM_NAMES = {"B": "기본 지시", "G": "용어 고정", "S": "문체 고정", "GS": "결합"}

# 허용 번역(모든 팔) — 개념이 검출됐는지만 본다.
GLOSSARY = {
    "T": {
        "large language model": ["대규모 언어 모델"],
        "latency": ["지연 시간", "대기 시간", "레이턴시", "지연시간"],
        "throughput": ["처리량"],
        "batching": ["배치", "묶어", "일괄"],
        "quantization": ["양자화"],
        "weights": ["가중치"],
        "inference": ["추론"],
    },
    "C": {
        "setup instructions": ["설치 안내", "설명서", "설치 설명서"],
        "battery": ["배터리"],
        "two weeks": ["2주", "이 주", "두 주", "2 주"],
        "keeper": ["계속 쓸 만한 제품", "좋은 제품", "제품"],
    },
    "E": {
        "address discrepancy": ["주소 불일치", "주소 오류"],
        "regional facility": ["지역 시설", "지역 창고", "지역 센터", "해당 지역"],
        "three business days": ["영업일 기준 3일", "3영업일", "영업일 3일", "세 영업일", "영업일 기준으로 3일", "3일 이내"],
        "10 percent credit": ["10% 크레딧", "10 퍼센트 크레딧", "10%의 크레딧", "10퍼센트 크레딧", "10% 크레딧을", "10 퍼센트의 크레딧"],
    },
}
# G/GS 팔이 '정확히 이 표기'를 요구하는 항목(실행 전 고정).
GLOSSARY_EXACT = {
    "T": {
        "large language model": "대규모 언어 모델",
        "latency": "지연 시간",
        "throughput": "처리량",
        "batching": "배치",
        "quantization": "양자화",
        "weights": "가중치",
        "inference": "추론",
    },
    "C": {
        "setup instructions": "설치 안내",
        "battery": "배터리",
        "two weeks": "2주",
        "keeper": "계속 쓸 만한 제품",
    },
    "E": {
        "address discrepancy": "주소 불일치",
        "regional facility": "지역 시설",
        "three business days": "영업일 기준 3일",
        "10 percent credit": "10% 크레딧",
    },
}
CLAUSES = {
    "T": [
        ["균형", "밸런스"],
        ["지연 시간", "대기 시간", "레이턴시"],
        ["처리량"],
        ["배치", "묶어", "일괄"],
        ["양자화"],
        ["품질"],
    ],
    "C": [
        ["반품"],
        ["설치 안내", "설명서", "설치 설명서"],
        ["한 시간", "1시간"],
        ["배터리"],
        ["2주", "이 주", "두 주"],
        ["추천"],
    ],
    "E": [
        ["지연", "늦어", "배송 지연"],
        ["주소 불일치", "주소 오류"],
        ["지역 시설", "지역 창고", "해당 지역"],
        ["3일", "세 영업일", "영업일"],
        ["10%", "10 퍼센트", "10퍼센트"],
        ["문의", "연락"],
    ],
}
NUMERIC = {
    "T": {"required": [], "forbidden_any_number": True},
    "C": {"required": ["한 시간|1시간", "2주|이 주|두 주"], "forbidden_any_number": False},
    "E": {"required": ["3일|영업일 기준 3일|영업일 3일", "10%|10 퍼센트|10퍼센트"], "forbidden_any_number": False},
}
REGISTER_RULES = {"T": "다", "C": "요", "E": "습니다"}
REGISTER_TAILS = {
    # 실제 한국어 종결 어미(2026-09-22 수修: '습니다' 문자열만 endswith 하면
    # '감사합니다'(사+합니다)·'예정입니다'(정+입니다) 같은 정상 종결을 놓쳤다)
    "T": ("다.", "다"),
    "C": ("요.", "요", "죠.", "죠", "까요?", "네요.", "네요"),
    "E": ("합니다", "입니다", "됩니다", "바랍니다", "갑니다", "습니다", "십니다"),
}
NUM_RE = re.compile(r"\d+")
PRELUDE_RE = re.compile(r"^(번역|translation|결과|result|출력|output)\s*[:：]\s*", re.IGNORECASE)
# 문장 경계 = 문장부호만(2026-09-22 수修 3: '다/요' 뒤 공백에서도 잘랐더니
# "생각보다 훨씬 오래가고"의 부사구 '생각보다'가 완결문으로 오탐돼 해요체 검사가
# 깨졌다. 완결문은 문장부호로 끝난다 — 부호 없는 긴 문장은 한 문장으로 센다.)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\r\n", " ").replace("\n", " ").strip()
    return [p.strip() for p in SENT_SPLIT_RE.split(cleaned) if p.strip()]


def _term_coverage(task: str, text: str, exact: bool) -> dict[str, Any]:
    table = GLOSSARY_EXACT if exact else GLOSSARY
    per_term: dict[str, bool] = {}
    for term in table[task]:
        # exact 팔은 '지정 한국어 표기'를, 기본 팔은 '허용 번역 아무거나'를 찾는다.
        # (구현 결함 수修 2026-09-22: exact 에서 영어 키를 그대로 찾아 전부 False 였음)
        candidates = [table[task][term]] if exact else table[task][term]
        per_term[term] = any(c in text for c in candidates)
    return {"per_term": per_term, "pass": all(per_term.values()),
            "matched": sum(per_term.values()), "total": len(per_term)}


def _clause_coverage(task: str, text: str) -> dict[str, Any]:
    clauses = [{"id": f"C{idx+1}", "hit": any(a in text for a in alts)}
               for idx, alts in enumerate(CLAUSES[task])]
    return {"clauses": clauses, "pass": all(c["hit"] for c in clauses),
            "hit": sum(c["hit"] for c in clauses), "total": len(clauses)}


def _numeric_fidelity(task: str, text: str) -> dict[str, Any]:
    missing = [pat for pat in NUMERIC[task]["required"] if not re.search(pat, text)]
    return {"missing": missing, "pass": not missing}


def _unsupported_number(task: str, text: str) -> dict[str, Any]:
    if not NUMERIC[task]["forbidden_any_number"]:
        return {"pass": True, "note": "과업 허용"}
    nums = NUM_RE.findall(text)
    return {"pass": not nums, "numbers": nums}


def _translation_only(text: str) -> dict[str, Any]:
    stripped = text.strip()
    reasons = []
    if PRELUDE_RE.match(stripped):
        # '번역:' 머리말은 지시 위반이다 — 다만 본문 자체는 strip 후 채점해 판정 근거를 남긴다.
        reasons.append("머리말(번역: 등)")
    if stripped.startswith("```") or stripped.endswith("```"):
        reasons.append("코드펜스")
    if re.search(r"(죄송|미안|참고로|설명드리|아래는|here is|다음은)", stripped, re.IGNORECASE):
        reasons.append("설명 문구")
    return {"pass": not reasons, "reasons": reasons,
            "text": PRELUDE_RE.sub("", stripped, count=1)}


def _register_check(task: str, text: str) -> dict[str, Any]:
    tails = REGISTER_TAILS[task]
    ok = True
    sentences = split_sentences(text)
    checked = []
    for sentence in sentences:
        tail = sentence.rstrip()
        bare = tail.rstrip(".!?")
        hit = any(bare.endswith(t) or tail.endswith(t) for t in tails)
        checked.append({"sentence": sentence[:40], "tail": tail[-6:], "hit": hit})
        ok = ok and hit
    return {"pass": bool(sentences) and ok, "sentences": checked,
            "rule": f"모든 완결문 종결이 {REGISTER_RULES[task]}계열"}


def score_response(arm: str, task: str, response_text: str) -> dict[str, Any]:
    if not response_text.strip():
        return {"valid": False, "reason": "empty_response"}
    only = _translation_only(response_text)
    body = only["text"]
    term = _term_coverage(task, body, exact=arm in ("G", "GS"))
    clause = _clause_coverage(task, body)
    numeric = _numeric_fidelity(task, body)
    unsupported = _unsupported_number(task, body)
    content_pass = (term["pass"] and clause["pass"] and numeric["pass"] and unsupported["pass"])
    register_detail = _register_check(task, body) if arm in ("S", "GS") else None
    instruction_parts = {"translation_only": only["pass"]}
    if arm in ("G", "GS"):
        instruction_parts["glossary_exact"] = term["pass"]
    if arm in ("S", "GS"):
        instruction_parts["register"] = register_detail["pass"]
    instruction_pass = all(instruction_parts.values())
    return {
        "valid": True,
        "arm": arm, "task": task,
        "content_contract_pass": content_pass,
        "instruction_contract_pass": instruction_pass,
        "full_contract_pass": content_pass and instruction_pass,
        "term_coverage": term,
        "clause_coverage": clause,
        "numeric_fidelity": numeric,
        "unsupported_number": unsupported,
        "translation_only": only,
        "register": register_detail,
        "register_pass": register_detail["pass"] if register_detail else None,
    }


def score_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for path in sorted(run_dir.glob("raw/*-score.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        results.append(row)
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [r for r in results if r.get("arm") == arm and r.get("valid")]
        per_task = {}
        for task in TASKS:
            task_rows = [r for r in rows if r.get("task") == task]
            per_task[task] = {
                "full": sum(1 for r in task_rows if r.get("full_contract_pass")),
                "content": sum(1 for r in task_rows if r.get("content_contract_pass")),
                "instruction": sum(1 for r in task_rows if r.get("instruction_contract_pass")),
                "n": len(task_rows),
            }
        by_arm[arm] = {
            "name": ARM_NAMES[arm],
            "full_pass": sum(1 for r in rows if r.get("full_contract_pass")),
            "content_pass": sum(1 for r in rows if r.get("content_contract_pass")),
            "instruction_pass": sum(1 for r in rows if r.get("instruction_contract_pass")),
            "n": len(rows),
            "per_task": per_task,
        }
    aggregate = {"arms": by_arm, "valid_runs": len(results),
                 "invalid_runs": sum(1 for r in results if not r.get("valid"))}
    return results, aggregate


def _self_test() -> int:
    # T 기본 팔: 6월 실측 출력(정상 번역문)은 내용 계약을 통과해야 한다.
    june_t = ("대규모 언어 모델을 운영하는 것은 대기 시간과 처리량 사이의 균형을 맞추는 일이다. "
              "대기 시간은 단일 사용자가 응답을 대기하는 시간을 의미하고, 처리량은 시스템이 초당 처리하는 "
              "요청 수를 측정한다. 동일 GPU에서 더 많은 사용자를 지원하기 위해 엔지니어들은 여러 요청을 "
              "묶어서 처리하지만, 과도한 묶음은 개별 사용자가 경험하는 대기 시간을 증가시킬 수 있다. "
              "양자화는 모델 가중치의 숫자 정밀도를 낮춰 메모리 사용량을 줄이고 추론 속도를 높이는 "
              "방식으로, 출력 품질에 약간의 손실을 발생시킬 수 있다.")
    scored = score_response("B", "T", june_t)
    assert scored["valid"] and scored["content_contract_pass"], ("6월 T 정상본 내용 실패", scored)
    assert scored["instruction_contract_pass"], ("6월 T 정상본 지시 실패", scored)
    # 용어 고정 팔: '대기 시간'이 아니라 '지연 시간'을 요구 — 6월 출력은 실패해야 한다.
    g = score_response("G", "T", june_t)
    assert not g["term_coverage"]["pass"], ("G 팔이 지정 표기 아닌 것을 통과시킴", g)
    # G 팔 정상 케이스: 지정 표기가 모두 있으면 통과해야 한다(2026-09-22 구현 결함 회귀 검사).
    g_ok_text = june_t.replace("대기 시간", "지연 시간").replace("묶어서", "배치 처리해")
    g_ok = score_response("G", "T", g_ok_text)
    assert g_ok["term_coverage"]["pass"], ("G 팔 정상 표기를 실패시킴", g_ok)
    # 빈 응답 = invalid.
    empty = score_response("B", "C", "")
    assert not empty["valid"]
    # 머리말·설명 문구 = translation_only 실패.
    with_prelude = score_response("B", "T", "번역: " + june_t)
    assert not with_prelude["instruction_contract_pass"], ("머리말 미검출", with_prelude)
    # 문체 팔: 해요체 문장은 E 과업 습니다체 검사에서 실패해야 한다.
    casual = "설명서가 완전 엉망이었어요. 배터리가 오래가요. 추천해요."
    reg = score_response("S", "E", casual)
    assert not reg["instruction_contract_pass"], ("문체 검사 오탐", reg)
    # E 숫자 누락 검사.
    no_num = score_response("B", "E", "주소 불일치로 지연됐고 문의 감사합니다. 지역 시설에서 문제를 해결했습니다. 크레딧을 적용했습니다.")
    assert not no_num["numeric_fidelity"]["pass"], ("숫자 누락 미검출", no_num)
    print("[self-test] offline_translate_score 6/6 통과")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.run_dir:
        parser.error("--run-dir 또는 --self-test 필요")
    results, aggregate = score_run_dir(args.run_dir)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
