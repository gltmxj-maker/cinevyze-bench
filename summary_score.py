#!/usr/bin/env python3
"""문서요약 벤치 채점기 — D1~D4 x S3/S5/S8 결정론 채점(집필 서브 Schrodinger 설계 계약 2026-09-22).

모든 판정은 문서별 manifest(고정 규칙표)에서 나온다. 사람이 수치를 적어 넣을 자리가 없다.
- core_slots: 슬롯별 앵커 그룹(전부 매치되어야 통과)
- allowed_numbers: 원문에 있는 숫자의 정규화 집합
- contradiction_patterns: 원문과 반대 방향 패턴
- retired_fact_patterns: 철회·폐기 사실을 최종 사실처럼 쓰는 패턴(표지 단어 동반 시 면제)
- fact_buckets: 문장이 연결될 수 있는 사실 버킷

D1은 localcloud_score.score_summary 와 같은 규칙을 유지한다(7월 런과의 회귀 기준선).
"""
from __future__ import annotations

import re

# ── 문장 분리(기존 score_summary 규칙 유지) ─────────────────────────────
def split_sentences(response: str):
    numbered = re.findall(r"^\s*\d\.\s*(.+)$", response, re.MULTILINE)
    if numbered and len(numbered) >= 3:
        return numbered, True
    parts = re.split(r"(?<=[.!?])\s+|\n+", response.strip())
    return [p for p in parts if p.strip()], False

PREAMBLE_RE = re.compile(r"^\s*(다음은|요약입니다|핵심\s*\d+문장|\d+문장\s*요약|제시해|위s*글)")

# ── 문서별 manifest ────────────────────────────────────────────────────
MANIFESTS = {
    "D1": {
        "core_slots": {
            "K1_METHOD_AND_TYPES": [r"추출\s*방식", r"드립", r"에스프레소", r"침지"],
            "K2_DRIP": [r"중력", r"깔끔|산뜨|깨끗"],
            "K3_ESPRESSO": [r"고압|9\s*기압|압력", r"진하|농도|크레마"],
            "K4_IMMERSION_AND_COLDBREW": [r"침지|프렌치프레스", r"콜드브루"],
            "K5_CONTROL_VARIABLES": [r"분쇄도|분쇄", r"온도", r"시간", r"취향|조절"],
        },
        "allowed_numbers": {"3", "4", "8", "9", "12", "25", "30", "90", "96"},
        "contradiction_patterns": [
            r"높은\s*온도.*(신맛|산미)",
            r"낮은\s*온도.*(쓴맛|쓴\s*맛)",
            r"짧은\s*시간.*곱게",
            r"오래.*(에스프레소|곱게)",
        ],
        "retired_fact_patterns": [],
        "retired_marker_words": [],
        "fact_buckets": [
            r"추출|드립|에스프레소|침지|콜드브루|프렌치프레스|커피|원두",
            r"맛|산뜨|쓴|신|바디|크레마|농도|향",
            r"온도|물|중력|압력|기압|분쇄|시간|변수",
        ],
    },
    "D2": {
        "core_slots": {
            "K1_IMPACT": [r"240", r"37"],
            "K2_HYPOTHESIS_REJECTED": [r"네트워크\s*지연", r"기각|원인이\s*아니|아니었|아님"],
            "K3_REAL_CAUSE": [r"만료", r"인증서"],
            "K4_RECOVERY": [r"롤백", r"12"],
            "K5_NO_LOSS_ROTATION": [r"유실", r"14"],
        },
        "allowed_numbers": {"1", "2", "5", "8", "9", "10", "11", "12", "14", "15", "20", "30", "37", "40", "42", "45", "50", "240"},
        "contradiction_patterns": [
            # 긍정형 유실 주장만 모순 — "발생하지 않았/없었/없이" 는 unless 가드로 면제(2026-09-22 오탐 수리)
            {"pattern": r"유실[^.\n]{0,30}(있었|발생했|발생하였|생겼)",
             "unless": ["않", "없", "미"]},
        ],
        "retired_fact_patterns": [
            r"네트워크\s*지연.{0,40}(원인|때문)",
        ],
        "retired_marker_words": ["가설", "추정", "의심", "기각", "아니", "오류", "판명", "초반", "처음", "1차"],
        "fact_buckets": [
            r"서명|오류|장애|인증서|롤백|재시도|유실|회전|문서|복구|배치",
            r"네트워크|지연|응답|모니터링|경보|알림|검증|게이트웨이",
            r"사용자|담당|시간|가설|원인|점검|교훈|온콜",
        ],
    },
    "D3": {
        "core_slots": {
            "K1_BASIC": [r"Basic|베이직", r"10\s*문서", r"7\s*일|7일"],
            "K2_PRO": [r"Pro|프로", r"200\s*문서", r"30\s*일|30일"],
            "K3_TEAM": [r"Team|팀", r"1,000\s*문서|1000\s*문서|1,000문서", r"90\s*일|90일"],
            "K4_OCR_RANGE": [r"OCR", r"Pro.*Team|Team.*Pro|Pro와\s*Team|Pro와 Team"],
            "K5_DELETE_DISCOUNT": [r"24\s*시간|24시간", r"연간", r"20\s*%|20%"],
        },
        "allowed_numbers": {"0", "1", "2", "3", "5", "7", "9", "10", "11", "20", "24", "30", "50", "90", "190", "200", "950", "1000", "12900", "29900", "2025", "2026"},
        "contradiction_patterns": [
            # Basic OCR 제공 주장 — "제공하지 않" 은 면제
            {"pattern": r"Basic[^.\n]{0,60}OCR[^.\n]{0,60}(제공|지원)",
             "unless": ["않", "없", "미"]},
            # Team 연간 할인 적용 주장 — 정확히 "Team(에게) 할인이 적용/있음" 형태만 잡는다.
            # 2026-09-22 v2·v3 연속 오탐 수리: "Team에서만 제공되고 할인은 Pro에만" 같은 정상 문장은
            # unless 단어 나열로 못 막는다 — Team 과 할인이 한 문장에 공존하는 올바른 요약이 흔하다.
            {"pattern": r"Team\s*요금제[도는이에게의]?(?![^.\n]{0,50}(제공되[고며]|지원되[고며]|차감))[^\n]{0,40}(할인이\s*(적용|있)|할인\s*적용)",
             "unless": ["않", "없", "제외"]},
        ],
        "retired_fact_patterns": [],
        "retired_marker_words": [],
        "fact_buckets": [
            r"Basic|Pro|Team|요금|요금제|가격|원",
            r"문서|보관|한도|삭제|OCR|MB|파일",
            r"결제|할인|연간|환불|이동|개정|이력",
        ],
    },
    "D4": {
        "core_slots": {
            "K1_FINAL_SAMPLE": [r"480"],
            "K2_OVERALL_ACC": [r"71\.2", r"74\.8"],
            "K3_SUBGROUP_B": [r"68\.4", r"65\.1"],
            "K4_LATENCY_COST": [r"420", r"610", r"18\s*%|18%"],
            "K5_HOLD_REEVAL": [r"보류", r"1,000|1000"],
        },
        "allowed_numbers": {"1", "2", "3", "4", "6", "7", "18", "400", "420", "480", "610", "1000", "3.3", "3.6", "65.1", "68.4", "71.2", "74.8", "76.1", "2026"},
        "contradiction_patterns": [
            # 2026-09-22 v4 오탐 수리: "결정할 수 없다/없으며"(교훈·부정)는 모순이 아니다.
            {"pattern": r"(배포|도입|교체)(하기로|했다|를 결정|를 승인)",
             "unless": ["않", "보류", "미", "없"]},
        ],
        "retired_fact_patterns": [
            r"76\.1.{0,50}(정확도|였|이다)",
            r"400건.{0,50}(표본|평가|였|이다)",
        ],
        "retired_marker_words": ["초안", "폐기", "구버전", "과정", "오류", "남겨", "기록"],
        "fact_buckets": [
            r"모델|평가|정확도|배포|보류|검토|재평가",
            r"하위집단|지연|비용|신고|트래픽|호출",
            r"표본|검수|측정|환경|요금|결정|회의|한계|교훈",
        ],
    },
}

TARGETS = {"S3": 3, "S5": 5, "S8": 8}

def normalize_numbers(text: str) -> str:
    """숫자 정규화 — 쉼표 제거(1,000→1000). 소수점은 유지."""
    return re.sub(r"(\d),(\d{3})", r"\1\2", text)

def _strip_number_violation_sources(response: str) -> str:
    """목록 번호와 'N문장' 지시 에코를 숫자 위반 검사에서 제외(설계 계약)."""
    src = re.sub(r"^\s*\d+[.)]\s*", "", response, flags=re.MULTILINE)
    src = re.sub(r"[^\n]{0,40}\d+\s*문장[^\n]{0,40}", " ", src)
    return src

def score_response(doc: str, arm: str, clean_text: str) -> dict:
    if doc not in MANIFESTS:
        raise ValueError("모르는 문서: " + doc)
    if arm not in TARGETS:
        raise ValueError("모르는 팔: " + arm)
    m = MANIFESTS[doc]
    target = TARGETS[arm]

    sentences, numbered = split_sentences(clean_text)
    preamble = bool(sentences and PREAMBLE_RE.match(sentences[0]))
    if preamble and len(sentences) > 1:
        sentences = sentences[1:]

    core = {k: all(re.search(p, clean_text) for p in pats)
            for k, pats in m["core_slots"].items()}

    num_src = _strip_number_violation_sources(normalize_numbers(clean_text))
    numbers = re.findall(r"\d+\.?\d*", num_src)
    allowed = m["allowed_numbers"]
    unsupported = sorted({n for n in numbers if n not in allowed})

    contradiction_flags = []
    for i, rule in enumerate(m["contradiction_patterns"]):
        pat = rule["pattern"] if isinstance(rule, dict) else rule
        guards = rule.get("unless", []) if isinstance(rule, dict) else []
        hit = False
        for sent in sentences:
            if re.search(pat, sent) and not any(g in sent for g in guards):
                hit = True
                break
        if hit:
            contradiction_flags.append(i)

    # 폐기 사실 판정 — retired 패턴이 매치된 문장에 표지 단어가 없으면 채택으로 본다.
    retired_flags = []
    for i, pat in enumerate(m["retired_fact_patterns"]):
        for sent in sentences:
            if re.search(pat, sent):
                has_marker = any(w in sent for w in m["retired_marker_words"])
                if not has_marker:
                    retired_flags.append(i)
                break

    unmapped = [i + 1 for i, s in enumerate(sentences)
                if not any(re.search(b, s) for b in m["fact_buckets"])]

    result = {
        "doc": doc,
        "arm": arm,
        "target_sentence_count": target,
        "numbered_list": numbered,
        "preamble_present": preamble,
        "sentence_count": len(sentences),
        "sentence_count_pass": len(sentences) == target,
        "core_coverage": core,
        "core_coverage_count": sum(core.values()),
        "missing_slot_ids": [k for k, v in core.items() if not v],
        "core_coverage_pass": all(core.values()),
        "unsupported_numeric_tokens": unsupported,
        "unsupported_numeric_count": len(unsupported),
        "contradiction_flags": contradiction_flags,
        "retired_fact_flags": retired_flags,
        "contradiction_retired_count": len(contradiction_flags) + len(retired_flags),
        "unmapped_sentences": unmapped,
        "unmapped_count": len(unmapped),
    }
    result["grounding_rule_pass"] = (not unsupported and not contradiction_flags
                                     and not retired_flags and not unmapped)
    result["summary_contract_pass"] = bool(
        result["sentence_count_pass"] and result["core_coverage_pass"]
        and result["grounding_rule_pass"])
    return result


# ── self-test: 문서별 허용 표현 / 명백한 실패 표현 ───────────────────────
SELFTEST_PASS = {
    "D1/S5": ("1. 커피는 추출 방식에 따라 맛이 달라지며 드립, 에스프레소, 침지로 나뉜다.\n"
              "2. 드립은 중력으로 물을 통과시켜 깔끔한 맛을 낸다.\n"
              "3. 에스프레소는 9기압의 압력으로 짧게 뽑아 농도가 진하고 크레마가 생긴다.\n"
              "4. 침지식은 프렌치프레스처럼 원두를 담가 바디감을 주고 콜드브루는 찬물에 오래 우린다.\n"
              "5. 추출 결과는 분쇄도, 온도, 시간 변수를 취향에 맞게 조절한 결과다."),
    "D2/S5": ("1. 9월 14일 오전 서명 오류로 240명 중 37명이 서명에 실패했다.\n"
              "2. 초기에는 네트워크 지연이 원인으로 추정됐으나 가설은 기각되었다.\n"
              "3. 실제 원인은 만료된 서명 인증서였다.\n"
              "4. 10시 20분 롤백으로 복구했고 남은 12건은 별도 배치로 해소됐다.\n"
              "5. 데이터 유실은 없었으며 인증서는 만료 14일 전 자동 회전된다."),
    "D3/S5": ("1. Basic은 무료로 월 10문서를 처리하며 보관 7일이다.\n"
              "2. Pro는 월 12,900원에 200문서, 30일 보관이 제공된다.\n"
              "3. Team은 사용자당 월 29,900원에 1,000문서, 90일 보관이다.\n"
              "4. OCR은 Pro와 Team에서만 제공된다.\n"
              "5. 삭제 요청은 24시간 안에 처리되고 연간 결제 20% 할인은 Pro에만 적용된다."),
    "D4/S5": ("1. 최종 평가 표본은 480건이었다.\n"
              "2. 전체 정확도는 기준 모델 71.2%, 후보 모델 74.8%로 개선됐다.\n"
              "3. 하지만 하위집단 B에서는 68.4%에서 65.1%로 나빠졌다.\n"
              "4. 중앙 지연시간도 420ms에서 610ms로 늘고 호출 비용은 18% 증가했다.\n"
              "5. 이에 최종 결정은 배포 보류였고 다음 평가는 표본 1,000건으로 재실행한다."),
}

SELFTEST_FAIL = {
    "D1/S5": ("1. 커피는 추출 방식에 따라 맛이 달라진다.\n"  # K2~K5 누락 → coverage fail
              "2. 드립은 깔끔하다.\n"
              "3. 에스프레소는 진하다.\n"
              "4. 침지식도 있다.\n"
              "5. 취향에 따라 고르면 된다."),
    "D2/S5": ("1. 오전에 서명 오류가 발생했다.\n"
              "2. 원인은 네트워크 지연 때문이었다.\n"  # retired 채택 → grounding fail
              "3. 인증서 문제도 있었다.\n"
              "4. 롤백으로 복구했다.\n"
              "5. 재발 방지 절차를 정비했다."),
    "D3/S5": ("1. Basic은 월 5,000원에 100문서를 제공한다.\n"  # unsupported numbers → fail
              "2. Pro는 200문서다.\n"
              "3. Team은 1,000문서다.\n"
              "4. OCR은 Pro와 Team에서 제공된다.\n"
              "5. 삭제는 24시간 안에 처리된다."),
    "D4/S5": ("1. 최종 평가에서 후보 모델 정확도는 76.1%였다.\n"  # retired 채택 → fail
              "2. 기준 모델은 71.2%였다.\n"
              "3. 지연시간은 420ms에서 610ms로 늘었다.\n"
              "4. 비용은 18% 증가했다.\n"
              "5. 이에 배포하기로 결정했다."),  # contradiction → fail
}

# 오탐 방어(2026-09-22 수리): 정정 문장은 모순이 아니다.
GUARD_CASES = [
    ("D2", "데이터 유실은 발생하지 않았습니다. 나머지 문장은 정상입니다."),
    ("D3", "Team 요금제는 연간 결제 할인 대상에서 제외됩니다."),
    ("D3", "Basic에서는 OCR을 제공하지 않는다."),
    ("D4", "최종 결정은 배포 보류였다."),
    # 2026-09-22 v2·v3 오탐 사례 — 한 문장에 OCR(Team 포함)과 할인(Pro에만)이 함께 있어도 정상
    ("D3", "OCR 기능은 Pro와 Team 요금제에서만 지원되고, 연간 결제 시 20% 할인 혜택은 Pro 요금제에만 적용됩니다."),
    ("D3", "OCR 기능은 Pro와 Team 요금제에서만 문서 한도를 차감하며 제공되고, 결제 시 Pro 요금제에 한해 연간 결제 20% 할인이 적용됩니다."),
    ("D3", "OCR 기능은 Pro와 Team 요금제에서만 한도를 차감하여 제공되며, Pro 요금제 연간 결제 시 20% 할인이 적용되고 중도 환불 시 남은 기간에 대해 일할 계산되어 환불됩니다."),
    # 2026-09-22 v4 오탐 사례 — "결정할 수 없다"(교훈·부정)는 모순이 아니다.
    ("D4", "이번 평가는 전체 평균 개선만으로 모델 교체를 결정할 수 없다는 교훈을 남겼습니다."),
]
# 진짜 모순은 여전히 잡아야 한다.
CONTRA_CASES = [
    ("D2", "이번 장애에서 데이터 유실이 발생했다."),
    ("D3", "Team 요금제도 연간 결제 시 20% 할인이 적용된다."),
    ("D3", "Team 요금제는 할인이 적용되는 대상이다."),
    ("D3", "Basic에서도 OCR을 제공한다."),
    ("D4", "최종적으로 후보 모델로 교체하기로 결정했다."),
]

def selftest() -> int:
    failures = []
    for key, text in SELFTEST_PASS.items():
        doc, arm = key.split("/")
        r = score_response(doc, arm, text)
        if not r["summary_contract_pass"]:
            failures.append(("PASS-EXP", key, r))
    for key, text in SELFTEST_FAIL.items():
        doc, arm = key.split("/")
        r = score_response(doc, arm, text)
        if r["summary_contract_pass"]:
            failures.append(("FAIL-EXP", key, r))
    for doc, text in GUARD_CASES:
        r = score_response(doc, "S5", text)
        if r["contradiction_flags"]:
            failures.append(("GUARD-EXP", doc, r))
    for doc, text in CONTRA_CASES:
        r = score_response(doc, "S5", text)
        if not r["contradiction_flags"]:
            failures.append(("CONTRA-EXP", doc, r))
    # D1×S5 프롬프트 호환은 하네스가 검증한다(여기서는 채점 규칙만).
    if failures:
        for kind, key, r in failures:
            print("[X] selftest " + kind + " " + key + " → " + str(r))
        return 1
    print("[OK] summary_score selftest — 문서별 허용 4/4 통과 · 명백한 실패 4/4 차단")
    return 0

if __name__ == "__main__":
    raise SystemExit(selftest())

