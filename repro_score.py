#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repro_score.py — REP(재현성) 실측 결정론 채점기. TESTSET.md §3-C의 지표 SSOT.

★손으로 세지 않는다. 본문에 들어갈 모든 수치는 이 스크립트 출력에서만 온다(prm_score.py 선례).

세는 것 / 안 세는 것:
  - 센다 = **출력의 동일성**. 정규화한 문자열이 회차 간 같은가, 구조화 값(라벨 벡터·필드)이 같은가.
  - 안 센다 = **정답률**. 매번 똑같이 틀리는 모델도 재현성은 100%다. 단 REP-02의 `총금액`만은
    수량×단가로 **검산 가능**하므로 재현성과 따로 계산해 둘을 분리해 보여준다(혼동 방지).

정규화(normalize) 규칙 — 이것 자체가 결과를 만드므로 명시:
  ① 코드펜스(```json … ```) 제거 ② 앞뒤 공백 제거 ③ 줄 끝 공백 제거 ④ 빈 줄 축약.
  대소문자·문장부호는 **건드리지 않는다**(모델이 실제로 다르게 쓴 것을 같다고 뭉개지 않기 위해).

사용:
  /usr/bin/python3 repro_score.py --selftest                 # 자체검증(단언)
  /usr/bin/python3 repro_score.py test_runs/<run_dir> [...]  # 채점 → repro_results.json + 표 출력
"""
import argparse
import itertools
import json
import os
import re
import statistics
import sys

try:
    import yaml
except ImportError:                                    # 정직 차단(조용한 폴백 금지)
    print("PyYAML 미설치 — run.yaml 을 읽을 수 없다(pip install pyyaml).", file=sys.stderr)
    raise

# REP-02 정답(입력 이메일에서 사람이 읽어 고정 — 검산용 SSOT. testset.py `_REP02_MAIL` 과 동기).
REP02_TRUTH = [
    {"품목": "SB-204", "수량": 1200, "단가": 7200,  "납기일": "2026-08-12", "총금액": 1200 * 7200},
    {"품목": "AH-31",  "수량": 450,  "단가": 15400, "납기일": "2026-08-26", "총금액": 450 * 15400},
]
REP01_ITEMS = 10
REP02_FIELDS = ("업체명", "담당자", "품목", "수량", "단가", "납기일", "총금액")

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.S)


def normalize(text):
    """비교용 정규화. 규칙은 모듈 docstring 참조(대소문자·문장부호는 보존)."""
    t = (text or "").strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1)
    lines = [ln.rstrip() for ln in t.strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    out, blank = [], False
    for ln in lines:
        if not ln:
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(ln)
    return "\n".join(out)


def parse_json_block(text):
    """모델 출력에서 JSON 값을 뽑는다. 실패 = None(추측 보정 금지 — 파싱 실패도 데이터다)."""
    t = normalize(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None


def pair_rate(values):
    """서로 같은 쌍의 비율(0~1). n<2면 None(쌍이 없으면 '100%'라고 우기지 않는다)."""
    if len(values) < 2:
        return None
    pairs = list(itertools.combinations(values, 2))
    same = sum(1 for a, b in pairs if a == b)
    return round(same / len(pairs), 4)


def rep01_labels(text):
    """REP-01 출력 → 번호순 라벨 튜플. 파싱 실패/형식 이탈 = None.

    ★번호 중복은 실패로 본다(T2 적대검증 2026-07-29). dict 에 그냥 넣으면 뒤엣것이 앞엣것을
    덮어써서 '11개 중 하나를 잃은 출력'이 정상 10개로 통과했다."""
    v = parse_json_block(text)
    if not isinstance(v, list):
        return None
    got = {}
    for e in v:
        if not isinstance(e, dict):
            return None
        n, lab = e.get("번호"), e.get("분류")
        if isinstance(n, bool):                  # True는 int 1로 통과해 버린다
            return None
        try:
            n = int(n)
        except (TypeError, ValueError):
            return None
        if not isinstance(lab, str) or n in got:
            return None
        got[n] = lab.strip()
    if sorted(got) != list(range(1, REP01_ITEMS + 1)):
        return None
    return tuple(got[i] for i in range(1, REP01_ITEMS + 1))


def _num(x):
    """'8,640,000원' / '8640000' → 8640000. 숫자로 못 읽으면 원값 그대로(문자열 비교로 넘어감).

    ★비숫자 문자를 통째로 지우면 'SB-204' 가 -204 가 된다(자체검증이 잡은 실버그 2026-07-29).
    쉼표·공백·말미 단위만 떼고, **남은 게 순수 숫자일 때만** 숫자로 본다."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = re.sub(r"(원|개|EA|ea)$", "", x.strip().replace(",", "").replace(" ", ""))
        if re.fullmatch(r"-?\d+(\.0+)?", s):
            return int(float(s))
    return x


def rep02_records(text):
    """REP-02 출력 → 필드 정규화한 레코드 리스트. 파싱 실패 = None."""
    v = parse_json_block(text)
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, list) or not v:
        return None
    out = []
    for e in v:
        if not isinstance(e, dict):
            return None
        rec = {}
        for k in REP02_FIELDS:
            val = e.get(k)
            rec[k] = _num(val) if k in ("수량", "단가", "총금액") else (
                val.strip() if isinstance(val, str) else val)
        if all(x is None for x in rec.values()):
            return None                          # 빈 dict {} 가 '전 필드 None 레코드'로 통과하던 구멍
        out.append(rec)
    return out or None


def rep03_titles(text):
    """REP-03 출력 → 번호 제거한 제목 리스트."""
    titles = []
    for ln in normalize(text).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        titles.append(re.sub(r"^\s*\d+\s*[.)·\-]\s*", "", ln).strip())
    return titles


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────
# 셀(모델×태스크×조건) 채점
# ─────────────────────────────────────────────────────────────
def score_cell(task, outputs, runs):
    """outputs = 원문 리스트(회차순) · runs = 대응 run 레코드. 반환 = 지표 dict."""
    n = len(outputs)
    norms = [normalize(o) for o in outputs]
    lens = [len(x) for x in norms]
    cell = {
        "n": n,
        "exact_pair_rate": pair_rate(norms),
        # ★n<2 는 '전부 같음'이 아니라 '판정 불가'다. n=1에 True를 주면 리허설 1회차가 100% 재현으로
        #   읽힌다(T2 적대검증 2026-07-29 재현 확인).
        "all_identical": (len(set(norms)) == 1) if n > 1 else None,
        "distinct_outputs": len(set(norms)),
        "len_min": min(lens) if lens else None,
        "len_max": max(lens) if lens else None,
        "len_mean": round(statistics.mean(lens), 1) if lens else None,
        "len_stdev": round(statistics.stdev(lens), 1) if n > 1 else None,
    }

    el = [r.get("elapsed_s") for r in runs if isinstance(r.get("elapsed_s"), (int, float))]
    ev = [r.get("eval_duration_ns") for r in runs if isinstance(r.get("eval_duration_ns"), (int, float))]
    ld = [r.get("load_duration_ns") for r in runs if isinstance(r.get("load_duration_ns"), (int, float))]
    tk = [r.get("eval_count") for r in runs if isinstance(r.get("eval_count"), (int, float))]
    # ★총소요(로딩 포함)와 순수 생성을 끝까지 분리해 들고 간다 — 표에서 섞으면 안 된다.
    cell["elapsed_s_mean"] = round(statistics.mean(el), 2) if el else None
    cell["gen_s_mean"] = round(statistics.mean(ev) / 1e9, 2) if ev else None
    cell["load_s_mean"] = round(statistics.mean(ld) / 1e9, 2) if ld else None
    cell["eval_tokens_mean"] = round(statistics.mean(tk), 1) if tk else None

    base = task.split("/")[0]
    if base == "REP-01":
        vecs = [rep01_labels(o) for o in outputs]
        ok = [v for v in vecs if v is not None]
        cell["parse_fail"] = n - len(ok)
        cell["vector_exact_pair_rate"] = pair_rate(ok)
        cell["vector_all_identical"] = (len(set(ok)) == 1) if len(ok) > 1 else None
        cell["distinct_vectors"] = len(set(ok))
        per_item = []
        for i in range(REP01_ITEMS):
            labs = [v[i] for v in ok]
            dist = {}
            for lb in labs:
                dist[lb] = dist.get(lb, 0) + 1
            per_item.append({"번호": i + 1, "distinct": len(dist),
                             "dist": dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))})
        cell["per_item"] = per_item
        # ★파싱된 벡터가 2개 미만이면 '흔들린 문항 없음'이 아니라 판정 불가. []를 주면 전건 파싱실패가
        #   완전 안정으로 읽힌다(T2 적대검증 2026-07-29 재현 확인).
        cell["unstable_items"] = ([p["번호"] for p in per_item if p["distinct"] > 1]
                                  if len(ok) > 1 else None)

    elif base == "REP-02":
        recs = [rep02_records(o) for o in outputs]
        ok = [r for r in recs if r is not None]
        cell["parse_fail"] = n - len(ok)
        cell["row_count_dist"] = {}
        for r in ok:
            k = str(len(r))
            cell["row_count_dist"][k] = cell["row_count_dist"].get(k, 0) + 1
        two = [r for r in ok if len(r) == len(REP02_TRUTH)]
        # ★행 수가 안 맞는 출력을 조용히 빼면 분모가 줄어 실패가 숨는다 — 뺀 개수를 남긴다.
        cell["row_count_mismatch"] = len(ok) - len(two)
        field_rates, field_modal = {}, {}
        for fi in range(len(REP02_TRUTH)):
            for k in REP02_FIELDS:
                vals = [json.dumps(r[fi].get(k), ensure_ascii=False) for r in two]
                key = f"행{fi + 1}.{k}"
                field_rates[key] = pair_rate(vals)
                if vals:
                    # ★set() 순회 순서는 프로세스마다 달라 동률일 때 최빈값이 바뀐다 — 결정론 채점기가
                    #   비결정론이 되던 실버그(T2 적대검증 2026-07-29 재현: 4회 중 A,B,A,B).
                    modal = max(sorted(set(vals)), key=vals.count)
                    field_modal[key] = {"modal": json.loads(modal), "freq": vals.count(modal),
                                        "of": len(vals)}
        cell["field_pair_rates"] = field_rates
        cell["field_modal"] = field_modal
        comparable = [v for v in field_rates.values() if v is not None]
        cell["unstable_fields"] = ([k for k, v in field_rates.items() if v is not None and v < 1.0]
                                   if comparable else None)   # 비교 쌍 0 = 안정이 아니라 판정 불가
        # 재현성과 별개 축 = 검산(수량×단가). 맞고 틀림은 여기서만 말한다.
        # ★행 순서를 가정하지 않는다 — 품목 코드로 짝짓는다(모델이 순서를 바꿔 내도 정답 판정이 안 흔들리게).
        correct = 0
        for r in two:
            hit = True
            for truth in REP02_TRUTH:
                row = next((x for x in r if truth["품목"] in str(x.get("품목") or "")), None)
                if row is None or row.get("총금액") != truth["총금액"]:
                    hit = False
                    break
            correct += 1 if hit else 0
        cell["total_amount_correct"] = correct
        cell["total_amount_of"] = len(two)

    elif base == "REP-03":
        titlesets = [rep03_titles(o) for o in outputs]
        cell["title_count_dist"] = {}
        for t in titlesets:
            k = str(len(t))
            cell["title_count_dist"][k] = cell["title_count_dist"].get(k, 0) + 1
        pairs = list(itertools.combinations(titlesets, 2))
        cell["title_jaccard_mean"] = (round(statistics.mean(jaccard(a, b) for a, b in pairs), 4)
                                      if pairs else None)
        allt = [t for ts in titlesets for t in ts]
        cell["distinct_titles"] = len(set(allt))
        cell["total_titles"] = len(allt)
    return cell


def score_run_dir(run_dir):
    with open(os.path.join(run_dir, "run.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cells = {}
    for r in data.get("runs", []):
        task = r.get("task", "")
        name = r.get("output_file", "")
        p = os.path.join(run_dir, name)
        # run.yaml 은 기계가 쓰지만, 절대경로/상위탈출이 들어오면 run_dir 밖을 읽는다 — 파일명만 받는다.
        if not name or os.path.isabs(name) or os.path.dirname(name) or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:      # ★이중 open + 핸들 누수였다(T2 2026-07-29)
            body = f.read()
        cells.setdefault(task, {"outputs": [], "runs": []})
        cells[task]["outputs"].append(body)
        cells[task]["runs"].append(r)
    scored = {t: score_cell(t, c["outputs"], c["runs"]) for t, c in sorted(cells.items())}
    return {"run_dir": run_dir, "tool": data.get("tool"), "model": data.get("model"),
            "date": data.get("date"), "generated_by": data.get("generated_by"), "cells": scored}


def print_report(res):
    print(f"\n══ {res['model']} ({res['run_dir']}) · date={res['date']} · origin={res['generated_by']}")
    print(f"{'셀':<14}{'n':>3}{'출력완전일치':>13}{'서로다른출력':>13}"
          f"{'길이평균':>10}{'총소요s':>9}{'생성s':>8}{'로딩s':>8}")
    for task, c in res["cells"].items():
        pr = "—" if c["exact_pair_rate"] is None else f"{c['exact_pair_rate'] * 100:.1f}%"
        print(f"{task:<14}{c['n']:>3}{pr:>13}{c['distinct_outputs']:>13}"
              f"{str(c['len_mean']):>10}{str(c['elapsed_s_mean']):>9}"
              f"{str(c['gen_s_mean']):>8}{str(c['load_s_mean']):>8}")
    for task, c in res["cells"].items():
        if "vector_exact_pair_rate" in c:
            vr = "—" if c["vector_exact_pair_rate"] is None else f"{c['vector_exact_pair_rate'] * 100:.1f}%"
            print(f"  {task} 라벨벡터 일치 {vr} · 서로다른벡터 {c['distinct_vectors']} · "
                  f"흔들린 문항 {c['unstable_items'] or '없음'} · 파싱실패 {c['parse_fail']}")
        if "field_pair_rates" in c:
            print(f"  {task} 흔들린 필드 {c['unstable_fields'] or '없음'} · 파싱실패 {c['parse_fail']} · "
                  f"총금액 검산 정답 {c['total_amount_correct']}/{c['total_amount_of']}")
        if "title_jaccard_mean" in c:
            print(f"  {task} 제목집합 자카드 평균 {c['title_jaccard_mean']} · "
                  f"서로다른 제목 {c['distinct_titles']}/{c['total_titles']}")


# ─────────────────────────────────────────────────────────────
# 자체검증 — 채점기가 틀리면 본문 숫자가 전부 틀린다.
# ─────────────────────────────────────────────────────────────
def selftest():
    n = 0

    def eq(a, b, msg):
        nonlocal n
        assert a == b, f"{msg}: {a!r} != {b!r}"
        n += 1

    eq(normalize("```json\n[1]\n```"), "[1]", "코드펜스 제거")
    # 줄 끝 공백·앞뒤 빈 줄은 지우지만 **들여쓰기는 보존**한다. 같은 JSON을 들여쓰기만 다르게 뱉은 것도
    # '문자열로는 다른 출력'이 맞다 — 의미 동일성은 벡터/필드 지표가 따로 잰다(축 분리).
    eq(normalize("  a  \n\n\n b \n"), "a\n\n b", "빈줄 축약·줄끝 공백 제거·들여쓰기 보존")
    eq(normalize("A b"), "A b", "대소문자·내부공백 보존")
    eq(normalize(None), "", "None 안전")

    eq(pair_rate(["x", "x", "x"]), 1.0, "전부 같으면 1.0")
    eq(pair_rate(["x", "y"]), 0.0, "둘 다르면 0.0")
    eq(pair_rate(["x", "x", "y"]), round(1 / 3, 4), "3개 중 한 쌍만 일치")
    eq(pair_rate(["x"]), None, "n=1이면 None(쌍 없음)")
    eq(pair_rate([]), None, "n=0이면 None")

    eq(parse_json_block('```json\n{"a":1}\n```'), {"a": 1}, "펜스 JSON")
    eq(parse_json_block('앞말 [1,2] 뒷말'), [1, 2], "괄호 추출")
    eq(parse_json_block("깨진 {"), None, "파싱실패=None")

    good = json.dumps([{"번호": i, "분류": "환불"} for i in range(1, 11)], ensure_ascii=False)
    eq(rep01_labels(good), tuple(["환불"] * 10), "라벨벡터 정상")
    eq(rep01_labels(json.dumps([{"번호": 1, "분류": "환불"}], ensure_ascii=False)), None, "10개 미만=None")
    eq(rep01_labels("[]"), None, "빈배열=None")
    shuffled = json.dumps([{"번호": i, "분류": "배송" if i == 3 else "환불"}
                           for i in [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]], ensure_ascii=False)
    eq(rep01_labels(shuffled)[2], "배송", "번호 뒤섞여도 번호순 정렬")

    eq(_num("8,640,000원"), 8640000, "숫자 정규화")
    eq(_num(1200), 1200, "정수 통과")
    eq(_num("SB-204"), "SB-204", "숫자 아니면 원값")

    r = rep02_records(json.dumps([{k: ("x" if k not in ("수량", "단가", "총금액") else "1,000")
                                   for k in REP02_FIELDS}], ensure_ascii=False))
    eq(r[0]["수량"], 1000, "필드 숫자화")
    eq(rep02_records("아무말"), None, "REP-02 파싱실패=None")

    eq(rep03_titles("1. 가\n2) 나\n3 - 다"), ["가", "나", "다"], "번호 제거")
    eq(jaccard(["a", "b"], ["a", "b"]), 1.0, "자카드 동일")
    eq(jaccard(["a"], ["b"]), 0.0, "자카드 무교집합")

    # 셀 채점 — 완전 동일 3회
    same = [good] * 3
    runs = [{"elapsed_s": 1.0, "eval_duration_ns": 1e9, "load_duration_ns": 5e8, "eval_count": 10}] * 3
    c = score_cell("REP-01/C", same, runs)
    eq(c["exact_pair_rate"], 1.0, "동일 3회 완전일치")
    eq(c["vector_all_identical"], True, "벡터 동일")
    eq(c["unstable_items"], [], "흔들린 문항 없음")
    eq(c["gen_s_mean"], 1.0, "생성시간 ns→s")
    eq(c["load_s_mean"], 0.5, "로딩시간 ns→s")

    # 셀 채점 — 3번 문항만 흔들림
    v2 = json.dumps([{"번호": i, "분류": ("배송" if i == 3 else "환불")} for i in range(1, 11)],
                    ensure_ascii=False)
    c2 = score_cell("REP-01/A", [good, good, v2], runs)
    eq(c2["unstable_items"], [3], "3번만 흔들림 검출")
    eq(c2["vector_exact_pair_rate"], round(1 / 3, 4), "벡터 쌍일치 1/3")
    eq(c2["distinct_vectors"], 2, "서로 다른 벡터 2")

    # REP-02 — 총금액 오답이어도 재현성은 100%일 수 있다(축 분리 확인)
    wrong = json.dumps([{"업체명": "대성정밀", "담당자": "김현우", "품목": "스테인리스 브래킷 SB-204",
                         "수량": 1200, "단가": 7200, "납기일": "2026-08-12", "총금액": 864000},
                        {"업체명": "대성정밀", "담당자": "김현우", "품목": "알루미늄 하우징 AH-31",
                         "수량": 450, "단가": 15400, "납기일": "2026-08-26", "총금액": 681000}],
                       ensure_ascii=False)
    c3 = score_cell("REP-02/B", [wrong] * 3, runs)
    eq(c3["unstable_fields"], [], "필드 전부 안정")
    eq(c3["total_amount_correct"], 0, "총금액은 전부 오답")
    eq(c3["total_amount_of"], 3, "검산 대상 3회")

    c4 = score_cell("REP-03/A", ["1. 가\n2. 나", "1. 가\n2. 다"], runs[:2])
    eq(c4["title_jaccard_mean"], round(1 / 3, 4), "제목집합 자카드")
    eq(c4["distinct_titles"], 3, "서로 다른 제목 3")

    # ── T2 적대검증(2026-07-29)에서 실재로 확인된 8건 회귀 방어 ──
    # ① n<2 는 '전부 같음'이 아니라 판정 불가
    c5 = score_cell("REP-01/A", [good], runs[:1])
    eq(c5["exact_pair_rate"], None, "n=1 쌍일치 None")
    eq(c5["all_identical"], None, "n=1 all_identical None(True 아님)")
    eq(c5["vector_all_identical"], None, "n=1 벡터 all_identical None")
    eq(c5["unstable_items"], None, "n=1 흔들린문항 None([] 아님)")

    # ② 전건 파싱실패를 '안정'으로 위장하지 않는다
    c6 = score_cell("REP-01/A", ["깨진 출력"] * 3, runs)
    eq(c6["parse_fail"], 3, "파싱실패 3건 계수")
    eq(c6["unstable_items"], None, "파싱 전멸 시 흔들린문항 None")

    # ③ REP-01 번호 중복 = 실패(덮어쓰기로 조용히 통과하지 않는다)
    dup = json.dumps([{"번호": 1, "분류": "환불"}]
                     + [{"번호": i, "분류": "배송"} for i in range(1, 11)], ensure_ascii=False)
    eq(rep01_labels(dup), None, "번호 중복=None")
    eq(rep01_labels(json.dumps([{"번호": True, "분류": "환불"}], ensure_ascii=False)), None, "bool 번호=None")

    # ④ REP-02 빈 dict 는 레코드가 아니다
    eq(rep02_records("{}"), None, "빈 dict=None")

    # ⑤ REP-02 행 수 불일치 = 뺀 개수를 남기고 '안정'으로 위장하지 않는다
    one = json.dumps([{k: ("x" if k not in ("수량", "단가", "총금액") else 1)
                       for k in REP02_FIELDS}], ensure_ascii=False)
    c7 = score_cell("REP-02/A", [one] * 3, runs)
    eq(c7["row_count_mismatch"], 3, "행수 불일치 3건 기록")
    eq(c7["unstable_fields"], None, "비교 대상 0 → unstable_fields None([] 아님)")

    # ⑥ field_modal 동률 = 결정론(프로세스 해시 순서에 안 흔들림)
    ra = json.dumps([{k: ("A" if k == "업체명" else 1) for k in REP02_FIELDS}] * 2, ensure_ascii=False)
    rb = json.dumps([{k: ("B" if k == "업체명" else 1) for k in REP02_FIELDS}] * 2, ensure_ascii=False)
    eq(score_cell("REP-02/A", [ra, rb], runs[:2])["field_modal"]["행1.업체명"]["modal"], "A",
       "동률 최빈값 = 사전순 첫값(결정론)")

    # ⑦ 총금액 검산은 행 순서를 가정하지 않는다(뒤집어 내도 같은 판정)
    def _mk(order):
        rows = [{"업체명": "대성정밀", "담당자": "김현우", "품목": f"품 {t['품목']}",
                 "수량": t["수량"], "단가": t["단가"], "납기일": t["납기일"],
                 "총금액": t["총금액"]} for t in REP02_TRUTH]
        return json.dumps(rows if order else rows[::-1], ensure_ascii=False)
    eq(score_cell("REP-02/A", [_mk(True), _mk(False)], runs[:2])["total_amount_correct"], 2,
       "행 순서 뒤집혀도 검산 정답 2")

    # ⑧ run_dir 밖을 가리키는 output_file 은 읽지 않는다(경로 탈출 차단)
    eq(os.path.isabs("/etc/passwd"), True, "경로 가드 전제(절대경로 판별)")

    print(f"[selftest] 단언 {n}건 전부 통과")
    return 0


def _font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


TASK_LABEL = {"REP-01": "문의 분류(10건)", "REP-02": "발주 메일 추출", "REP-03": "제목 5개(자유생성)"}
COND_LABEL = {"A": "모델 기본값", "B": "temperature 0", "C": "temperature 0 + seed 고정"}
COND_COLOR = {"A": "#c8ccd4", "B": "#7aa2f7", "C": "#2f6fed"}


def make_chart_rates(results, path):
    """조건별 '출력 완전일치율' 묶음 막대. 셀이 비면 그리지 않는다(빈 칸을 0%로 위장하지 않음)."""
    plt = _font()
    groups = []                                   # [(그룹라벨, {조건: 비율%})]
    for res in results:
        bases = []
        for task in res["cells"]:
            b = task.split("/")[0]
            if b not in bases:
                bases.append(b)
        for b in bases:
            vals = {}
            for cond in ("A", "B", "C"):
                c = res["cells"].get(f"{b}/{cond}")
                if c and c["exact_pair_rate"] is not None:
                    vals[cond] = c["exact_pair_rate"] * 100
            if vals:
                groups.append((f"{TASK_LABEL.get(b, b)}\n{res['model']}", vals))
    if not groups:
        return False

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=160)
    w, x = 0.26, range(len(groups))
    for k, cond in enumerate(("A", "B", "C")):
        # ★값 없는 조건은 막대를 안 그린다. 0으로 채우면 '측정 못 함'이 '0%'로 보인다
        #   (docstring이 그러지 않겠다고 해놓고 .get(cond, 0) 이던 실버그 — T2 2026-07-29).
        idxs = [i for i, g in enumerate(groups) if cond in g[1]]
        if not idxs:
            continue
        xs = [i + (k - 1) * w for i in idxs]
        ys = [groups[i][1][cond] for i in idxs]
        bars = ax.bar(xs, ys, w, label=COND_LABEL[cond], color=COND_COLOR[cond])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("같은 답이 나온 비율 (%)")
    ax.set_title("같은 질문 8번 — 조건을 바꾸면 답이 얼마나 같아지나 (2026년 7월 29일 측정)",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"차트 저장: {path}")
    return True


def make_chart_items(results, path, model_hint="gemma3:4b"):
    """모델 기본값(조건 A)에서 '겉모습(문자열)'과 '결론(내용)'이 갈리는 정도.

    ★왜 이 그림인가: REP-01/A는 문자열로는 57%만 같았는데 분류 라벨은 8회 전부 같았다.
      "답이 매번 다르다"는 체감이 실제로는 서식 차이인 경우가 있다는 걸 한 장으로 보여준다.
      (문항별 흔들림 차트는 실측에서 흔들린 문항이 0이라 정보가 없어 폐기 — 2026-07-29.)"""
    plt = _font()
    res = next((r for r in results if model_hint in str(r.get("model"))), None)
    if not res:
        return False
    rows = []                                     # (라벨, 문자열%, 내용%, 내용지표이름)
    for base in ("REP-01", "REP-02", "REP-03"):
        c = res["cells"].get(f"{base}/A")
        if not c or c["exact_pair_rate"] is None:
            continue
        if base == "REP-01":
            sub, name = c.get("vector_exact_pair_rate"), "분류 라벨 10개"
        elif base == "REP-02":
            vals = [v for v in (c.get("field_pair_rates") or {}).values() if v is not None]
            sub, name = (sum(vals) / len(vals) if vals else None), "추출 필드 14개 평균"
        else:
            sub, name = c.get("title_jaccard_mean"), "제목 겹침(자카드)"
        if sub is None:
            continue
        rows.append((f"{TASK_LABEL[base]}\n({name})", c["exact_pair_rate"] * 100, sub * 100))
    if not rows:
        return False

    fig, ax = plt.subplots(figsize=(8.8, 4.3), dpi=160)
    x, w = range(len(rows)), 0.34
    b1 = ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w,
                label="글자 그대로 같았나", color="#c8ccd4")
    b2 = ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w,
                label="결론(내용)이 같았나", color="#2f6fed")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("같은 결과가 나온 비율 (%)")
    ax.set_title(f"모델 기본값에서 — 겉모습이 달라도 결론은 같았나 ({res['model']} · 8회)",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"차트 저장: {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="test_runs/repro_results.json")
    ap.add_argument("--chart", help="조건별 완전일치율 차트 PNG 경로")
    ap.add_argument("--chart-items", help="REP-01 문항별 흔들림 차트 PNG 경로")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.run_dirs:
        ap.error("run_dir 를 지정하라(또는 --selftest)")
    selftest()                                   # ★채점 전 항상 자체검증(깨진 채점기로 본문 숫자 만들지 않는다)
    results = [score_run_dir(d) for d in a.run_dirs]
    for r in results:
        print_report(r)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n[repro_score] → {a.out}")
    if a.chart:
        make_chart_rates(results, a.chart)
    if a.chart_items:
        make_chart_items(results, a.chart_items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
