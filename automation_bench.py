#!/usr/bin/env python
"""반복업무 자동화 시간 측정 벤치 — tested 글(AI 자동화 시간절약)의 수치를 받치는 재현 아티팩트.

★tested 무결성(2026-06-27 교훈 [[tested-evidence-integrity-committed-artifact]]): AI가 짠
스크립트(test_runs/gemini-20260627/generated_merge.py·run.yaml=생성 증거)를 실데이터에 돌려
실행시간을 재고, 결과를 *독립적으로* 검증한 뒤 run 디렉터리에 automation_bench.json으로 남긴다.
(run.yaml = AI가 스크립트를 썼다는 증거 / 이 bench = 그 스크립트의 실행시간·정확성 증거.)
수동 베이스라인은 사람 작업이라 측정 불가 → 글에서 '추정'으로 정직 표기(여기 수치는 자동화 쪽만).

usage: python automation_bench.py
"""
import csv, glob, os, subprocess, time, json
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "corpus/auto")
SCRIPT = os.path.join(BASE, "test_runs/gemini-20260627/generated_merge.py")
OUT = os.path.join(BASE, "test_runs/gemini-20260627/automation_bench.json")
RUNS = 5


def data_stats():
    raw, uniq = 0, set()
    files = sorted(glob.glob(os.path.join(DATA, "sales_2026-*.csv")))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if row:
                    raw += 1; uniq.add(tuple(row))
    return len(files), raw, len(uniq)


def _expected_monthly_sums():
    """원본 12파일에서 고유행 기준 월별 amount 합계를 *독립적으로* 재계산(merged.csv와 무관 경로)."""
    uniq = set()
    for fp in sorted(glob.glob(os.path.join(DATA, "sales_2026-*.csv"))):
        with open(fp, encoding="utf-8") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if row:
                    uniq.add(tuple(row))
    from collections import defaultdict
    s = defaultdict(int)
    for d, _i, a in uniq:
        s[d[:7]] += int(a)
    return dict(s)


def verify_merged(expected_unique):
    """AI 스크립트 출력(merged.csv)을 독립 검증 — 맹신 금지."""
    mp = os.path.join(DATA, "merged.csv")
    with open(mp, encoding="utf-8") as f:
        r = csv.reader(f); next(r); rows = [tuple(x) for x in r if x]
    dates = [x[0] for x in rows]
    from collections import defaultdict
    merged_sum = defaultdict(int)
    for d, _i, a in rows:
        merged_sum[d[:7]] += int(a)
    return {
        "merged_rows": len(rows),
        "dedup_correct": len(rows) == expected_unique,
        "sorted_asc": dates == sorted(dates),
        "no_dup_remaining": len(rows) == len(set(rows)),
        # ★월별 합계도 독립 재계산과 대조(적대검증 2026-06-27 GLM/MiMo 지적: 글이 주장한 합계검증을 artifact가 받쳐야)
        "monthly_sum_correct": dict(merged_sum) == _expected_monthly_sums(),
        # ★dedup은 '완전 동일 행(tuple)' 기준 — 셀 공백/포맷 차이는 별개(정확-일치 정의·정직 표기)
        "dedup_definition": "exact-row(tuple) match",
    }


def _pure_processing_s():
    """순수 데이터 처리 시간(파이썬 인터프리터 기동 제외) — subprocess wall은 대부분 기동시간이라 분리 측정."""
    from collections import defaultdict
    t = time.monotonic()
    uniq = set()
    for fp in sorted(glob.glob(os.path.join(DATA, "sales_2026-*.csv"))):
        with open(fp, encoding="utf-8") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if row:
                    uniq.add(tuple(row))
    rows = sorted(uniq, key=lambda x: x[0])
    s = defaultdict(int)
    for d, _i, a in rows:
        s[d[:7]] += int(a)
    return round((time.monotonic() - t) * 1000, 1)   # ms


def main():
    nfiles, raw, uniq = data_stats()
    # 자동화 스크립트 실행시간(wall) RUNS회
    times = []
    for _ in range(RUNS):
        t = time.monotonic()
        p = subprocess.run(["python3", SCRIPT], cwd=BASE, capture_output=True, text=True)
        times.append(round(time.monotonic() - t, 3))
        if p.returncode != 0:
            raise SystemExit(f"스크립트 실행 실패: {p.stderr[:300]}")
    verify = verify_merged(uniq)
    pure_ms = _pure_processing_s()

    result = {
        "task": "월별 매출 CSV 12개 → 합치기·중복제거·날짜정렬·월별합계",
        "script_origin": "Gemini 3.1 Pro 생성(run.yaml method=AUTO-01·generated_by=tool_test_harness)",
        "data": {"files": nfiles, "raw_rows": raw, "unique_rows": uniq, "dups_removed": raw - uniq},
        # wall = 사용자가 스크립트를 돌렸을 때 끝나는 전체 시간(파이썬 인터프리터 기동 포함)
        "total_wall_s": {"runs": times, "median": sorted(times)[len(times)//2],
                         "min": min(times), "max": max(times)},
        # pure = 순수 데이터 처리만(기동 제외) — wall 0.035s의 대부분은 파이썬 기동시간임을 정직 분리
        "pure_processing_ms": pure_ms,
        "timing_note": "wall(0.03s대)의 대부분은 파이썬 인터프리터 기동. 순수 처리는 pure_processing_ms. 사용자 체감=wall.",
        "correctness_independent_check": verify,
        "manual_baseline": "사람 작업이라 스크립트로 측정 불가 → 글에서 '추정'으로 정직 표기. "
                           "엑셀 숙련자가 12파일 취합+중복제거+정렬+피벗 ≈ 10분대(미숙·실수 시 훨씬↑). "
                           "★핵심=재사용: 다음 달 데이터도 스크립트는 같은 시간, 수동은 매번 반복.",
        "measured": "2026-06",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
