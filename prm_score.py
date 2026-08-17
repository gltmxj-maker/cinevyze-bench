# -*- coding: utf-8 -*-
"""
prm_score.py — PRM(프롬프트 품질 A/B) 결정론 채점기.

tool_test_harness가 남긴 run.yaml + NN-output.txt를 읽어 **숨은 루브릭** 충족 여부를 기계 판정한다.
사람이 점수를 매기지 않는다(§3 게이트 위조 금지와 같은 취지 — 숫자는 손으로 쓰지 않는다).

루브릭 SSOT = 이 파일의 RUBRICS. testset.py의 F 프롬프트 [출력형식] 항목과 **1:1 대응**해야 하며,
프롬프트를 고치면 여기도 같은 턴에 고친다(안 그러면 채점이 무의미).

  V(vague)  = 조건을 하나도 안 들은 출력  → 점수 = "말 안 해도 얻어걸리는 비율"
  F(framed) = 조건을 전부 들은 출력        → 점수 = "말해줘도 안 지키는 비율"의 반대

사용:
  python prm_score.py test_runs/ollama-gemma3-4b-20260726
  python prm_score.py <run_dir> --chart ./out/<post>/images/prompt-ab-score.png
"""
import os
import re
import sys
import json
import argparse
from collections import defaultdict

import yaml

# ── 존댓말 종결 판정.
#   해요체는 어미가 무엇이든 '요'로 끝난다(해요·할게요·하는데요·하잖아요…) → 어미 열거 대신 '요$'가 정답.
#   합쇼체는 '니다/니까/시오'. '죠'는 '지요'의 축약.
#   ★해라체('~한다','~하다')는 '다'로 끝나지만 '습니다'도 '다'로 끝나므로 **경어 판정을 먼저** 한다.
#   ★어느 쪽 어미도 아닌 조각(호격 "여러분", 제목, 목록 항목)은 반말의 증거가 아니므로 **분모에서 제외**.
#     (2026-07-26 firsthand: '알려드릴게요'를 놓치고 '안녕하세요, 여러분'을 반말로 세어 0% 오판정 → 수정)
_POLITE_END = re.compile(r"(요|죠|니다|니까|시오)$")
_PLAIN_END = re.compile(r"(다|네|야|어|아|지|군|구나|자|라|래|나|냐|니|까|음|슴)$")
_SENT_SPLIT = re.compile(r"[.!?\n]+")


def _nospace_len(s):
    return len(re.sub(r"\s+", "", s or ""))


def _paragraphs(s):
    return [p for p in re.split(r"\n\s*\n", (s or "").strip()) if p.strip()]


def _lines(s):
    return [ln.rstrip() for ln in (s or "").splitlines()]


def _nonempty_lines(s):
    return [ln for ln in _lines(s) if ln.strip()]


def _strip_fences(s):
    """모델이 ```로 감싸는 경우 펜스 라인만 제거(내용은 보존). 펜스는 '설명 문장'이 아니다."""
    return "\n".join(ln for ln in _lines(s) if not ln.strip().startswith("```"))


def _polite_ratio(s):
    """경어체 종결 비율 = 경어 / (경어 + 반말). 어느 쪽도 아닌 조각은 분모 제외.

    판정 가능한 문장이 하나도 없으면 0.0(확인 불가를 통과로 치지 않는다)."""
    sents = [x.strip().rstrip('"\'」』)”’') for x in _SENT_SPLIT.split(s or "")]
    sents = [x for x in sents if len(re.sub(r"[^가-힣A-Za-z0-9]", "", x)) >= 2]
    # 목록 기호·머리표는 문장 판정에서 제외(형식 요소지 문장이 아님)
    sents = [re.sub(r"^[-*•>#\d.)\s]+", "", x).strip() for x in sents]
    polite = plain = 0
    for x in sents:
        if not x:
            continue
        if _POLITE_END.search(x):
            polite += 1
        elif _PLAIN_END.search(x):
            plain += 1
    return polite / (polite + plain) if (polite + plain) else 0.0


def _table(s):
    """마크다운 표 파싱 → (헤더셀, 데이터행수, 표라인집합) / 표 없으면 (None, 0, set())."""
    lines = _lines(_strip_fences(s))
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\|?\s*:?-{2,}", ln.replace(" ", "")) and "|" in ln and i > 0:
            sep_i = i
            break
        if "|" in ln and re.fullmatch(r"[\s|:\-]+", ln) and ln.count("-") >= 3 and i > 0:
            sep_i = i
            break
    else:
        return None, 0, set()
    hdr = [c.strip() for c in lines[sep_i - 1].strip().strip("|").split("|")]
    rows, idxs = 0, {sep_i - 1, sep_i}
    for j in range(sep_i + 1, len(lines)):
        if "|" in lines[j] and lines[j].strip():
            rows += 1
            idxs.add(j)
        elif lines[j].strip():
            break
        else:
            break
    return hdr, rows, idxs


# ─────────────────────────────────────────────────────────────
# 루브릭 — testset.py의 F [출력형식]과 1:1
# 각 항목: (짧은이름, 판정함수(text) -> bool)
# ─────────────────────────────────────────────────────────────
def _r01_len(t):
    return 250 <= _nospace_len(t) <= 400


def _r01_para(t):
    return 1 <= len(_paragraphs(t)) <= 3


def _r01_polite(t):
    return _polite_ratio(t) >= 0.9


def _r01_word(t):
    return "자영업자" in (t or "")


def _r01_tail(t):
    ne = _nonempty_lines(_strip_fences(t))
    return bool(ne) and ne[-1].lstrip("*_# ").startswith("한 줄 요약:")


def _mail_body(t):
    """첫 줄이 '제목:'이면 그 줄을 뺀 나머지가 본문."""
    ne = _nonempty_lines(_strip_fences(t))
    if ne and ne[0].lstrip("*_# ").startswith("제목:"):
        raw = _strip_fences(t)
        idx = raw.find(ne[0]) + len(ne[0])
        return raw[idx:]
    return _strip_fences(t)


def _r02_subject(t):
    ne = _nonempty_lines(_strip_fences(t))
    return bool(ne) and ne[0].lstrip("*_# ").startswith("제목:")


def _r02_para(t):
    return 1 <= len(_paragraphs(_mail_body(t))) <= 3


def _r02_len(t):
    return _nospace_len(_mail_body(t)) <= 300


def _r02_date(t):
    return "2026년 8월 10일" in (t or "")


def _r02_polite(t):
    return _polite_ratio(_mail_body(t)) >= 0.9


def _r03_table(t):
    return _table(t)[0] is not None


def _r03_only(t):
    hdr, _, idxs = _table(t)
    if hdr is None:
        return False
    lines = _lines(_strip_fences(t))
    return all((not ln.strip()) or (i in idxs) for i, ln in enumerate(lines))


def _r03_cols(t):
    hdr, _, _ = _table(t)
    return hdr is not None and len(hdr) == 3


def _r03_rows(t):
    return _table(t)[1] == 4


def _r03_hdr(t):
    hdr, _, _ = _table(t)
    if hdr is None:
        return False
    j = " ".join(hdr)
    return all(w in j for w in ("도구", "장점", "단점"))


RUBRICS = {
    "PRM-01": [("250~400자", _r01_len), ("3문단 이하", _r01_para), ("전 문장 존댓말", _r01_polite),
               ("'자영업자' 포함", _r01_word), ("끝줄 '한 줄 요약:'", _r01_tail)],
    "PRM-02": [("첫줄 '제목:'", _r02_subject), ("본문 3문단 이하", _r02_para),
               ("본문 300자 이하", _r02_len), ("날짜 정확 표기", _r02_date),
               ("전 문장 존댓말", _r02_polite)],
    "PRM-03": [("마크다운 표 존재", _r03_table), ("표 외 산문 없음", _r03_only),
               ("컬럼 정확히 3", _r03_cols), ("데이터 행 정확히 4", _r03_rows),
               ("헤더 도구·장점·단점", _r03_hdr)],
}

# 정성 관찰(점수 아님) — V가 못 들은 날짜를 '지어내는지' vs '비워두는지'
_PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}|○+|OO|XX|__+|■+|●●")
_CONCRETE_DATE = re.compile(r"\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}[-./]\d{1,2}[-./]\d{1,2}")


def date_behavior(t):
    if "2026년 8월 10일" in (t or ""):
        return "정확"
    if _CONCRETE_DATE.search(t or ""):
        return "지어냄"
    if _PLACEHOLDER.search(t or ""):
        return "플레이스홀더"
    return "날짜없음"


def score_dir(run_dir):
    with open(os.path.join(run_dir, "run.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out = []
    for r in data.get("runs", []):
        tid = r.get("task", "")
        base, variant = tid[:-1], tid[-1:]
        if base not in RUBRICS:
            continue
        p = os.path.join(run_dir, r.get("output_file", ""))
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read()
        items = [(name, bool(fn(text))) for name, fn in RUBRICS[base]]
        rec = {
            "task": tid, "base": base, "variant": variant,
            "output_file": r.get("output_file"), "elapsed_s": r.get("elapsed_s"),
            "score": sum(1 for _, ok in items if ok), "max": len(items),
            "items": items, "chars": _nospace_len(text),
        }
        if base == "PRM-02":
            rec["date_behavior"] = date_behavior(text)
        out.append(rec)
    return data, out


def report(data, recs):
    print(f"모델: {data.get('model')}  ·  도구: {data.get('tool')}  ·  날짜: {data.get('date')}")
    print(f"총 {len(recs)}회 채점\n")
    print(f"{'task':<10}{'회차':<5}{'점수':<7}{'자수':<7}{'초':<7}미충족 항목")
    seen = defaultdict(int)
    for r in recs:
        seen[r["task"]] += 1
        miss = "·".join(n for n, ok in r["items"] if not ok) or "—"
        print(f"{r['task']:<10}{seen[r['task']]:<5}{r['score']}/{r['max']:<5}"
              f"{r['chars']:<7}{r['elapsed_s']:<7}{miss}")

    print("\n[ 집계 · 업무별 V vs F ]")
    print(f"{'업무':<10}{'V 평균':<12}{'F 평균':<12}차이")
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        agg[r["base"]][r["variant"]].append(r["score"])
    for base in sorted(agg):
        v = agg[base].get("V", []); f = agg[base].get("F", [])
        av = sum(v) / len(v) if v else 0
        af = sum(f) / len(f) if f else 0
        print(f"{base:<10}{av:.2f}/5 (n={len(v)}) {af:.2f}/5 (n={len(f)})  +{af - av:.2f}")
    allv = [r["score"] for r in recs if r["variant"] == "V"]
    allf = [r["score"] for r in recs if r["variant"] == "F"]
    if allv and allf:
        pv, pf = sum(allv) / (5 * len(allv)) * 100, sum(allf) / (5 * len(allf)) * 100
        print(f"\n전체 충족률   V(막연) {pv:.1f}%   →   F(4칸) {pf:.1f}%")

    print("\n[ 항목별 충족률 (V → F) ]")
    per = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for n, ok in r["items"]:
            per[(r["base"], n)][r["variant"]].append(ok)
    for (base, name) in sorted(per):
        d = per[(base, name)]
        v = d.get("V", []); f = d.get("F", [])
        rv = sum(v) / len(v) * 100 if v else 0
        rf = sum(f) / len(f) * 100 if f else 0
        flag = "  ← 말해줘도 실패" if rf < 100 else ""
        print(f"  {base} {name:<22} {rv:5.0f}% → {rf:5.0f}%{flag}")

    db = [r.get("date_behavior") for r in recs if r["base"] == "PRM-02" and r["variant"] == "V"]
    if db:
        print(f"\n[ 정성관찰 ] PRM-02V(날짜를 안 알려준 경우) 모델 행동: "
              + " · ".join(f"{x}={db.count(x)}" for x in sorted(set(db))))


def make_chart(recs, path):
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

    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        agg[r["base"]][r["variant"]].append(r["score"])
    labels = {"PRM-01": "블로그 소개문", "PRM-02": "거래처 이메일", "PRM-03": "비교표"}
    bases = sorted(agg)
    vs = [sum(agg[b]["V"]) / len(agg[b]["V"]) for b in bases]
    fs = [sum(agg[b]["F"]) / len(agg[b]["F"]) for b in bases]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    x = range(len(bases))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], vs, w, label="막연한 한 줄", color="#c8ccd4")
    b2 = ax.bar([i + w / 2 for i in x], fs, w, label="4칸 프롬프트", color="#2f6fed")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{bar.get_height():.1f}", ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(b, b) for b in bases], fontsize=11)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("요구조건 충족 개수 (5점 만점)")
    ax.set_title("같은 모델, 프롬프트만 바꿨을 때 — 요구조건 충족도 (gemma3:4b · 업무당 3회)",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print(f"\n차트 저장: {path}")


def main():
    ap = argparse.ArgumentParser(description="PRM 프롬프트 A/B 결정론 채점")
    ap.add_argument("run_dir")
    ap.add_argument("--chart", help="차트 PNG 저장 경로")
    ap.add_argument("--json", help="채점 결과 JSON 저장 경로")
    a = ap.parse_args()
    data, recs = score_dir(a.run_dir)
    if not recs:
        print("PRM run이 없습니다 — run.yaml의 task ID 확인.", file=sys.stderr)
        sys.exit(2)
    report(data, recs)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 저장: {a.json}")
    if a.chart:
        make_chart(recs, a.chart)


if __name__ == "__main__":
    main()
