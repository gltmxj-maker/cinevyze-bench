# -*- coding: utf-8 -*-
"""ocr_score.py — Tesseract OCR 실측 채점(커밋 재현물).

tool_test_harness가 생성한 test_runs/tesseract-<lang>-<date>/ (run.yaml + NN-output.txt + NN-invocation.log)를
읽어 corpus/ocr/refs.json 정답과 대조 → 조건별(clean/small/lowres/blur/noise/rotate) CER·WER·속도 집계.
두 언어설정(kor·kor+eng)을 나란히 비교 = 영문 토큰(OCR·AI) 오독이 -l 설정으로 갈리는지 정량화.

★수치의 write-origin = harness(generated_by=tool_test_harness) 출력. 이 스크립트는 채점만(전사=tesseract).
  지표는 자체검증(_selftest) 통과 후 계산 — stt_score.py와 동일 정의·독립 구현.

사용: python3 ocr_score.py [YYYYMMDD]   (기본=오늘)
"""
import os, sys, re, json, unicodedata, statistics, datetime

import yaml
import bench_config as cfg

CONFIGS = ["kor", "kor+eng"]


# ── 지표(stt_score.py와 동일 정의·독립 구현) ──────────────────────────────
def _lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return la or lb
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb]


_PUNCT = re.compile(r"[^\w가-힣]+", re.UNICODE)


def _norm(s, keep_space):
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if keep_space else s.replace(" ", "")


def cer(ref, hyp):
    r, h = _norm(ref, False), _norm(hyp, False)
    return (_lev(list(r), list(h)) / len(r)) if r else (0.0 if not h else 1.0)


def wer(ref, hyp):
    r, h = _norm(ref, True).split(), _norm(hyp, True).split()
    return (_lev(r, h) / len(r)) if r else (0.0 if not h else 1.0)


def _selftest():
    assert _lev(list("kitten"), list("sitting")) == 3
    assert abs(cer("가나다라", "가나다마") - 0.25) < 1e-9
    assert abs(wer("오늘 날씨 좋다", "오늘 날씨 나쁘다") - 1 / 3) < 1e-9
    assert cer("가나다", "") == 1.0 and cer("", "") == 0.0
    print("[selftest] CER/WER metric OK")


def _log_meta(log_path):
    """invocation.log에서 harness가 기록한 재현 파라미터(proc_s) 파싱."""
    meta = {}
    if not os.path.exists(log_path):
        return meta
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(proc_s|elapsed_s):\s*([\d.]+)", line)
            if m:
                meta[m.group(1)] = float(m.group(2))
            if line.startswith("--- PROMPT ---"):
                break
    return meta


def score_config(config, date_str, refs):
    run_dir = os.path.join(cfg.ADSENSE_TEST_RUNS_DIR, f"tesseract-{cfg.safe_name(config)}-{date_str}")
    ry = os.path.join(run_dir, "run.yaml")
    if not os.path.exists(ry):
        print(f"[skip] {config}: run.yaml 없음 ({ry})")
        return None
    with open(ry, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    per = []
    for r in data.get("runs", []):
        clip = os.path.basename(r.get("input", ""))          # corpus/ocr/clip-0N.png → clip-0N.png
        meta_ref = refs.get(clip, {})
        ref = meta_ref.get("ref", "")
        if not ref:
            print(f"[warn] {config}: {clip} 정답 없음 — 스킵"); continue
        hyp = open(os.path.join(run_dir, r["output_file"]), encoding="utf-8").read().strip()
        m = _log_meta(os.path.join(run_dir, r["log_file"]))
        per.append({"clip": clip, "condition": meta_ref.get("condition"), "desc": meta_ref.get("desc"),
                    "cer": cer(ref, hyp), "wer": wer(ref, hyp), "proc_s": m.get("proc_s"),
                    "ref": ref, "hyp": hyp})
    if not per:
        return None
    return {
        "config": config, "n": len(per),
        "mean_cer": round(statistics.mean(p["cer"] for p in per), 4),
        "median_cer": round(statistics.median(p["cer"] for p in per), 4),
        "mean_wer": round(statistics.mean(p["wer"] for p in per), 4),
        "mean_proc_s": round(statistics.mean(p["proc_s"] for p in per if p["proc_s"]), 3)
        if any(p["proc_s"] for p in per) else None,
        "per": per,
    }


def main():
    _selftest()
    date_str = (sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()).replace("-", "")
    refs = json.load(open(os.path.join(cfg.BASE_DIR, "corpus/ocr/refs.json"), encoding="utf-8"))
    rows = [r for r in (score_config(c, date_str, refs) for c in CONFIGS) if r]
    if not rows:
        print("[!] 채점할 run 없음 — harness run 먼저."); return []

    # 조건별 CER 표(행=조건·열=설정) — 화질 열화 효과 + kor vs kor+eng 대비
    order = [p["clip"] for p in rows[0]["per"]]
    by = {r["config"]: {p["clip"]: p for p in r["per"]} for r in rows}
    print("\n" + "=" * 72)
    print(f"{'조건':22}" + "".join(f"{c + ' CER':>13}" for c in CONFIGS))
    print("-" * 72)
    for clip in order:
        desc = by[rows[0]["config"]][clip]["desc"] or clip
        line = f"{desc:22}"
        for c in CONFIGS:
            p = by.get(c, {}).get(clip)
            line += f"{(p['cer'] if p else float('nan')):>12.1%} "
        print(line)
    print("-" * 72)
    for r in rows:
        print(f"[{r['config']:8}] meanCER={r['mean_cer']:.1%}  medianCER={r['median_cer']:.1%}  "
              f"meanWER={r['mean_wer']:.1%}  이미지당~{r['mean_proc_s']}s  n={r['n']}")
    print("=" * 72)

    out = os.path.join(cfg.ADSENSE_TEST_RUNS_DIR, f"tesseract-ocr-score-{date_str}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")
    return rows


if __name__ == "__main__":
    main()
