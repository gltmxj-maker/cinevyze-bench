# -*- coding: utf-8 -*-
"""stt_score.py — Whisper STT 실측 채점(커밋 재현물).

tool_test_harness가 생성한 test_runs/whisper-<size>-<date>/ (run.yaml + NN-output.txt + NN-invocation.log)를
읽어 corpus/stt/refs.json 정답과 대조 → CER(주지표·한국어)·WER(어절)·RTF·load 집계.

★수치의 write-origin = harness(generated_by=tool_test_harness) 출력. 이 스크립트는 그 출력을 '채점'만 한다
  (전사 생성은 harness·adapter). 지표는 자체검증(_selftest) 통과 후 계산.

사용: python stt_score.py [YYYYMMDD]   (기본=오늘)
"""
import os, sys, re, json, glob, unicodedata, statistics, datetime

import yaml
import soundfile as sf  # noqa: F401  (미사용이나 audio 검증 의존 명시)
import bench_config as cfg

SIZES = ["tiny", "base", "small", "large-v3", "large-v3-turbo"]


# ── 지표(run_bench.py와 동일 정의·독립 구현) ──────────────────────────────
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
    assert cer("겨울에북발트해", ",,,,,,,,") == 1.0 and cer("가나다", "") == 1.0
    print("[selftest] CER/WER metric OK")


def _log_meta(log_path):
    """invocation.log에서 harness가 기록한 재현 파라미터(audio_s·infer_s·load_s) 파싱."""
    meta = {}
    if not os.path.exists(log_path):
        return meta
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(audio_s|infer_s|load_s|rtf|elapsed_s):\s*([\d.]+)", line)
            if m:
                meta[m.group(1)] = float(m.group(2))
            if line.startswith("--- PROMPT ---"):
                break
    return meta


def score_model(size, date_str):
    run_dir = os.path.join(cfg.ADSENSE_TEST_RUNS_DIR, f"whisper-{cfg.safe_name(size)}-{date_str}")
    ry = os.path.join(run_dir, "run.yaml")
    if not os.path.exists(ry):
        print(f"[skip] {size}: run.yaml 없음 ({ry})")
        return None
    with open(ry, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    refs = json.load(open(os.path.join(cfg.BASE_DIR, "corpus/stt/refs.json"), encoding="utf-8"))
    per = []
    for r in data.get("runs", []):
        clip = os.path.basename(r.get("input", ""))          # corpus/stt/clip-0N.wav → clip-0N.wav
        ref = refs.get(clip, {}).get("ref", "")
        if not ref:
            print(f"[warn] {size}: {clip} 정답 없음 — 스킵"); continue
        hyp = open(os.path.join(run_dir, r["output_file"]), encoding="utf-8").read().strip()
        meta = _log_meta(os.path.join(run_dir, r["log_file"]))
        per.append({"clip": clip, "cer": cer(ref, hyp), "wer": wer(ref, hyp),
                    "audio_s": meta.get("audio_s"), "infer_s": meta.get("infer_s"),
                    "load_s": meta.get("load_s"), "ref": ref, "hyp": hyp})
    if not per:
        return None
    warm = per[1:] or per                                     # 1번째=모델로드 포함 → RTF는 warm(2번째~)
    aud = sum(p["audio_s"] for p in warm if p["audio_s"])
    inf = sum(p["infer_s"] for p in warm if p["infer_s"])
    return {
        "model": size, "n": len(per),
        "mean_cer": round(statistics.mean(p["cer"] for p in per), 4),
        "median_cer": round(statistics.median(p["cer"] for p in per), 4),
        "mean_wer": round(statistics.mean(p["wer"] for p in per), 4),
        "rtf_warm": round(inf / aud, 3) if aud else None,
        "load_s": per[0].get("load_s"),
        "total_audio_s": round(sum(p["audio_s"] for p in per if p["audio_s"]), 1),
        "per": per,
    }


def main():
    _selftest()
    date_str = (sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()).replace("-", "")
    rows = [r for r in (score_model(s, date_str) for s in SIZES) if r]
    print("\n" + "=" * 70)
    print(f"{'model':16}{'n':>3}{'meanCER':>9}{'medCER':>8}{'meanWER':>9}{'RTF_warm':>10}{'load_s':>8}")
    for r in rows:
        print(f"{r['model']:16}{r['n']:>3}{r['mean_cer']:>8.1%}{r['median_cer']:>8.1%}"
              f"{r['mean_wer']:>8.1%}{str(r['rtf_warm']):>10}{str(r['load_s']):>8}")
    print("=" * 70)
    out = os.path.join(cfg.ADSENSE_TEST_RUNS_DIR, f"whisper-stt-score-{date_str}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "per"} | {"per": r["per"]} for r in rows],
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")
    return rows


if __name__ == "__main__":
    main()
