#!/usr/bin/env python3
"""빅3 같은 과제 채점 하네스 (2026-09-25 · 빅3 재제작 R2).

응답 수집은 이 스크립트가 하지 않는다 — 시네 크롬(로그인 상태)에서 세션이 새 대화마다
같은 문장을 넣고 받은 원문을 `responses.jsonl` 에 옮겨 적었다(session-chrome, 캡처 등록됨).
이 스크립트는 그 원문을 **다시 채점**하고 결과·집계·run.yaml 을 쓴다.

  T1 표 집계   — 응답 본문의 지역별 합계·최대 지역·전체 합계를 정답표와 대조
  T2 엑셀 수식 — 시험 시트(실날짜 8·텍스트 날짜 4)에 수식을 넣고 LibreOffice headless 로 계산
  T3 파이썬    — 받은 함수를 불러 시험 11건 실행

과제 문장·정답표·시험 11건 = `big3_same_task_cases.json`. 응답 원문(`test_runs/<run>/responses.jsonl`)과
받은 코드(`code/*_T3.py`)는 비공개 런 폴더에 있다 — 재현하려면 같은 문장을 직접 넣고 그 형식으로 옮겨 적는다.

사용: python3 run_big3_same_task_grade.py [--run big3-same-task-20260925]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile

from live_mark import mark

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["ChatGPT", "Claude", "Gemini"]

T2_DATA = [("2026-09-01", "date", 1000), ("2026-08-31", "date", 2000), ("2026-09-15", "text", 4000),
           ("2026-09-30", "date", 8000), ("2026-10-01", "date", 16000), ("2026-09-02", "text", 32000),
           ("2025-09-10", "date", 64000), ("2026-08-20", "text", 128000), ("2026-09-18", "date", 256000),
           ("2026-09-29", "text", 512000), ("2026-09-05", "date", 1024), ("2026-11-09", "date", 2048)]


def _nums(text):
    return [int(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*", text or "")]


def grade_t1(rec, truth):
    text = rec.get("text") or ""
    fails = []
    for region, want in truth["by_region"].items():
        line = next((ln for ln in text.splitlines() if region in ln and "가장" not in ln), "")
        if want not in _nums(line):
            fails.append(f"{region} 합계 {want:,} 없음")
    top_line = next((ln for ln in text.splitlines() if "가장" in ln), "")
    if truth["top"] not in top_line:
        fails.append(f"최대 지역 {truth['top']} 아님")
    total_line = next((ln for ln in text.splitlines() if "전체" in ln), "")
    if truth["total"] not in _nums(total_line):
        fails.append(f"전체 합계 {truth['total']:,} 없음")
    return {"ok": not fails, "detail": "; ".join(fails) or "표·최대 지역·전체 합계 일치"}


def _fods(formulas):
    def cell(d, k):
        if k == "date":
            return (f'<table:table-cell office:value-type="date" office:date-value="{d}">'
                    f'<text:p>{d}</text:p></table:table-cell>')
        return f'<table:table-cell office:value-type="string"><text:p>{d}</text:p></table:table-cell>'
    rows = ['<table:table-row><table:table-cell office:value-type="string"><text:p>날짜</text:p></table:table-cell>'
            '<table:table-cell office:value-type="string"><text:p>금액</text:p></table:table-cell></table:table-row>']
    for d, k, a in T2_DATA:
        rows.append(f'<table:table-row>{cell(d, k)}<table:table-cell office:value-type="float" '
                    f'office:value="{a}"><text:p>{a}</text:p></table:table-cell></table:table-row>')
    rows.append('<table:table-row><table:table-cell/></table:table-row>')
    cells = "".join(f'<table:table-cell table:formula="msoxl:{html.escape(f, quote=True)}"/>' for f in formulas)
    rows.append(f"<table:table-row>{cells}</table:table-row>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<office:document '
            'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
            'xmlns:msoxl="http://schemas.microsoft.com/office/excel/formula" office:version="1.3" '
            'office:mimetype="application/vnd.oasis.opendocument.spreadsheet"><office:body><office:spreadsheet>'
            f'<table:table table:name="t2">{"".join(rows)}</table:table>'
            '</office:spreadsheet></office:body></office:document>')


def lo_eval(formulas, keep_dir):
    """수식 목록 → LibreOffice 가 계산한 값 문자열 목록."""
    tmp = tempfile.mkdtemp(prefix="big3t2-")
    try:
        src = os.path.join(tmp, "sheet.fods")
        with open(src, "w", encoding="utf-8") as f:
            f.write(_fods(formulas))
        subprocess.run(["soffice", "--headless", "--calc", "--convert-to", "csv", "--outdir", tmp, src],
                       check=True, capture_output=True, timeout=180)
        out = os.path.join(tmp, "sheet.csv")
        with open(out, encoding="utf-8") as f:
            last = [r for r in csv.reader(f)][-1]
        os.makedirs(keep_dir, exist_ok=True)
        shutil.copy(src, os.path.join(keep_dir, "sheet.fods"))
        shutil.copy(out, os.path.join(keep_dir, "sheet.csv"))
        return last
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def grade_t3(code_path, tests):
    spec = importlib.util.spec_from_file_location("cand", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for inp, exp in tests:
        got = mod.normalize_phone(inp)
        rows.append({"input": inp, "expected": exp, "got": got, "ok": got == exp})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="big3-same-task-20260925")
    a = ap.parse_args(argv)
    run_dir = os.path.join(ROOT, "test_runs", a.run)
    # 과제·정답·시험 케이스 = 저장소 루트 big3_same_task_cases.json(공개). 응답 원문 responses.jsonl 은 비공개 런 폴더.
    cases = os.path.join(ROOT, "big3_same_task_cases.json")
    if not os.path.isfile(cases):
        cases = os.path.join(run_dir, "tasks.json")
    tasks = json.load(open(cases, encoding="utf-8"))["tasks"]
    resp = [json.loads(ln) for ln in open(os.path.join(run_dir, "responses.jsonl"), encoding="utf-8")]
    by = {(r["service"], r["task"]): r for r in resp}
    results = []

    mark("turn", "T1 표 집계 채점 시작 — 정답 서울 430,000 · 부산 351,900 · 대구 157,300 · 전체 939,200")
    for svc in ORDER:
        g = grade_t1(by[(svc, "T1")], tasks["T1"]["truth"])
        print(f"  T1 {svc:8s} {'정답' if g['ok'] else '오답'} — {g['detail']}", flush=True)
        if not g["ok"]:
            mark("break", f"T1 {svc} 오답 — {g['detail']}")
        results.append({"task": "T1", "service": svc, "ok": g["ok"], "detail": g["detail"]})

    expected = sum(x for d, _, x in T2_DATA if d.startswith("2026-09"))
    mark("turn", f"T2 엑셀 수식 채점 시작 — LibreOffice 로 계산 · 9월 정답 {expected:,}")
    vals = lo_eval([by[(s, "T2")]["code"] for s in ORDER], os.path.join(run_dir, "t2"))
    for svc, v in zip(ORDER, vals):
        ok = v.strip() == str(expected)
        print(f"  T2 {svc:8s} 계산값 {v} → {'정답' if ok else '불일치'}", flush=True)
        if not ok:
            mark("break", f"T2 {svc} 수식이 LibreOffice 에서 {v} — 9월 정답 {expected:,} 과 다름")
        results.append({"task": "T2", "service": svc, "ok": ok, "detail": f"LibreOffice 계산값 {v}"})
    t2_check = lo_eval(["=LET(x,5,x*2)",
                        "=SUMPRODUCT((IF(ISNUMBER(A2:A13),A2:A13,DATEVALUE(A2:A13))>=DATE(2026,9,1))"
                        "*(IF(ISNUMBER(A2:A13),A2:A13,DATEVALUE(A2:A13))<DATE(2026,10,1))*B2:B13)"],
                       os.path.join(run_dir, "t2", "check"))
    print(f"  T2 확인 — LET 단독 =LET(x,5,x*2) → {t2_check[0]} · ChatGPT 식에서 LET 만 풀어 쓴 식 → {t2_check[1]}",
          flush=True)

    mark("turn", f"T3 휴대폰 번호 함수 채점 시작 — 시험 {len(tasks['T3']['tests'])}건")
    for svc in ORDER:
        rows = grade_t3(os.path.join(run_dir, "code", f"{svc.lower()}_T3.py"), tasks["T3"]["tests"])
        n_ok = sum(r["ok"] for r in rows)
        print(f"  T3 {svc:8s} {n_ok}/{len(rows)}", flush=True)
        for r in rows:
            if not r["ok"]:
                mark("break", f"T3 {svc} 실패 — 입력 {r['input']!r} · 기대 {r['expected']!r} · 받은 값 {r['got']!r}")
        results.append({"task": "T3", "service": svc, "ok": n_ok == len(rows),
                        "detail": f"{n_ok}/{len(rows)}", "cases": rows})

    agg = {s: {t: next(r["ok"] for r in results if r["service"] == s and r["task"] == t)
               for t in ("T1", "T2", "T3")} for s in ORDER}
    agg["t3_pass"] = {s: next(r["detail"] for r in results if r["service"] == s and r["task"] == "T3") for s in ORDER}
    agg["t2_check"] = {"let_alone": t2_check[0], "chatgpt_logic_without_let": t2_check[1], "expected": expected}
    summary = " · ".join(f"{s} T1 {'O' if agg[s]['T1'] else 'X'} T2 {'O' if agg[s]['T2'] else 'X'} "
                         f"T3 {agg['t3_pass'][s]}" for s in ORDER)
    mark("agg", f"집계 — {summary}")

    json.dump(results, open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(run_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "service", "ok", "detail"])
        for r in results:
            w.writerow([r["task"], r["service"], r["ok"], r["detail"]])
    json.dump(agg, open(os.path.join(run_dir, "aggregate.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 응답 원문·채점 로그를 서비스×과제마다 파일로 남긴다 — run.yaml runs/compare 가 이 파일을 가리킨다.
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    runs, compare = [], []
    for r in results:
        rec = by[(r["service"], r["task"])]
        stem = f"{r['service'].lower()}_{r['task']}"
        body = rec.get("code") or ""
        if rec.get("code_file"):
            body = open(os.path.join(run_dir, rec["code_file"]), encoding="utf-8").read()
        with open(os.path.join(raw_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"# {r['service']} {r['task']} · {rec.get('model')} · {rec.get('at')}\n# {rec.get('url')}\n"
                    f"## prompt\n{tasks[r['task']]['prompt']}\n## response\n{body}\n{rec.get('text') or ''}\n")
        with open(os.path.join(raw_dir, stem + ".grade.log"), "w", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
        runs.append({"task": r["task"], "service": r["service"], "output_file": f"raw/{stem}.txt",
                     "log_file": f"raw/{stem}.grade.log", "notes": r["detail"]})
        compare.append({"task": r["task"], "service": r["service"], "ok": r["ok"], "detail": r["detail"]})

    y_path = os.path.join(run_dir, "run.yaml")
    y = json.load(open(y_path, encoding="utf-8")) if os.path.isfile(y_path) else {}
    y.update({"runs": runs, "compare": compare, "access": "web_session",
              "method": "BIG3-T1·T2·T3 (tasks.json)"})
    y.update({"tool": "big3-same-task", "generated_by": "run_big3_same_task_grade.py",
              "date": dt.date.today().isoformat(), "graded_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
              "models_ui": {s: by[(s, "T1")].get("model") + " · " + (by[(s, "T1")].get("effort_ui") or "")
                            for s in ORDER},
              "responses_source": "responses.jsonl = 세션이 크롬 대화 화면 DOM 에서 옮긴 원문(session-chrome 캡처 등록)",
              "grader": "LibreOffice " + subprocess.run(["soffice", "--version"], capture_output=True,
                                                         text=True).stdout.split("(")[0].strip()})
    json.dump(y, open(y_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[+] results.json · results.csv · aggregate.json · run.yaml", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
