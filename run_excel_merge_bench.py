#!/usr/bin/env python3
"""엑셀 취합 자동화 하네스(2026-09-23 · 19번 재제작) — Gemini 3.1 Pro 가 짠 스크립트를 실제로 돌려 정답과 대조한다.

2026-06-27 첫 런(AUTO-01)은 Antigravity 구독 CLI 로 스크립트를 **1회** 생성해 원본 CSV 에서 실행시간만 쟀다.
재제작은 측정 경로가 **Vertex AI REST(gemini-3.1-pro-preview · global)** 로 바뀐다 — 본문에 공시한다.
  - 지시 2종 = 첫 런 지시문 그대로(plain) / 그 끝에 「이 CSV들은 엑셀에서 저장한 파일이다.」 한 줄(hint)
  - 조건별 생성 5회 · 생각 수준 high · 온도 미지정
  - 생성된 스크립트마다 데이터 3벌에서 실행(각각 임시 폴더 · 제한시간 60초 · `python3 -I`):
      clean       = 첫 런 원본 12개 파일(UTF-8 · CRLF)
      excel_utf8  = 같은 행을 UTF-8 BOM + CRLF 로 다시 쓴 것(엑셀 「CSV UTF-8」 저장 형식을 재현)
      excel_cp949 = 같은 행을 CP949 + CRLF 로 다시 쓴 것(한국어 윈도우 엑셀 「CSV」 저장 형식을 재현)
    ★엑셀로 직접 저장한 파일이 아니라 그 형식을 파이썬으로 재현한 것이다 — 본문에 공시한다.
  - 정답 = 원본 행에서 독립 계산(고유 12,034행 · 월별 합계). merged.csv 행 집합·정렬·헤더·콘솔 월 합계를 대조.
  - 실행시간 = clean 에서 통과한 스크립트만 5회 wall.
  - 실촬영 마커(본문 예산 = break ≤2 + agg 1): clean 첫 실패 = break · 엑셀 형식 첫 실패 = break ·
    조건 전환 = turn · 집계 = agg · 호출 오류/잘림 = infra

usage:
  python3 run_excel_merge_bench.py --mode pilot --run-dir test_runs/gemini-31-pro-vertex-excelmerge-20260923
  python3 run_excel_merge_bench.py --mode full  --run-dir test_runs/gemini-31-pro-vertex-excelmerge-20260923
  python3 run_excel_merge_bench.py --rescore     --run-dir test_runs/gemini-31-pro-vertex-excelmerge-20260923
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import vertex_gemini as vg
from live_mark import mark as live_mark

HARNESS_VERSION = "1.1"  # 1.1 = 2026-09-23 T2 수리(파서·타이밍·격리 상한·재채점 시 실행시간 승계) · 판정 동일
BASE = Path(__file__).resolve().parent
SRC_DIR = BASE / "corpus" / "auto"
DEFAULT_RUN_DIR = BASE / "test_runs" / "gemini-31-pro-vertex-excelmerge-20260923"
MODEL = "gemini-3.1-pro-preview"
THINKING = "high"
MAX_OUT = 32768
ARMS = ("plain", "hint")
ARM_KO = {"plain": "첫 런 지시문 그대로", "hint": "「엑셀에서 저장한 파일」 한 줄 추가"}
HINT_LINE = "이 CSV들은 엑셀에서 저장한 파일이다."
DATASETS = ("clean", "excel_utf8", "excel_cp949")
DS_KO = {"clean": "원본(UTF-8)", "excel_utf8": "엑셀 CSV UTF-8 형식(BOM)", "excel_cp949": "엑셀 CSV 형식(CP949)"}
REPS = 5
TIMING_RUNS = 5
BASE_PROMPT = ("corpus/auto/ 폴더에 sales_2026-01.csv 부터 sales_2026-12.csv 까지 월별 매출 파일 12개가 있다"
               "(각 파일 컬럼: date,item,amount · date 형식 YYYY-MM-DD). 이 12개를 모두 합쳐 ①완전히 동일한 중복 행을 "
               "제거하고 ②date 오름차순 정렬한 뒤 ③corpus/auto/merged.csv 로 저장하고, ④월(YYYY-MM)별 amount 합계를 "
               "콘솔에 출력하는 파이썬 스크립트를 작성해줘. 표준 라이브러리만 사용하고, 실행 가능한 전체 코드를 한 블록으로 줘.")


def build_prompt(arm: str) -> str:
    return BASE_PROMPT + (("\n" + HINT_LINE) if arm == "hint" else "")


# ---------- 데이터 · 정답 ----------

def source_rows() -> tuple[list[str], dict[str, list[list[str]]]]:
    files = {}
    header = None
    for fp in sorted(glob.glob(str(SRC_DIR / "sales_2026-*.csv"))):
        with open(fp, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        files[Path(fp).name] = [r for r in rows[1:] if r]
    return header, files


def reference() -> dict[str, Any]:
    _, files = source_rows()
    raw = [tuple(r) for rs in files.values() for r in rs]
    uniq = set(raw)
    sums: dict[str, int] = defaultdict(int)
    for d, _i, a in uniq:
        sums[d[:7]] += int(a)
    return {"files": len(files), "raw_rows": len(raw), "unique_rows": len(uniq), "dups": len(raw) - len(uniq),
            "unique": uniq, "monthly": dict(sorted(sums.items()))}


def write_dataset(dst: Path, kind: str) -> None:
    corpus = dst / "corpus" / "auto"
    corpus.mkdir(parents=True)
    if kind == "clean":
        for fp in glob.glob(str(SRC_DIR / "sales_2026-*.csv")):
            shutil.copy2(fp, corpus / Path(fp).name)
        return
    header, files = source_rows()
    enc = "utf-8-sig" if kind == "excel_utf8" else "cp949"
    for name, rows in files.items():
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n")
        w.writerow(header)
        w.writerows(rows)
        (corpus / name).write_bytes(buf.getvalue().encode(enc))


def dataset_fingerprint(kind: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        write_dataset(Path(td), kind)
        p = Path(td) / "corpus" / "auto" / "sales_2026-01.csv"
        b = p.read_bytes()
        return {"first_bytes_hex": b[:8].hex(), "bom": b.startswith(b"\xef\xbb\xbf"), "crlf": b"\r\n" in b,
                "utf8_decodes": _decodes(b, "utf-8"), "bytes_jan": len(b)}


def _decodes(b: bytes, enc: str) -> bool:
    try:
        b.decode(enc)
        return True
    except UnicodeDecodeError:
        return False


# ---------- 채점 ----------

def extract_code(text: str) -> str | None:
    blocks = [m.group(1) for m in re.finditer(r"```(?:python|py)?[^\n]*\n(.*?)```", text, re.S)]
    return max(blocks, key=len) if blocks else None


def _read_merged(path: Path) -> tuple[list[list[str]] | None, str | None]:
    b = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return list(csv.reader(io.StringIO(b.decode(enc), newline=""))), enc
        except UnicodeDecodeError:
            continue
    return None, None


def parse_monthly(stdout: str) -> dict[str, float]:
    """줄마다 「YYYY-MM … 숫자」 한 쌍을 읽는다. 파일명(sales_2026-01.csv)·날짜(2026-01-12) 속 월은 건너뛰고,
    같은 달이 서로 다른 값으로 두 번 나오면 그 달은 판정 불가(NaN)로 둔다."""
    out: dict[str, float] = {}
    for ln in stdout.splitlines():
        m = re.search(r"(?<![\w-])(2026-\d{2})(?![-\d])\D{0,40}?(-?\d[\d,]*(?:\.\d+)?)", ln)
        if not m:
            continue
        v = float(m.group(2).replace(",", ""))
        k = m.group(1)
        out[k] = v if k not in out or out[k] == v else float("nan")
    return out


def run_script(code: str, kind: str, timeout: int = 60, timing: bool = False) -> dict[str, Any]:
    ref = REF
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_dataset(root, kind)
        (root / "merge_script.py").write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "HOME": td}
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, "-I", "merge_script.py"], cwd=td, capture_output=True, text=True,
                               timeout=timeout, env=env, preexec_fn=vg.limit_child)
            wall = round(time.monotonic() - t0, 4)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return {"dataset": kind, "ok": False, "exit": None, "error": f"timeout {timeout}s", "wall_s": None}
        res: dict[str, Any] = {"dataset": kind, "exit": rc, "wall_s": wall, "stdout_tail": so[-1500:],
                               "stderr_tail": se[-1500:]}
        err_line = next((ln for ln in reversed(se.strip().splitlines()) if ln.strip()), "") if se.strip() else ""
        res["error"] = err_line if rc != 0 else None
        mp = root / "corpus" / "auto" / "merged.csv"
        checks = {"exit0": rc == 0, "merged_exists": mp.exists(), "header": False, "rows_exact": False,
                  "sorted": False, "monthly_ok": False}
        if mp.exists():
            try:
                rows, enc = _read_merged(mp)
            except (csv.Error, OSError) as exc:  # 깨진 merged.csv 는 그 회차의 실패다
                rows, enc = None, f"read_error {type(exc).__name__}"
            res["merged_encoding"] = enc
            if rows:
                checks["header"] = [c.strip() for c in rows[0]] == ["date", "item", "amount"] and rows[0][0] == "date"
                body = [tuple(r) for r in rows[1:] if r]
                res["merged_rows"] = len(body)
                res["merged_header_raw"] = rows[0]
                checks["rows_exact"] = Counter(body) == Counter(ref["unique"])
                dates = [r[0] for r in body]
                checks["sorted"] = dates == sorted(dates)
        got = parse_monthly(so)
        want = ref["monthly"]
        checks["monthly_ok"] = set(got) == set(want) and all(abs(got[k] - want[k]) < 0.01 for k in want)
        res["monthly_months_parsed"] = len(got)
        res["checks"] = checks
        res["ok"] = all(checks.values())
        if timing and res["ok"]:
            walls, bad = [], None
            for _ in range(TIMING_RUNS):
                (root / "corpus" / "auto" / "merged.csv").unlink(missing_ok=True)
                t1 = time.monotonic()
                try:
                    q = subprocess.run([sys.executable, "-I", "merge_script.py"], cwd=td, capture_output=True,
                                       text=True, timeout=timeout, env=env, preexec_fn=vg.limit_child)
                except subprocess.TimeoutExpired:
                    bad = f"timeout {timeout}s"
                    break
                if q.returncode != 0 or not (root / "corpus" / "auto" / "merged.csv").exists():
                    bad = f"exit {q.returncode}"
                    break
                walls.append(round(time.monotonic() - t1, 4))
            res["timing_walls_s"] = walls
            # 반복 중 한 번이라도 실패하면 시간은 내지 않는다 — 실패한 실행을 실행시간에 섞지 않는다
            res["timing_median_s"] = statistics.median(walls) if not bad else None
            res["timing_error"] = bad
        return res


REF = reference()


def score_generation(run_dir: Path, inv: dict[str, Any], timing: bool = True) -> dict[str, Any]:
    text = (run_dir / inv["response_file"]).read_text(encoding="utf-8")
    code = extract_code(text)
    out: dict[str, Any] = {"run_id": inv["run_id"], "key": inv["key"], "arm": inv["arm"], "rep": inv["rep"],
                           "code_found": code is not None, "datasets": {}}
    if code is None:
        for k in DATASETS:
            out["datasets"][k] = {"dataset": k, "ok": False, "error": "코드 블록 없음"}
        return out
    (run_dir / inv["code_file"]).write_text(code, encoding="utf-8")
    out["code_file"] = inv["code_file"]
    out["code_mentions"] = {"utf-8-sig": "utf-8-sig" in code.lower(), "cp949": bool(re.search(r"cp949|euc-kr|euc_kr", code, re.I)),
                            "errors_arg": "errors=" in code, "encoding_fallback": code.count("encoding=") > 2}
    for k in DATASETS:
        out["datasets"][k] = run_script(code, k, timing=(timing and k == "clean"))
    return out


def aggregate(scored: list[dict[str, Any]], infra: int) -> dict[str, Any]:
    agg: dict[str, Any] = {"reference": {k: v for k, v in REF.items() if k != "unique"}, "generations": len(scored),
                           "infra_errors": infra, "arms": {}}
    for arm in ARMS:
        rs = [s for s in scored if s["arm"] == arm]
        if not rs:
            continue
        blk: dict[str, Any] = {"generations": len(rs)}
        for k in DATASETS:
            ok = [s for s in rs if s["datasets"][k]["ok"]]
            blk[k] = {"hit": len(ok), "total": len(rs),
                      "errors": Counter((s["datasets"][k].get("error") or "").split(":")[0] or
                                        ",".join(c for c, v in (s["datasets"][k].get("checks") or {}).items() if not v)
                                        for s in rs if not s["datasets"][k]["ok"])}
        blk["all_three"] = {"hit": sum(1 for s in rs if all(s["datasets"][k]["ok"] for k in DATASETS)), "total": len(rs)}
        walls = [s["datasets"]["clean"]["timing_median_s"] for s in rs
                 if s["datasets"]["clean"].get("timing_median_s") is not None]
        blk["clean_wall_median_s"] = statistics.median(walls) if walls else None
        blk["clean_wall_range_s"] = [min(walls), max(walls)] if walls else None
        blk["code_mentions"] = {m: sum(1 for s in rs if (s.get("code_mentions") or {}).get(m)) for m in
                                ("utf-8-sig", "cp949", "errors_arg", "encoding_fallback")}
        agg["arms"][arm] = blk
    return agg


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for p in (run_dir / "raw").glob("*-invocation.json"):
        keys.add(json.loads(p.read_text(encoding="utf-8"))["key"])
    for p in list((run_dir / "raw").glob("*-*")) + list((run_dir / "raw" / "_infra").glob("*-*")):
        m = re.match(r"(\d+)-", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return keys, maximum


def _clip(text: str, limit: int = 185) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


TIMING_KEYS = ("timing_walls_s", "timing_median_s", "timing_error")


def _prior_timing(run_dir: Path) -> dict[str, dict[str, Any]]:
    """기존 results.json 의 원본 데이터 실행시간. ★실행시간은 한 번만 잰다 — 재채점이 다시 재면
    발행된 숫자와 원자료가 조용히 어긋난다(T2 적대검증 2026-09-23)."""
    p = run_dir / "results.json"
    if not p.exists():
        return {}
    out = {}
    for sc in json.loads(p.read_text(encoding="utf-8")):
        c = sc.get("datasets", {}).get("clean", {})
        if any(k in c for k in TIMING_KEYS):
            out[sc["key"]] = {k: c.get(k) for k in TIMING_KEYS}
    return out


def score_all(run_dir: Path, write_pilot: bool,
              prior: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """판정은 매번 다시 실행해 매긴다(결정론). 실행시간은 prior 에 있는 회차면 이어받고 다시 재지 않는다."""
    prior = _prior_timing(run_dir) if prior is None else prior
    scored, infra = [], 0
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        inv = json.loads(p.read_text(encoding="utf-8"))
        if inv.get("infra_error"):
            infra += 1
            continue
        keep = prior.get(inv["key"])
        sc = score_generation(run_dir, inv, timing=keep is None)
        if keep is not None and sc["datasets"]["clean"].get("ok"):
            sc["datasets"]["clean"].update(keep)
        scored.append(sc)
    agg = aggregate(scored, infra)
    agg["dataset_fingerprints"] = {k: dataset_fingerprint(k) for k in DATASETS}
    (run_dir / "results.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    invs = {json.loads(p.read_text(encoding="utf-8"))["key"]: json.loads(p.read_text(encoding="utf-8"))
            for p in (run_dir / "raw").glob("*-invocation.json")}
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "key", "arm", "rep", "gen_s", "retries", "thought_tokens", "output_tokens",
                    "clean_ok", "excel_utf8_ok", "excel_cp949_ok", "clean_wall_median_s", "cp949_error"])
        for sc in scored:
            inv = invs[sc["key"]]
            u = inv.get("usage") or {}
            d = sc["datasets"]
            w.writerow([sc["run_id"], sc["key"], sc["arm"], sc["rep"], inv["attempts"][-1]["elapsed_s"],
                        len(inv["attempts"]) - 1, u.get("thoughtsTokenCount"), u.get("candidatesTokenCount"),
                        d["clean"]["ok"], d["excel_utf8"]["ok"], d["excel_cp949"]["ok"],
                        d["clean"].get("timing_median_s"), (d["excel_cp949"].get("error") or "").split(":")[0]])
    if write_pilot:
        reasons = []
        if infra:
            reasons.append(f"infra {infra}건")
        if not scored or not any(s["code_found"] for s in scored):
            reasons.append("코드 추출 0")
        agg["pilot_decision"] = {"proceed": not reasons, "reasons": reasons, "generations": len(scored)}
        (run_dir / "pilot_decision.json").write_text(json.dumps(agg["pilot_decision"], ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    (run_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored, agg


def write_run_yaml(run_dir: Path, date: str | None = None, extra: dict[str, Any] | None = None) -> None:
    entries, versions = [], set()
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("model_version"):
            versions.add(row["model_version"])
        entries.append({"key": row["key"], "arm": row["arm"], "rep": row["rep"], "mode": row["mode"],
                        "output_file": row["transcript_file"], "response_file": row["response_file"],
                        "code_file": row["code_file"], "log_file": str(p.relative_to(run_dir)),
                        "gen_s": row.get("gen_s"), "finish_reason": row.get("finish_reason"),
                        "infra_error": row.get("infra_error")})
    compare = []
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))
        for arm, blk in agg.get("arms", {}).items():
            for k in DATASETS:
                compare.append({"metric": "script_correct", "arm": arm, "dataset": k,
                                "hit": blk[k]["hit"], "total": blk[k]["total"]})
    payload = {
        "tool": "gemini", "date": date or dt.date.today().isoformat(), "method": "AUTO-R2", "access": "api",
        "model": MODEL, "model_versions_seen": sorted(versions), "generated_by": "run_excel_merge_bench.py",
        "harness_version": HARNESS_VERSION, "tos_confirmed": True,
        "tos_source_url": "Google Cloud Vertex AI generateContent REST(서비스 계정 · 종량 과금) — 첫 런(2026-06-27)은 Antigravity 구독 CLI",
        "endpoint": "aiplatform.googleapis.com · locations/global · publishers/google/models/" + MODEL,
        "arms": {k: ARM_KO[k] for k in ARMS}, "hint_line": HINT_LINE, "reps": REPS,
        "datasets": DS_KO, "dataset_note": "엑셀 형식 2벌은 엑셀로 저장한 파일이 아니라 BOM·CP949·CRLF 를 파이썬으로 재현한 것",
        "request": vg.request_body_shape(THINKING, MAX_OUT), "script_timeout_s": 60, "timing_runs": TIMING_RUNS,
        "compare": compare,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "git_sha_at_run": _git_sha()},
        "runs": entries,
    }
    if extra:
        payload.update(extra)
    tmp = run_dir / "run.yaml.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run.yaml")


def run(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    if args.rescore:
        prev = json.loads((run_dir / "run.yaml").read_text(encoding="utf-8"))
        _, agg = score_all(run_dir, write_pilot=False)
        keep = {k: prev[k] for k in ("environment", "harness_version", "request") if k in prev}
        keep.update({"rescored_at": dt.datetime.now().isoformat(timespec="seconds"), "rescored_git_sha": _git_sha()})
        write_run_yaml(run_dir, date=prev["date"], extra=keep)
        print(json.dumps({a: {k: b[k] for k in DATASETS} for a, b in agg["arms"].items()}, ensure_ascii=False, default=str))
        return 0
    reps = (1,) if args.mode == "pilot" else tuple(range(1, REPS + 1))
    plan = [{"key": f"{arm}/r{rep}", "arm": arm, "rep": rep} for arm in ARMS for rep in reps]
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "generations": len(plan), "reference": {k: v for k, v in REF.items() if k != "unique"},
                          "fingerprints": {k: dataset_fingerprint(k) for k in DATASETS},
                          "prompt_hint": build_prompt("hint")[-80:], "endpoint": vg.endpoint(MODEL)},
                         ensure_ascii=False, indent=1))
        return 0
    if args.mode == "full":
        dp = run_dir / "pilot_decision.json"
        if not dp.exists() or not json.loads(dp.read_text(encoding="utf-8")).get("proceed"):
            print("파일럿 판정 없음/중단 — 같은 --run-dir 에서 --mode pilot 을 먼저 통과하세요.")
            return 3
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.retry_infra:
        moved = 0
        for p in sorted((run_dir / "raw").glob("*-invocation.json")):
            if json.loads(p.read_text(encoding="utf-8")).get("infra_error"):
                (run_dir / "raw" / "_infra").mkdir(exist_ok=True)
                stem = p.name.split("-")[0]
                for q in (run_dir / "raw").glob(f"{stem}-*"):
                    q.rename(run_dir / "raw" / "_infra" / q.name)
                moved += 1
        print(f"[retry-infra] infra 회차 {moved}건을 raw/_infra/ 로 옮겼다(번호는 이어서 새로 붙는다)")
    existing, next_id = _existing(run_dir)
    pending = [r for r in plan if r["key"] not in existing]
    print(f"mode={args.mode} model={MODEL} planned={len(plan)} existing={len(plan) - len(pending)} running_now={len(pending)}")
    print(f"정답(독립 계산) — 파일 {REF['files']}개 · 원본 {REF['raw_rows']:,}행 · 고유 {REF['unique_rows']:,}행 · 중복 {REF['dups']}행")
    seen: set[str] = set()
    for p in (run_dir / "results.json",):
        if p.exists():
            for s in json.loads(p.read_text(encoding="utf-8")):
                if not s["datasets"]["clean"]["ok"]:
                    seen.add("clean")
                if not all(s["datasets"][k]["ok"] for k in ("excel_utf8", "excel_cp949")):
                    seen.add("excel")
    if pending:
        live_mark("turn", _clip(("파일럿 시작" if args.mode == "pilot" else "본측정으로 전환") +
                                f" — {MODEL}(Vertex) · 지시 2종 × 생성 {len(reps)}회 · 스크립트마다 데이터 3벌 실행 · {len(pending)}회 생성"))
    live_timing = _prior_timing(run_dir)
    prev_arm = None
    for pos, row in enumerate(pending, 1):
        if prev_arm is not None and row["arm"] != prev_arm:
            live_mark("turn", _clip(f"조건 전환 — {ARM_KO[prev_arm]} → {ARM_KO[row['arm']]}"))
        prev_arm = row["arm"]
        prompt = build_prompt(row["arm"])
        resp, attempts, infra = vg.generate(MODEL, prompt, thinking_level=THINKING, max_output_tokens=MAX_OUT,
                                            timeout=args.timeout)
        ex = vg.extract(resp)
        text = ex["text"]
        if resp is not None and not infra:
            if ex["finish_reason"] != "STOP":
                infra = f"finishReason={ex['finish_reason']}"
            elif not text.strip():
                infra = "empty_response"
        if infra or any(a["error"] for a in attempts):
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · {infra or attempts[0]['error']} · 시도 {len(attempts)}회"))
        next_id += 1
        stem = f"{next_id:03d}"
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "arm": row["arm"], "rep": row["rep"],
               "prompt": prompt, "model": MODEL, "request": vg.request_body_shape(THINKING, MAX_OUT),
               "attempts": attempts, "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "finish_reason": ex["finish_reason"], "model_version": ex["model_version"], "usage": ex["usage"],
               "response_file": f"raw/{stem}-response.txt", "transcript_file": f"raw/{stem}-output.txt",
               "code_file": f"raw/{stem}-merge_script.py"}
        (run_dir / inv["response_file"]).write_text(text, encoding="utf-8")
        (run_dir / inv["transcript_file"]).write_text(
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {MODEL}\nthinking_level: {THINKING}\n\n[PROMPT]\n{prompt}\n"
            f"[RAW RESPONSE]\n{text}\n\n[API RESPONSE JSON]\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8")
        (run_dir / f"raw/{stem}-invocation.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        if infra:
            print(f"[{pos}/{len(pending)}] {row['key']} INFRA {infra}", flush=True)
            continue
        u = ex["usage"] or {}
        s = score_generation(run_dir, inv)
        live_timing[row["key"]] = {k: s["datasets"]["clean"].get(k) for k in TIMING_KEYS}
        line = " · ".join(f"{DS_KO[k]} {'통과' if s['datasets'][k]['ok'] else '실패'}" for k in DATASETS)
        cw = s["datasets"]["clean"].get("timing_median_s")
        print(f"[{pos}/{len(pending)}] {row['key']} 생성 {inv['gen_s']:.1f}s · 생각 {u.get('thoughtsTokenCount', 0)}tok · "
              f"{line}" + (f" · 실행 중앙값 {cw:.3f}s" if cw else ""), flush=True)
        c = s["datasets"]["clean"]
        if not c["ok"] and "clean" not in seen:
            seen.add("clean")
            why = c.get("error") or ",".join(k for k, v in (c.get("checks") or {}).items() if not v)
            live_mark("break", _clip(f"원본 데이터 첫 실패 — {ARM_KO[row['arm']]}/생성{row['rep']} · {why}"))
        bad = [k for k in ("excel_utf8", "excel_cp949") if not s["datasets"][k]["ok"]]
        if bad and "excel" not in seen:
            seen.add("excel")
            k = bad[0]
            d = s["datasets"][k]
            why = d.get("error") or ",".join(x for x, v in (d.get("checks") or {}).items() if not v)
            live_mark("break", _clip(f"엑셀 형식 첫 실패 — {ARM_KO[row['arm']]}/생성{row['rep']} · {DS_KO[k]} · {why}"))

    write_run_yaml(run_dir)
    scored, agg = score_all(run_dir, write_pilot=(args.mode == "pilot"), prior=live_timing)
    write_run_yaml(run_dir)

    def part(arm: str) -> str:
        b = agg["arms"].get(arm)
        if not b:
            return ""
        return f"{ARM_KO[arm]} " + " ".join(f"{DS_KO[k].split('(')[0].strip()} {b[k]['hit']}/{b[k]['total']}" for k in DATASETS)

    complete = {r["key"] for r in plan} <= {s["key"] for s in scored}
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(scored)}회 집계 — {part('plain')} · {part('hint')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(f"스크립트 {len(scored)}개 집계 — {part('plain')} · {part('hint')} · 인프라 {agg['infra_errors']}건"))
    status = {"mode": args.mode, "generations": len(scored), "infra_errors": agg["infra_errors"], "complete": complete,
              "pilot_decision": agg.get("pilot_decision")}
    (run_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.mode == "pilot" and not (agg.get("pilot_decision") or {}).get("proceed"):
        return 3
    return 0 if complete or args.mode == "pilot" else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"))
    ap.add_argument("--rescore", action="store_true", help="기존 raw 를 재채점(과금 없음 · 판정용으로 스크립트는 다시 실행 · 실행시간은 기존 값 승계)")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-infra", action="store_true", help="infra 회차 기록을 raw/_infra/ 로 옮기고 그 회차만 다시 돈다")
    args = ap.parse_args()
    if not args.mode and not args.rescore:
        ap.error("--mode 또는 --rescore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
