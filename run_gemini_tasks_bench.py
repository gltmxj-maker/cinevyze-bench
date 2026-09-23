#!/usr/bin/env python3
"""Gemini 3.1 Pro 실무 3종 하네스(2026-09-23 · 18번 재제작) — 코딩·요약·번역을 같은 시험지로 반복한다.

2026-06-24 첫 런(`tool_test_harness` TXT-02·TXT-01·TXT-05)은 Antigravity 구독 CLI 로 과업당 3회, 9회였다.
재제작은 **측정 경로가 Vertex AI REST(gemini-3.1-pro-preview · global)로 바뀐다** — 본문에 공시한다.
  - 시험지 = 첫 런의 입력 3종을 글자 그대로 승계(`gemini_tasks_cases.json`)
  - 조건 2종 = 생각 수준 high / low(thinkingConfig.thinkingLevel) · 반복 9회 · 온도 미지정
  - 채점 = `gemini_tasks_score.py`(결정론 · 코드는 실행해서 잰다)
  - ★생각 토큰이 출력 상한을 먹는다(HUB 2026-09-23) → 상한 32768 · finishReason ≠ STOP 은 infra 로 뺀다
  - 실촬영 마커: 조건별 첫 실패 = break(본문 예산 break ≤2 + agg 1) · 조건 전환 = turn · 집계 = agg · 잘림/오류 = infra

usage:
  python3 run_gemini_tasks_bench.py --mode pilot --run-dir test_runs/gemini-31-pro-vertex-tasks-20260923
  python3 run_gemini_tasks_bench.py --mode full  --run-dir test_runs/gemini-31-pro-vertex-tasks-20260923
  python3 run_gemini_tasks_bench.py --rescore     --run-dir test_runs/gemini-31-pro-vertex-tasks-20260923
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import vertex_gemini as vg
from gemini_tasks_score import score, score_run_dir
from live_mark import mark as live_mark

HARNESS_VERSION = "1.1"  # 1.1 = 2026-09-23 T2 수리(러너 출력 방어·자원 상한·retry-infra 번호) · 판정 동일
BASE = Path(__file__).resolve().parent
DEFAULT_CASES = BASE / "gemini_tasks_cases.json"
DEFAULT_RUN_DIR = BASE / "test_runs" / "gemini-31-pro-vertex-tasks-20260923"
MODEL = "gemini-3.1-pro-preview"
ARMS = ("high", "low")
ARM_KO = {"high": "생각 수준 high", "low": "생각 수준 low"}
TASKS = ("code", "summary", "translate")
TASK_KO = {"code": "코딩", "summary": "요약", "translate": "번역"}
REPS = 9
MAX_OUT = 32768
CHECK_KO = {"all_cases": "숨은 케이스 실패", "example_shown": "예시 출력 누락", "five_sentences": "5문장 아님",
            "no_extra_numbers": "원문 밖 숫자", "terms_all": "통용 용어 누락", "no_stray_english": "영어 단어 잔존"}


def planned(mode: str) -> list[dict[str, Any]]:
    reps = (1,) if mode == "pilot" else tuple(range(1, REPS + 1))
    return [{"key": f"{arm}/r{rep}/{task}", "arm": arm, "rep": rep, "task": task}
            for arm in ARMS for rep in reps for task in TASKS]


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for p in (run_dir / "raw").glob("*-invocation.json"):
        keys.add(json.loads(p.read_text(encoding="utf-8"))["key"])
    for p in list((run_dir / "raw").glob("*-*")) + list((run_dir / "raw" / "_infra").glob("*-*")):
        m = re.match(r"(\d+)-", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return keys, maximum


def _seen(run_dir: Path, cfg: dict[str, Any]) -> set[str]:
    seen = set()
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("infra_error"):
            continue
        if not score(row["task"], (run_dir / row["response_file"]).read_text(encoding="utf-8"), cfg)["pass"]:
            seen.add(row["arm"])
    return seen


def _clip(text: str, limit: int = 185) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_yaml(run_dir: Path, date: str | None = None, extra: dict[str, Any] | None = None) -> None:
    entries = []
    versions = set()
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("model_version"):
            versions.add(row["model_version"])
        entries.append({"key": row["key"], "task": row["task"], "arm": row["arm"], "rep": row["rep"],
                        "mode": row["mode"], "output_file": row["transcript_file"],
                        "response_file": row["response_file"], "log_file": str(p.relative_to(run_dir)),
                        "gen_s": row.get("gen_s"), "finish_reason": row.get("finish_reason"),
                        "infra_error": row.get("infra_error")})
    compare = []
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))
        for arm, block in agg.get("arms", {}).items():
            for task, b in block.items():
                compare.append({"metric": "pass", "arm": arm, "task": task, **b["pass"]})
    payload = {
        "tool": "gemini",
        "date": date or dt.date.today().isoformat(),
        "method": "TXT-R2",
        "access": "api",
        "model": MODEL,
        "model_versions_seen": sorted(versions),
        "generated_by": "run_gemini_tasks_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "Google Cloud Vertex AI generateContent REST(서비스 계정 · 종량 과금) — 첫 런(2026-06-24)은 Antigravity 구독 CLI",
        "endpoint": "aiplatform.googleapis.com · locations/global · publishers/google/models/" + MODEL,
        "tasks": {t: TASK_KO[t] for t in TASKS}, "arms": {k: ARM_KO[k] for k in ARMS}, "reps": REPS,
        "request": {arm: vg.request_body_shape(arm, MAX_OUT) for arm in ARMS},
        "request_parallelism": 1,
        "compare": compare,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "git_sha_at_run": _git_sha()},
        "runs": entries,
    }
    if extra:
        payload.update(extra)
    tmp = run_dir / "run.yaml.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run.yaml")


def rescore(args: argparse.Namespace) -> int:
    prev = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
    _, agg = score_run_dir(args.run_dir, args.cases, write_pilot=False)
    keep = {k: prev[k] for k in ("environment", "harness_version", "request") if k in prev}
    keep.update({"rescored_at": dt.datetime.now().isoformat(timespec="seconds"), "rescored_git_sha": _git_sha()})
    write_run_yaml(args.run_dir, date=prev["date"], extra=keep)
    print(json.dumps({a: {t: b["pass"] for t, b in blk.items()} for a, blk in agg["arms"].items()}, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.rescore:
        return rescore(args)
    cfg = json.loads(args.cases.read_text(encoding="utf-8"))
    plan = planned(args.mode)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "keys": [r["key"] for r in plan][:6],
                          "request": vg.request_body_shape(plan[0]["arm"], MAX_OUT),
                          "endpoint": vg.endpoint(MODEL)}, ensure_ascii=False, indent=1))
        return 0
    run_dir: Path = args.run_dir
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
    seen = _seen(run_dir, cfg)
    if pending:
        if args.mode == "pilot":
            live_mark("turn", _clip(f"파일럿 시작 — {MODEL}(Vertex) · 코딩·요약·번역 × 생각 수준 high/low · {len(pending)}회"))
        else:
            live_mark("turn", _clip(f"본측정으로 전환 — {MODEL}(Vertex) · 과업 3종 × 생각 수준 2종 × 반복 {REPS}회 · "
                                    f"기존 {len(plan) - len(pending)}회 포함 {len(pending)}회 추가"))
    prev_arm = None
    for pos, row in enumerate(pending, 1):
        if prev_arm is not None and row["arm"] != prev_arm:
            live_mark("turn", _clip(f"조건 전환 — {ARM_KO[prev_arm]} → {ARM_KO[row['arm']]}"))
        prev_arm = row["arm"]
        prompt = cfg["tasks"][row["task"]]["prompt"]
        resp, attempts, infra = vg.generate(MODEL, prompt, thinking_level=row["arm"], max_output_tokens=MAX_OUT,
                                            timeout=args.timeout)
        ex = vg.extract(resp)
        text = ex["text"]
        if resp is not None and not infra:
            if ex["finish_reason"] != "STOP":
                infra = f"finishReason={ex['finish_reason']}"  # 잘린 답은 오답이 아니라 측정 실패다
            elif not text.strip():
                infra = "empty_response"
        if infra or any(a["error"] for a in attempts):
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · {infra or attempts[0]['error']} · 시도 {len(attempts)}회"))
        next_id += 1
        stem = f"{next_id:03d}"
        resp_rel, tr_rel, inv_rel = f"raw/{stem}-response.txt", f"raw/{stem}-output.txt", f"raw/{stem}-invocation.json"
        (run_dir / resp_rel).write_text(text, encoding="utf-8")
        (run_dir / tr_rel).write_text(
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {MODEL}\nthinking_level: {row['arm']}\n\n[PROMPT]\n{prompt}\n"
            f"[RAW RESPONSE]\n{text}\n\n[API RESPONSE JSON]\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8")
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "task": row["task"], "arm": row["arm"],
               "rep": row["rep"], "prompt": prompt, "model": MODEL,
               "request": vg.request_body_shape(row["arm"], MAX_OUT), "attempts": attempts,
               "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "finish_reason": ex["finish_reason"], "model_version": ex["model_version"], "usage": ex["usage"],
               "response_file": resp_rel, "transcript_file": tr_rel}
        (run_dir / inv_rel).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        u = ex["usage"] or {}
        if infra:
            print(f"[{pos}/{len(pending)}] {row['key']} INFRA {infra}", flush=True)
            continue
        s = score(row["task"], text, cfg)
        failed = [k for k, v in s["checks"].items() if not v]
        print(f"[{pos}/{len(pending)}] {row['key']} {inv['gen_s']:.1f}s · 생각 {u.get('thoughtsTokenCount', 0)}tok"
              f" · 답 {u.get('candidatesTokenCount', 0)}tok · 통과={s['pass']} · 실패={','.join(failed) or '-'}", flush=True)
        if not s["pass"] and row["arm"] not in seen:  # 본문 캡처 예산(break ≤2) — 조건별 첫 실패 1장씩
            seen.add(row["arm"])
            d = s["detail"]
            why = ", ".join(CHECK_KO.get(k, k) for k in failed)
            if row["task"] == "summary":
                why += f" · {d['sentence_count']}문장 · 원문 밖 숫자 {', '.join(d['extra_numbers']) or '-'}"
            elif row["task"] == "translate":
                why += f" · 빠진 용어 {', '.join(d['missing_terms']) or '-'} · 남은 영어 {', '.join(d['stray_english'][:4]) or '-'}"
            else:
                bad = [c for c in d["cases"] if not c["ok"]]
                why += f" · {d.get('load_error') or ''} 실패 케이스 {len(bad)}/{len(d['cases'])}"
            live_mark("break", _clip(f"{TASK_KO[row['task']]} 첫 실패 — {ARM_KO[row['arm']]}/반복{row['rep']} · {why}"))

    write_run_yaml(run_dir)
    rows, agg = score_run_dir(run_dir, args.cases, write_pilot=(args.mode == "pilot"))
    write_run_yaml(run_dir)

    def part(arm: str) -> str:
        blk = agg["arms"].get(arm, {})
        return f"{ARM_KO[arm]} " + " ".join(f"{TASK_KO[t]} {b['pass']['hit']}/{b['pass']['total']}" for t, b in blk.items())

    complete = {r["key"] for r in plan} <= {r["key"] for r in rows}
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(rows)}회 집계 — {part('high')} · {part('low')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(f"{len(rows)}회 집계 — {part('high')} · {part('low')} · 인프라 {agg['infra_errors']}건"))
    status = {"mode": args.mode, "runs": len(rows), "infra_errors": agg["infra_errors"], "complete": complete,
              "pilot_decision": agg.get("pilot_decision")}
    (run_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.mode == "pilot" and not (agg.get("pilot_decision") or {}).get("proceed"):
        return 3
    return 0 if complete or args.mode == "pilot" else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"))
    ap.add_argument("--rescore", action="store_true", help="기존 raw 를 재채점(과금 없음)")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
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
