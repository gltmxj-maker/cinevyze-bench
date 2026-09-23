#!/usr/bin/env python3
"""한국 상식·거짓 전제 하네스(2026-09-23) — 무료 로컬 모델(qwen3-vl:8b)이 모르는 것을 지어내는지 잰다.

2026-07-21 첫 런(`tool_test_harness` TXT-04)은 질문 5개를 한 프롬프트에 묶어 4회 넣은 것이었다.
재제작은 질문을 넓히고 하나씩 따로 묻는다(시네 승인 2026-09-23 · 표준 규모 · 생각 모드 기본값).
  - 질문 24개(`facts_bench_cases.json`): 사실 12(정답이 정해진 한국 상식) + 거짓 전제 함정 12
  - 조건 2종: 기본(질문만) / 정직 지시 한 줄 추가(옛 런 지시문 그대로)
  - 반복 3회 · 온도·seed·think 미지정(모델 기본값 — 생각 모드 켜짐)
  - 채점 = `facts_score.py`(결정론 1차) + 함정 응답 전수 원문 대조(`trap_audit.json`)
  - 기록 = 답 본문(response)과 생각(thinking)을 따로 저장 · done_reason·eval_count 보존
    (생각 토큰이 상한을 먹어 답이 잘리면 오답이 아니라 truncated 로 센다)
  - 실촬영 마커: 함정 첫 지어냄·사실 첫 오답 = break · 조건/반복 전환 = turn · 집계 = agg.

usage:
  python3 run_facts_bench.py --mode pilot --run-dir test_runs/ollama-qwen3-vl-8b-facts-20260923
  python3 run_facts_bench.py --mode full  --run-dir test_runs/ollama-qwen3-vl-8b-facts-20260923
  python3 run_facts_bench.py --rescore     --run-dir test_runs/ollama-qwen3-vl-8b-facts-20260923
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from live_mark import mark as live_mark
from facts_score import score, score_run_dir

HARNESS_VERSION = "1.0"
BASE = Path(__file__).resolve().parent
DEFAULT_CASES = BASE / "facts_bench_cases.json"
DEFAULT_RUN_DIR = BASE / "test_runs" / "ollama-qwen3-vl-8b-facts-20260923"
ARMS = ("base", "honest")
ARM_KO = {"base": "기본(질문만)", "honest": "정직 지시 한 줄 추가"}
REPS = 3
PILOT_QIDS = ("F01", "F03", "T01", "T02")


def build_prompt(q: dict[str, Any], cfg: dict[str, Any], arm: str) -> str:
    head = "다음 질문에 한국어로 답해 주세요."
    if arm == "honest":
        head += "\n" + cfg["honest_line"]
    return f"{head}\n\n질문: {q['q']}\n"


def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(base: str, model: str, prompt: str, timeout: int):
    payload = {"model": model, "prompt": prompt, "stream": False}
    attempts = []
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            resp = api_json(base + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": None})
            return resp, attempts, None
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": err})
            if attempt == 2:
                return None, attempts, err
            time.sleep(1)
    return None, attempts, attempts[-1]["error"]


def planned(cfg: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    reps = (1,) if mode == "pilot" else tuple(range(1, REPS + 1))
    runs = []
    for rep in reps:
        for arm in ARMS:
            for q in cfg["questions"]:
                if mode == "pilot" and q["id"] not in PILOT_QIDS:
                    continue
                runs.append({"key": f"r{rep}/{arm}/{q['id']}", "rep": rep, "arm": arm, "q": q})
    return runs


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for p in (run_dir / "raw").glob("*-invocation.json"):
        keys.add(json.loads(p.read_text(encoding="utf-8"))["key"])
    # ★번호는 raw 의 모든 파일에서 복원한다 — 원출력만 쓰고 invocation 전에 죽은 회차를 덮어쓰지 않게
    for p in (run_dir / "raw").glob("*-*"):
        m = re.match(r"(\d+)-", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return keys, maximum


def _seen(run_dir: Path, cfg: dict[str, Any]) -> set[str]:
    qs = {q["id"]: q for q in cfg["questions"]}
    seen = set()
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("infra_error"):
            continue
        s = score(qs[row["qid"]], (run_dir / row["response_file"]).read_text(encoding="utf-8"))
        if s["verdict"] == "fabricated":
            seen.add("fabricated")
        if s["verdict"] == "wrong":
            seen.add("wrong")
    return seen


def _clip(text: str, limit: int = 185) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_name() -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       text=True, timeout=10).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


def write_run_yaml(run_dir: Path, model: str, meta: dict[str, Any], date: str | None = None) -> None:
    entries = []
    for p in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        entries.append({"key": row["key"], "qid": row["qid"], "arm": row["arm"], "rep": row["rep"], "mode": row["mode"],
                        "output_file": row["transcript_file"], "response_file": row["response_file"],
                        "log_file": str(p.relative_to(run_dir)), "gen_s": row.get("gen_s"),
                        "infra_error": row.get("infra_error")})
    compare = []
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))
        for arm, block in agg.get("arms", {}).items():
            for metric in ("fact_correct", "trap_flagged_auto", "trap_flagged_audit"):
                compare.append({"metric": metric, "arm": arm, **block[metric]})
    payload = {
        "tool": "ollama",
        "date": date or dt.date.today().isoformat(),
        "method": "TXT-04R",
        "access": "local",
        "model": model,
        "generated_by": "run_facts_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관) · 문항=자작(사실 문항 출처는 문항 파일 source)",
        "arms": {k: ARM_KO[k] for k in ARMS}, "reps": REPS,
        "options": {"temperature": "미지정(모델 기본값)", "seed": "미지정", "think": "미지정(모델 기본값)",
                    "num_ctx": "미지정(서버 기본값)", "num_predict": "미지정(서버 기본값)"},
        "request_parallelism": 1,
        "compare": compare,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "git_sha_at_run": _git_sha(), "ollama_version": meta.get("ollama_version"),
                        "hardware": meta.get("hardware")},
        "model_metadata": meta.get("model_metadata"),
        "runs": entries,
    }
    tmp = run_dir / "run.yaml.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "run.yaml")


def rescore(args: argparse.Namespace) -> int:
    prev = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
    meta = {"ollama_version": prev["environment"].get("ollama_version"),
            "hardware": prev["environment"].get("hardware"), "model_metadata": prev.get("model_metadata")}
    _, agg = score_run_dir(args.run_dir, args.cases, write_pilot=False)
    write_run_yaml(args.run_dir, prev["model"], meta, date=prev["date"])
    runs = json.loads((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
    # ★원 실행 기록은 보존한다 — 재채점이 바꾸는 것은 판정(compare·runs)뿐이다.
    for key in ("environment", "model_metadata", "harness_version", "options"):
        if key in prev:
            runs[key] = prev[key]
    runs["rescored_at"] = dt.datetime.now().isoformat(timespec="seconds")
    runs["rescored_git_sha"] = _git_sha()
    (args.run_dir / "run.yaml").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({a: {k: b[k] for k in ("fact_correct", "trap_flagged_auto", "trap_flagged_audit")}
                      for a, b in agg["arms"].items()}, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.rescore:
        return rescore(args)
    cfg = json.loads(args.cases.read_text(encoding="utf-8"))
    plan = planned(cfg, args.mode)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": plan[0]["key"],
                          "prompt": build_prompt(plan[0]["q"], cfg, plan[0]["arm"]),
                          "prompt_honest": build_prompt(plan[0]["q"], cfg, "honest")}, ensure_ascii=False, indent=1))
        return 0
    run_dir: Path = args.run_dir
    if args.mode == "full":
        dp = run_dir / "pilot_decision.json"
        if not dp.exists() or not json.loads(dp.read_text(encoding="utf-8")).get("proceed"):
            print("파일럿 판정 없음/중단 — 같은 --run-dir 에서 --mode pilot 을 먼저 통과하세요.")
            return 3
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    meta = {"ollama_version": None, "hardware": _gpu_name(), "model_metadata": None}
    try:
        meta["ollama_version"] = api_get(base + "/api/version", args.timeout).get("version")
        show = api_json(base + "/api/show", {"model": args.model}, args.timeout)
        meta["model_metadata"] = {"details": show.get("details"), "parameters": show.get("parameters"),
                                  "capabilities": show.get("capabilities"), "modified_at": show.get("modified_at")}
    except Exception as exc:  # noqa: BLE001
        print(f"[메타] 조회 실패: {exc}")

    existing, next_id = _existing(run_dir)
    pending = [r for r in plan if r["key"] not in existing]
    print(f"mode={args.mode} planned={len(plan)} existing={len(plan) - len(pending)} running_now={len(pending)}")
    seen = _seen(run_dir, cfg)
    n_fact = sum(1 for q in cfg["questions"] if q["type"] == "fact")
    n_trap = len(cfg["questions"]) - n_fact
    if pending:
        if args.mode == "pilot":
            live_mark("turn", _clip(f"파일럿 시작 — {args.model} · 질문 {len(PILOT_QIDS)}개 × 조건 2종 · {len(pending)}회"))
        else:
            live_mark("turn", _clip(f"본측정으로 전환 — {args.model} · 사실 {n_fact} + 함정 {n_trap} × 조건 2종 × 반복 {REPS}회"
                                    f" · 기존 {len(plan) - len(pending)}회 포함 {len(pending)}회 추가"))
    prev = None
    for pos, row in enumerate(pending, 1):
        cur = (row["rep"], row["arm"])
        if prev is not None and cur != prev:
            live_mark("turn", _clip(f"조건 전환 — 반복 {prev[0]}·{ARM_KO[prev[1]]} → 반복 {cur[0]}·{ARM_KO[cur[1]]}"))
        prev = cur
        q = row["q"]
        prompt = build_prompt(q, cfg, row["arm"])
        resp, attempts, infra = generate(base, args.model, prompt, args.timeout)
        text = (resp or {}).get("response", "") if isinstance((resp or {}).get("response", ""), str) else ""
        thinking = (resp or {}).get("thinking", "") if isinstance((resp or {}).get("thinking", ""), str) else ""
        if resp is not None and not text.strip() and not infra:
            # 빈 답은 측정 실패다. 생각이 상한을 먹었으면(length) 그 사실을 남긴다.
            infra = "empty_response" + ("(done_reason=length)" if resp.get("done_reason") == "length" else "")
        if attempts and (infra or any(a["error"] for a in attempts)):
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · {infra or attempts[0]['error']} · 시도 {len(attempts)}/2"))
        next_id += 1
        stem = f"{next_id:03d}"
        resp_rel, think_rel = f"raw/{stem}-response.txt", f"raw/{stem}-thinking.txt"
        tr_rel, inv_rel = f"raw/{stem}-output.txt", f"raw/{stem}-invocation.json"
        (run_dir / resp_rel).write_text(text, encoding="utf-8")
        (run_dir / think_rel).write_text(thinking, encoding="utf-8")
        (run_dir / tr_rel).write_text(
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {args.model}\n\n[PROMPT]\n{prompt}\n[THINKING]\n{thinking}\n"
            f"\n[RAW RESPONSE]\n{text}\n\n[API METADATA]\n"
            f"{json.dumps({k: v for k, v in (resp or {}).items() if k not in ('context', 'response', 'thinking')}, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8")
        metrics = {k: (resp or {}).get(k) for k in ("done_reason", "total_duration", "load_duration",
                                                    "prompt_eval_count", "eval_count", "eval_duration")} if resp else None
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "qid": q["id"], "qtype": q["type"],
               "arm": row["arm"], "rep": row["rep"], "prompt": prompt,
               "payload": {"model": args.model, "stream": False}, "attempts": attempts,
               "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "response_file": resp_rel, "thinking_file": think_rel, "thinking_chars": len(thinking),
               "transcript_file": tr_rel, "api_metrics": metrics}
        (run_dir / inv_rel).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        if infra:
            print(f"[{pos}/{len(pending)}] {row['key']} INFRA {infra}", flush=True)
            continue
        s = score(q, text)
        print(f"[{pos}/{len(pending)}] {row['key']} {inv['gen_s']:.1f}s · {q['type']} → {s['verdict']}"
              f" · 생각 {len(thinking)}자 · done={metrics.get('done_reason')}", flush=True)
        if s["verdict"] == "fabricated" and "fabricated" not in seen:
            seen.add("fabricated")
            live_mark("break", _clip(f"거짓 전제를 받아 답함(자동 1차) — {q['id']}/{ARM_KO[row['arm']]}/반복{row['rep']}"
                                     f" · 「{q['q'][:40]}」 → 「{' '.join(text.split())[:60]}」"))
        if s["verdict"] == "wrong" and "wrong" not in seen:
            seen.add("wrong")
            live_mark("break", _clip(f"사실 질문 첫 오답 — {q['id']}/{ARM_KO[row['arm']]}/반복{row['rep']}"
                                     f" · 「{q['q'][:40]}」 → 「{' '.join(text.split())[:60]}」"))

    try:
        api_json(base + "/api/generate", {"model": args.model, "keep_alive": 0}, args.timeout)
    except Exception:  # noqa: BLE001
        pass
    write_run_yaml(run_dir, args.model, meta)
    rows, agg = score_run_dir(run_dir, args.cases, write_pilot=(args.mode == "pilot"))
    write_run_yaml(run_dir, args.model, meta)
    arms = agg["arms"]

    def part(a: str) -> str:
        b = arms[a]
        return (f"{ARM_KO[a]} 사실 정답 {b['fact_correct']['hit']}/{b['fact_correct']['total']}"
                f"·함정 교정(자동) {b['trap_flagged_auto']['hit']}/{b['trap_flagged_auto']['total']}")

    complete = {r["key"] for r in plan} <= {r["key"] for r in rows}
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(rows)}회 집계 — {part('base')} · {part('honest')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(f"{len(rows)}회 집계 — {part('base')} · {part('honest')} · 인프라 {agg['infra_errors']}건"))
    status = {"mode": args.mode, "runs": len(rows), "infra_errors": agg["infra_errors"], "truncated": agg["truncated"],
              "complete": complete, "pilot_decision": agg.get("pilot_decision")}
    (run_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.mode == "pilot" and not (agg.get("pilot_decision") or {}).get("proceed"):
        return 3
    return 0 if complete or args.mode == "pilot" else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"))
    ap.add_argument("--rescore", action="store_true", help="기존 raw 를 재채점(GPU 불필요)")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.mode and not args.rescore:
        ap.error("--mode 또는 --rescore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
