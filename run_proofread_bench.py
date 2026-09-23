#!/usr/bin/env python3
"""맞춤법·문장 교정 하네스(2026-09-23) — 무료 로컬 모델(qwen3-vl:8b)이 심어 둔 오류를 고치는지 잰다.

2026-07-21 첫 런(`tool_test_harness` TXT-03)은 지문 1개를 「다듬어줘」 한 지시로 4회 넣은 것이었다.
재제작은 오류 자리를 정해 두고 센다(시네 승인 2026-09-23 · 표준 규모 · 생각 모드 기본값).
  - 초안 10개(`proofread_bench_cases.json`) × 초안마다 표준 맞춤법 오류 5개 = 50자리 · D01 = 첫 런 지문 그대로
  - 조건 2종: 맞춤법만 고치기 / 자연스럽게 다듬기(첫 런 지시문 그대로)
  - 반복 3회 · 온도·seed·think 미지정(모델 기본값 — 생각 모드 켜짐)
  - 채점 = `proofread_score.py`(결정론) — fixed / missed / rewritten · 숫자·이름 보존
  - 기록 = 답 본문(response)과 생각(thinking)을 따로 저장 · done_reason·eval_count 보존
  - 실촬영 마커: 심은 오류 첫 방치·보존 표지 첫 소실 = break · 조건/반복 전환 = turn · 집계 = agg.

usage:
  python3 run_proofread_bench.py --mode pilot --run-dir test_runs/ollama-qwen3-vl-8b-proofread-20260923
  python3 run_proofread_bench.py --mode full  --run-dir test_runs/ollama-qwen3-vl-8b-proofread-20260923
  python3 run_proofread_bench.py --rescore     --run-dir test_runs/ollama-qwen3-vl-8b-proofread-20260923
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
from proofread_score import score, score_run_dir

HARNESS_VERSION = "1.0"
BASE = Path(__file__).resolve().parent
DEFAULT_CASES = BASE / "proofread_bench_cases.json"
DEFAULT_RUN_DIR = BASE / "test_runs" / "ollama-qwen3-vl-8b-proofread-20260923"
ARMS = ("spell", "polish")
ARM_KO = {"spell": "맞춤법만 고치기", "polish": "자연스럽게 다듬기"}
REPS = 3
PILOT_QIDS = ("D01", "D02")


def build_prompt(q: dict[str, Any], cfg: dict[str, Any], arm: str) -> str:
    # 첫 런 형식 그대로 — 지시 한 줄, 빈 줄, 초안
    return f"{cfg['arms'][arm]}\n\n{q['text']}"


def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(base: str, model: str, prompt: str, timeout: int, num_ctx: int | None = None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
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
            for q in cfg["drafts"]:
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
    qs = {q["id"]: q for q in cfg["drafts"]}
    seen = set()
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("infra_error"):
            continue
        s = score(qs[row["did"]], (run_dir / row["response_file"]).read_text(encoding="utf-8"))
        if s["missed"]:
            seen.add("missed")
        if s["anchors_lost"]:
            seen.add("anchor")
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
        entries.append({"key": row["key"], "did": row["did"], "arm": row["arm"], "rep": row["rep"], "mode": row["mode"],
                        "output_file": row["transcript_file"], "response_file": row["response_file"],
                        "log_file": str(p.relative_to(run_dir)), "gen_s": row.get("gen_s"),
                        "infra_error": row.get("infra_error")})
    compare = []
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))
        for arm, block in agg.get("arms", {}).items():
            for metric in ("errors_fixed", "errors_missed", "drafts_anchors_kept"):
                compare.append({"metric": metric, "arm": arm, **block[metric]})
    payload = {
        "tool": "ollama",
        "date": date or dt.date.today().isoformat(),
        "method": "TXT-03R",
        "access": "local",
        "model": model,
        "generated_by": "run_proofread_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관) · 초안=자작(D01 은 첫 런 지문)",
        "arms": {k: ARM_KO[k] for k in ARMS}, "reps": REPS,
        "options": {"temperature": "미지정(모델 기본값)", "seed": "미지정", "think": "미지정(모델 기본값)",
                    "num_ctx": meta.get("num_ctx") or "미지정(서버 기본값)", "num_predict": "미지정(서버 기본값)"},
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
    print(json.dumps({a: {k: b[k] for k in ("errors_fixed", "errors_missed", "drafts_anchors_kept")}
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
                          "prompt_polish": build_prompt(plan[0]["q"], cfg, "polish")}, ensure_ascii=False, indent=1))
        return 0
    run_dir: Path = args.run_dir
    if args.mode == "full":
        dp = run_dir / "pilot_decision.json"
        if not dp.exists() or not json.loads(dp.read_text(encoding="utf-8")).get("proceed"):
            print("파일럿 판정 없음/중단 — 같은 --run-dir 에서 --mode pilot 을 먼저 통과하세요.")
            return 3
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    meta = {"ollama_version": None, "hardware": _gpu_name(), "model_metadata": None, "num_ctx": args.num_ctx}
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
    n_draft = len(cfg["drafts"])
    if pending:
        if args.mode == "pilot":
            live_mark("turn", _clip(f"파일럿 시작 — {args.model} · 초안 {len(PILOT_QIDS)}개 × 조건 2종 · {len(pending)}회"))
        else:
            live_mark("turn", _clip(f"본측정으로 전환 — {args.model} · 초안 {n_draft}개(오류 {n_draft * 5}자리) × 조건 2종 × 반복 {REPS}회"
                                    f" · 기존 {len(plan) - len(pending)}회 포함 {len(pending)}회 추가"))
    prev = None
    for pos, row in enumerate(pending, 1):
        cur = (row["rep"], row["arm"])
        if prev is not None and cur != prev:
            live_mark("turn", _clip(f"조건 전환 — 반복 {prev[0]}·{ARM_KO[prev[1]]} → 반복 {cur[0]}·{ARM_KO[cur[1]]}"))
        prev = cur
        q = row["q"]
        prompt = build_prompt(q, cfg, row["arm"])
        resp, attempts, infra = generate(base, args.model, prompt, args.timeout, args.num_ctx)
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
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "did": q["id"],
               "arm": row["arm"], "rep": row["rep"], "prompt": prompt,
               "payload": {"model": args.model, "stream": False,
                           **({"options": {"num_ctx": args.num_ctx}} if args.num_ctx else {})}, "attempts": attempts,
               "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "response_file": resp_rel, "thinking_file": think_rel, "thinking_chars": len(thinking),
               "transcript_file": tr_rel, "api_metrics": metrics}
        (run_dir / inv_rel).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        if infra:
            print(f"[{pos}/{len(pending)}] {row['key']} INFRA {infra}", flush=True)
            continue
        s = score(q, text)
        print(f"[{pos}/{len(pending)}] {row['key']} {inv['gen_s']:.1f}s · 고침 {s['fixed']}/5 · 남김 {s['missed']}"
              f" · 바꿔씀 {s['rewritten']} · 표지소실 {','.join(s['anchors_lost']) or '-'} · 길이비 {s['len_ratio']}"
              f" · 생각 {len(thinking)}자 · done={metrics.get('done_reason')}", flush=True)
        if s["missed"] and "missed" not in seen:
            seen.add("missed")
            left = ", ".join(f"{e['wrong']}(→{e['right']})" for e in s["errors"] if e["verdict"] == "missed")
            live_mark("break", _clip(f"심은 오류 첫 방치 — {q['id']}/{ARM_KO[row['arm']]}/반복{row['rep']} · 그대로 남음 {left}"))
        if s["anchors_lost"] and "anchor" not in seen:
            seen.add("anchor")
            live_mark("break", _clip(f"보존 표지 첫 소실 — {q['id']}/{ARM_KO[row['arm']]}/반복{row['rep']}"
                                     f" · 사라진 것 {', '.join(s['anchors_lost'])}"))

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
        return (f"{ARM_KO[a]} 오류 고침 {b['errors_fixed']['hit']}/{b['errors_fixed']['total']}"
                f"·남김 {b['errors_missed']['hit']}·표지 보존 {b['drafts_anchors_kept']['hit']}/{b['drafts_anchors_kept']['total']}")

    complete = {r["key"] for r in plan} <= {r["key"] for r in rows}
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(rows)}회 집계 — {part('spell')} · {part('polish')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(f"{len(rows)}회 집계 — {part('spell')} · {part('polish')} · 인프라 {agg['infra_errors']}건"))
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
    # 2026-09-23 파일럿: 서버 기본 문맥(4,096)에서 생각 토큰이 문맥을 다 써 4회 중 3회 빈 답 → 시네 승인으로 문맥만 늘린다
    ap.add_argument("--num-ctx", type=int, default=None, help="Ollama options.num_ctx(생략 = 서버 기본값)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.mode and not args.rescore:
        ap.error("--mode 또는 --rescore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
