#!/usr/bin/env python3
"""블로그 글쓰기 과업 하네스(2026-09-23) — 무료 로컬 모델(gemma3:4b)에 블로그 글 조각을 맡긴다.

2026-07-10 첫 런(`tool_test_harness` WR-01)은 프롬프트 1개를 모델 4개에 1회씩 넣은 n=4 였다.
재제작은 모델을 gemma3:4b 하나로 두고 과업을 넓힌다(시네 승인 2026-09-23).
  - 글감 메모 10개(가상의 서비스) × 과업 3종(도입부 · 제목 5개 · 검색 설명문)
  - 조건 2종: 기본 지시 / 기본 지시 + 「메모에 없는 숫자는 쓰지 않습니다」 한 줄
  - 반복 3회 · 온도·seed 미지정(모델 기본 설정)
  - 채점 = `writing_score.py`(결정론). 과장어는 목록 기반이라는 한계를 본문에 공시한다.
  - 실촬영 마커: 메모 밖 숫자 첫 등장·형식 조건 첫 위반 = break · 조건/반복 전환 = turn · 집계 = agg.

★2026-09-23 첫 파일럿(…-20260923)은 채점기 수정 전 마커라 폐기하고 …-20260923b 로 다시 돌렸다.

usage:
  python3 run_writing_bench.py --mode pilot --run-dir test_runs/ollama-gemma3-4b-writing-20260923b
  python3 run_writing_bench.py --mode full  --run-dir test_runs/ollama-gemma3-4b-writing-20260923b
  python3 run_writing_bench.py --rescore     --run-dir test_runs/ollama-gemma3-4b-writing-20260923b
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
from writing_score import score, score_run_dir

HARNESS_VERSION = "1.0"
BASE = Path(__file__).resolve().parent
DEFAULT_CASES = BASE / "writing_bench_cases.json"
DEFAULT_RUN_DIR = BASE / "test_runs" / "ollama-gemma3-4b-writing-20260923b"
ARMS = ("base", "guard")
ARM_KO = {"base": "기본 지시", "guard": "숫자 금지 한 줄 추가"}
TASKS = ("intro", "titles", "meta")
REPS = 3
PILOT_MEMOS = ("M01", "M02")
CHECK_KO = {"keyword": "키워드 누락", "polite": "반말 문장", "length": "글자 수 범위 밖", "ends_question": "질문으로 안 끝남",
            "count": "제목 개수", "no_marks": "번호·기호", "single_paragraph": "두 문단 이상"}


def build_prompt(task: str, memo: dict[str, Any], cfg: dict[str, Any], arm: str) -> str:
    kw = memo["keyword"]
    t = cfg["tasks"][task]
    guard = ("\n" + cfg["guard_line"]) if arm == "guard" else ""
    memo_txt = f"서비스 이름: {memo['name']}\n" + "\n".join(f"- {f}" for f in memo["facts"])
    if task == "intro":
        head = "아래 글감 메모만 사용해 블로그 글의 도입부를 써 주세요."
        rules = (f"- 핵심 키워드 「{kw}」 — 한 번 이상 넣습니다.\n"
                 "- 한국어 존댓말로 씁니다.\n"
                 f"- 공백 포함 {t['min_chars']}~{t['max_chars']}자로 씁니다.\n"
                 "- 과장하거나 낚시하는 표현을 쓰지 않습니다.\n"
                 "- 마지막 문장은 독자에게 묻는 질문으로 끝냅니다.\n"
                 "- 도입부 본문만 출력합니다.")
    elif task == "titles":
        head = "아래 글감 메모로 블로그 글 제목 후보를 써 주세요."
        rules = (f"- 제목 {t['count']}개를 한 줄에 하나씩 씁니다.\n"
                 "- 번호나 기호 없이 제목만 씁니다.\n"
                 f"- 각 제목은 공백 포함 {t['max_chars']}자 이내입니다.\n"
                 f"- 모든 제목에 키워드 「{kw}」 — 빠짐없이 넣습니다.\n"
                 "- 과장하거나 낚시하는 표현을 쓰지 않습니다.")
    else:
        head = "아래 글감 메모로 검색 결과에 보일 블로그 글 설명문을 써 주세요."
        rules = ("- 한 문단으로 씁니다.\n"
                 f"- 공백 포함 {t['min_chars']}~{t['max_chars']}자로 씁니다.\n"
                 f"- 키워드 「{kw}」 — 한 번 이상 넣습니다.\n"
                 "- 한국어 존댓말로 씁니다.\n"
                 "- 과장하거나 낚시하는 표현을 쓰지 않습니다.\n"
                 "- 설명문만 출력합니다.")
    return f"{head}\n\n조건:\n{rules}{guard}\n\n[글감 메모]\n{memo_txt}\n"


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
            for task in TASKS:
                for memo in cfg["memos"]:
                    if mode == "pilot" and memo["id"] not in PILOT_MEMOS:
                        continue
                    runs.append({"key": f"r{rep}/{arm}/{task}/{memo['id']}", "rep": rep, "arm": arm,
                                 "task": task, "memo": memo})
    return runs


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys, maximum = set(), 0
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        keys.add(row["key"])
    # ★번호는 raw 의 모든 파일에서 복원한다 — 원출력만 쓰고 invocation 전에 죽은 회차를 덮어쓰지 않게
    for p in (run_dir / "raw").glob("*-*"):
        m = re.match(r"(\d+)-", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return keys, maximum


def _seen(run_dir: Path, cfg: dict[str, Any]) -> set[str]:
    memos = {m["id"]: m for m in cfg["memos"]}
    seen = set()
    for p in (run_dir / "raw").glob("*-invocation.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row.get("infra_error"):
            continue
        s = score(row["task"], (run_dir / row["response_file"]).read_text(encoding="utf-8"), memos[row["memo_id"]], cfg)
        if not s["checks"]["no_extra_numbers"]:
            seen.add("number")
        if not s["format_ok"]:
            seen.add("format")
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
        entries.append({"key": row["key"], "memo_id": row["memo_id"], "task": row["task"], "arm": row["arm"],
                        "rep": row["rep"], "mode": row["mode"], "output_file": row["transcript_file"],
                        "response_file": row["response_file"], "log_file": str(p.relative_to(run_dir)),
                        "gen_s": row.get("gen_s"), "infra_error": row.get("infra_error")})
    compare = []
    if (run_dir / "aggregate.json").exists():
        agg = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))
        for arm, block in agg.get("arms", {}).items():
            compare.append({"metric": "all_rules", "arm": arm, **block["all_rules"]})
            compare.append({"metric": "extra_numbers_runs", "arm": arm, **block["extra_numbers_runs"]})
    payload = {
        "tool": "ollama",
        "date": date or dt.date.today().isoformat(),
        "method": "WR-02",
        "access": "local",
        "model": model,
        "generated_by": "run_writing_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관) · 글감 메모=자작 가상 서비스",
        "tasks": list(TASKS), "arms": {k: ARM_KO[k] for k in ARMS}, "reps": REPS,
        "options": {"temperature": "미지정(모델 기본값)", "seed": "미지정", "num_ctx": "미지정(서버 기본값)"},
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
    # ★원 실행 기록은 보존한다 — 재채점이 바꾸는 것은 판정(compare·runs)뿐이다(T2 적대검증 2026-09-23).
    for key in ("environment", "model_metadata", "harness_version", "options"):
        if key in prev:
            runs[key] = prev[key]
    runs["rescored_at"] = dt.datetime.now().isoformat(timespec="seconds")
    runs["rescored_git_sha"] = _git_sha()
    (args.run_dir / "run.yaml").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({a: b["all_rules"] for a, b in agg["arms"].items()}, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.rescore:
        return rescore(args)
    cfg = json.loads(args.cases.read_text(encoding="utf-8"))
    plan = planned(cfg, args.mode)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "runs": len(plan), "first": plan[0]["key"],
                          "prompt": build_prompt(plan[0]["task"], plan[0]["memo"], cfg, plan[0]["arm"])},
                         ensure_ascii=False, indent=1))
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
                                  "modified_at": show.get("modified_at")}
    except Exception as exc:  # noqa: BLE001
        print(f"[메타] 조회 실패: {exc}")

    existing, next_id = _existing(run_dir)
    pending = [r for r in plan if r["key"] not in existing]
    print(f"mode={args.mode} planned={len(plan)} existing={len(plan) - len(pending)} running_now={len(pending)}")
    seen = _seen(run_dir, cfg)
    if pending:
        if args.mode == "pilot":
            live_mark("turn", _clip(f"파일럿 시작 — {args.model} · 글감 {len(PILOT_MEMOS)}개 × 과업 3종 × 조건 2종 · {len(pending)}회"))
        else:
            live_mark("turn", _clip(f"본측정으로 전환 — {args.model} · 글감 10개 × 과업 3종 × 조건 2종 × 반복 {REPS}회 · 기존 {len(plan) - len(pending)}회 포함 {len(pending)}회 추가"))
    prev = None
    for pos, row in enumerate(pending, 1):
        cur = (row["rep"], row["arm"])
        if prev is not None and cur != prev:
            live_mark("turn", _clip(f"조건 전환 — 반복 {prev[0]}·{ARM_KO[prev[1]]} → 반복 {cur[0]}·{ARM_KO[cur[1]]}"))
        prev = cur
        memo = row["memo"]
        prompt = build_prompt(row["task"], memo, cfg, row["arm"])
        resp, attempts, infra = generate(base, args.model, prompt, args.timeout)
        text = (resp or {}).get("response", "") if isinstance((resp or {}).get("response", ""), str) else ""
        if resp is not None and not text.strip() and not infra:
            infra = "empty_response"  # 빈 응답은 측정 실패다 — 형식 실패로 채점하지 않는다
        if attempts and (infra or any(a["error"] for a in attempts)):
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · {infra or attempts[0]['error']} · 시도 {len(attempts)}/2"))
        elif resp is not None and not text.strip():
            live_mark("infra", _clip(f"측정 흔들림 — {row['key']} · 응답은 왔지만 빈 텍스트"))
        next_id += 1
        stem = f"{next_id:03d}"
        resp_rel, tr_rel, inv_rel = f"raw/{stem}-response.txt", f"raw/{stem}-output.txt", f"raw/{stem}-invocation.json"
        (run_dir / resp_rel).write_text(text, encoding="utf-8")
        (run_dir / tr_rel).write_text(
            f"run_id: {next_id}\nkey: {row['key']}\nmodel: {args.model}\n\n[PROMPT]\n{prompt}\n[RAW RESPONSE]\n{text}\n"
            f"\n[API METADATA]\n{json.dumps({k: v for k, v in (resp or {}).items() if k != 'context'}, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8")
        metrics = {k: (resp or {}).get(k) for k in ("done_reason", "total_duration", "load_duration",
                                                    "prompt_eval_count", "eval_count", "eval_duration")} if resp else None
        inv = {"run_id": next_id, "key": row["key"], "mode": args.mode, "memo_id": memo["id"], "task": row["task"],
               "arm": row["arm"], "rep": row["rep"], "prompt": prompt,
               "payload": {"model": args.model, "stream": False}, "attempts": attempts,
               "gen_s": round(sum(a["elapsed_s"] for a in attempts), 3), "infra_error": infra,
               "response_file": resp_rel, "transcript_file": tr_rel, "api_metrics": metrics}
        (run_dir / inv_rel).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
        if infra:
            print(f"[{pos}/{len(pending)}] {row['key']} INFRA {infra}", flush=True)
            continue
        s = score(row["task"], text, memo, cfg)
        failed = [k for k, v in s["checks"].items() if not v]
        print(f"[{pos}/{len(pending)}] {row['key']} {inv['gen_s']:.1f}s · 전체통과={s['all_rules']}"
              f" · 실패={','.join(failed) or '-'} · 메모밖숫자={s['detail']['extra_numbers'] or '-'}", flush=True)
        if not s["checks"]["no_extra_numbers"] and "number" not in seen:
            seen.add("number")
            live_mark("break", _clip(
                f"메모 밖 숫자 첫 등장 — {memo['id']}/{cfg['tasks'][row['task']]['name']}/{ARM_KO[row['arm']]}/반복{row['rep']}"
                f" · 메모에 없는 숫자 {', '.join(s['detail']['extra_numbers'])} · 「{memo['keyword']}」"))
        if not s["format_ok"] and "format" not in seen:
            seen.add("format")
            live_mark("break", _clip(
                f"형식 조건 첫 위반 — {memo['id']}/{cfg['tasks'][row['task']]['name']}/{ARM_KO[row['arm']]}/반복{row['rep']}"
                f" · 어긴 조건 {', '.join(CHECK_KO.get(k, k) for k in failed if k != 'no_extra_numbers')}"))

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
        return (f"{ARM_KO[a]} 전체통과 {b['all_rules']['hit']}/{b['all_rules']['total']}"
                f"·메모 밖 숫자 {b['extra_numbers_runs']['hit']}회")

    complete = {r["key"] for r in plan} <= {r["key"] for r in rows}
    if args.mode == "pilot":
        d = agg.get("pilot_decision") or {}
        live_mark("agg", _clip(f"파일럿 {len(rows)}회 집계 — {part('base')} · {part('guard')} · "
                               + ("본측정 진행" if d.get("proceed") else f"중단: {d.get('reasons')}")))
    elif complete:
        live_mark("agg", _clip(f"{len(rows)}회 집계 — {part('base')} · {part('guard')} · 인프라 {agg['infra_errors']}건"))
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
    ap.add_argument("--rescore", action="store_true", help="기존 raw 를 재채점(GPU 불필요)")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.mode and not args.rescore:
        ap.error("--mode 또는 --rescore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
