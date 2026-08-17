#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_repro.py — Lane A 실측 드라이버: 같은 모델·같은 프롬프트에 **디코딩 설정만** 바꿔 재현성 측정.

★축 분리(기존 실측과 겹치지 않는 이유):
  07-20 = 모델 *크기* 축 · 07-21 = *모델 간* 축 · 07-26(quant) = *양자화* 축 ·
  07-26(PRM) = *지시문* 축 · **이 드라이버 = 디코딩 무작위성 축**(모델·프롬프트 고정).

조건(TESTSET.md §3-C):
  A 기본값       = options 미전달(모델 Modelfile 값 그대로)
  B temp0        = {"temperature": 0}
  C temp0+seed   = {"temperature": 0, "seed": 42}

역할 분담(위조 방지 · run_quant.py 선례):
  - run.yaml 은 `tool_test_harness` 가 **직접** 쓴다(generated_by=tool_test_harness = 문자 그대로 사실).
    이 드라이버는 run.yaml 을 손대지 않는다 — 오케스트레이션 + 인덱스 기록만.
  - 채점은 `repro_score.py`(결정론). 드라이버는 점수를 만들지 않는다.

§5-A 자원 규율(하드):
  - 모델 한 번에 하나(OLLAMA_MAX_LOADED_MODELS=1 · NUM_PARALLEL=1) · 상주 금지(OLLAMA_KEEP_ALIVE=0).
  - 신규 다운로드 0 — 기설치 모델만(미설치면 어댑터가 거부). 홈 디스크 보호.
  - 끝나면 unload + VRAM 확인(예외가 나도 반납 — unload는 finally).

사용:
  /usr/bin/python3 run_repro.py                      # 기본 계획(gemma3:4b 3태스크 + qwen3-vl:8b 교차 1태스크)
  /usr/bin/python3 run_repro.py --repeats 3          # n수 축소(리허설)
  /usr/bin/python3 run_repro.py --dry-run            # 계획·프롬프트만 출력(호출 0)
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

# ── 환경은 어댑터 import 전에 박는다(어댑터가 os.environ을 읽어 서버를 띄운다) ──
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
os.environ["OLLAMA_NUM_PARALLEL"] = "1"
os.environ["OLLAMA_KEEP_ALIVE"] = "0"          # ★VRAM 상주 금지 → 매 호출 콜드 로딩(본문에 정직 표기)

import tool_adapters                            # noqa: E402
import tool_test_harness as harness             # noqa: E402
import testset                                  # noqa: E402
import bench_config as cfg                    # noqa: E402

OLLAMA_BIN = os.path.expanduser("~/.local/ollama/bin/ollama")

# 조건 = 이 드라이버의 유일한 독립변수. 순서 고정(A→B→C)이 run 인덱스 해석의 SSOT.
CONDITIONS = [
    {"key": "A", "label": "기본값",      "options": {}},
    {"key": "B", "label": "temp0",       "options": {"temperature": 0}},
    {"key": "C", "label": "temp0+seed",  "options": {"temperature": 0, "seed": 42}},
]

# 계획: (tool_spec, [task_id...]) — 주력 1종 전 태스크 + 교차확인 1종 1태스크("이 모델만의 성질인가")
PLAN = [
    ("ollama:gemma3:4b",  ["REP-01", "REP-02", "REP-03"]),
    ("ollama:qwen3-vl:8b", ["REP-01"]),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_used_mib():
    """GPU0 사용량(MiB). 실패는 None(예외 전파 금지 — 측정 실패를 0으로 위장하지 않는다)."""
    try:
        r = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10)
        if r.returncode != 0:
            return None
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def unload(model):
    """모델 VRAM 반납. (ok, 잔존모델문자열)."""
    sh([OLLAMA_BIN, "stop", model], timeout=60)
    time.sleep(2)
    r = sh([OLLAMA_BIN, "ps"], timeout=30)
    ps = (r.stdout or "").strip()
    return (model not in ps), ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8, help="조건당 반복 횟수(기본 8)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="tool_spec 하나만 실행(예: ollama:qwen3-vl:8b)")
    a = ap.parse_args()
    plan = [p for p in PLAN if (a.only is None or p[0] == a.only)]
    if not plan:
        print(f"--only '{a.only}' 가 PLAN에 없다: {[p[0] for p in PLAN]}", file=sys.stderr)
        return 2

    date_str = datetime.date.today().isoformat()
    print(f"[run_repro] {date_str} · repeats={a.repeats} · 조건 {len(CONDITIONS)}종")
    total_planned = sum(len(t) for _, t in plan) * len(CONDITIONS) * a.repeats
    print(f"[run_repro] 계획 런 수 = {total_planned}")

    if a.dry_run:
        for tid in sorted({t for _, ts in PLAN for t in ts}):
            p = testset.build_prompt(tid)
            print(f"\n───── {tid} ({testset.TASKS[tid]['category']}) · {len(p)}자\n{p}")
        for c in CONDITIONS:
            print(f"\n[조건 {c['key']}] {c['label']} · options={c['options'] or '(미전달)'}")
        return 0

    out = os.path.join("test_runs", f"repro-index-{date_str.replace('-', '')}.json")
    # ★머지: --only 로 한 레그만 다시 돌려도 앞 레그 기록이 날아가지 않게(같은 tool 항목만 교체).
    index = {"date": date_str, "repeats": a.repeats,
             "conditions": CONDITIONS, "runs_dirs": [], "gpu_baseline_mib": gpu_used_mib()}
    unload_failed = False
    if os.path.exists(out):
        try:
            with open(out, encoding="utf-8") as f:
                prev = json.load(f)
            keep = {p[0] for p in plan}
            index["runs_dirs"] = [e for e in (prev.get("runs_dirs") or [])
                                  if isinstance(e, dict) and e.get("tool") not in keep]
        except Exception as e:
            # ★손상된 인덱스를 조용히 덮어쓰면 앞 레그 기록이 증발한다 — 원본을 옆에 남기고 간다.
            bak = out + ".corrupt"
            try:
                os.replace(out, bak)
                print(f"  ⚠ 기존 인덱스 읽기 실패({e}) — 원본을 {bak} 로 보존하고 새로 쓴다")
            except OSError:
                print(f"  ⚠ 기존 인덱스 읽기 실패({e}) · 백업도 실패 — 새로 쓴다")

    for tool_spec, task_ids in plan:
        adapter, model = tool_adapters.resolve(tool_spec)
        if not adapter.available():
            print(f"  ⚠ {tool_spec} 미가용 — 건너뜀(안 돌린 걸 돌린 척 금지)")
            index["runs_dirs"].append({"tool": tool_spec, "skipped": "adapter unavailable"})
            continue

        # ★조건을 못 받는 어댑터에 조건을 주면 A/B/C가 전부 기본값으로 돌면서 성공으로 기록된다
        #   = 안 바꾼 걸 바꾼 척(측정 위조). 조용히 넘어가지 않는다(T2 적대검증 2026-07-29).
        if not hasattr(adapter, "extra_options"):
            raise tool_adapters.AdapterError(
                f"{tool_spec} 어댑터가 extra_options(디코딩 옵션 전달)를 지원하지 않는다 — "
                "조건 A/B/C가 전부 기본값으로 돌아 '조건을 바꾼 척'이 된다. 어댑터부터 지원할 것.")
        slug = f"{adapter.name}-{cfg.safe_name(model)}-repro"
        run_dir = harness._run_dir(slug, date_str)
        print(f"\n══ {tool_spec} → {run_dir}")

        # ★`or 0` 은 측정 실패(None)를 0 MiB 로 위장한다 — gpu_used_mib 가 굳이 None 을 주는 이유를 무효화.
        runs, idx, failures = [], 0, []
        peak = gpu_used_mib()
        try:
            for tid in task_ids:
                prompt = testset.build_prompt(tid)
                for cond in CONDITIONS:
                    for rep in range(1, a.repeats + 1):
                        # ★어댑터는 프로세스 전역 공유 인스턴스다 — 조건은 '호출 직전마다' 명시한다.
                        #   조건 루프 밖에서 한 번만 세팅하면 중간에 누가 건드렸을 때 조용히 섞인다.
                        adapter.extra_options = dict(cond["options"])
                        idx += 1
                        t0 = time.monotonic()
                        try:
                            r = harness.run_task(adapter, model, f"{tid}/{cond['key']}", prompt,
                                                 timeout=300, run_dir=run_dir, idx=idx)
                        except tool_adapters.AdapterError as e:
                            # ★실패도 데이터다(thinking 모델은 가시 응답이 비어 오는 회차가 있다).
                            #   조용한 재시도로 덮지 않는다 — 회차를 버리고 사유를 인덱스에 남긴다.
                            #   run.yaml 에는 안 넣는다(출력 파일이 없는 run = 증거 불변식 위반).
                            idx -= 1
                            failures.append({"task": f"{tid}/{cond['key']}", "repeat": rep,
                                             "error": str(e)})
                            print(f"  [!!] {tid}/{cond['key']} #{rep} 실패 — {e}")
                            continue
                        # 조건·회차를 run 레코드에 박는다(채점기가 이걸로 셀을 묶는다)
                        r["condition"] = cond["key"]
                        r["condition_label"] = cond["label"]
                        r["options_sent"] = (adapter.last_meta or {}).get("options_sent")
                        r["repeat"] = rep
                        r["eval_count"] = (adapter.last_meta or {}).get("eval_count")
                        r["eval_duration_ns"] = (adapter.last_meta or {}).get("eval_duration_ns")
                        r["load_duration_ns"] = (adapter.last_meta or {}).get("load_duration_ns")
                        runs.append(r)
                        g = gpu_used_mib()
                        if g is not None and (peak is None or g > peak):
                            peak = g
                        print(f"  [{idx:03d}] {tid}/{cond['key']} #{rep} "
                              f"{time.monotonic() - t0:.1f}s out={r['output_file']}")
        finally:
            adapter.extra_options = {}
            if runs:
                # ★method = 접미사 없는 대표 태스크 ID(C14) — runs[].task 에만 조건이 붙는다.
                #   자유문자열이 새지 않게 각 조각이 실제 TESTSET ID 인지 확인하고 넘긴다.
                bad = [t for t in task_ids if t not in testset.TASKS]
                if bad:
                    raise KeyError(f"TESTSET에 없는 task ID {bad} — method 는 TESTSET ID 참조만(C14).")
                path = harness.record_run_yaml(adapter, model, runs, run_dir,
                                               date_str=date_str, method=",".join(task_ids))
                print(f"  run.yaml → {path} ({len(runs)} runs)")
            ok, ps = unload(model)
            unload_failed = unload_failed or not ok
            print(f"  [unload] {model} → {'OK' if ok else '⚠ 잔존'} · "
                  f"VRAM peak={peak if peak is not None else '측정 실패'} MiB")
            index["runs_dirs"].append({"tool": tool_spec, "model": model, "run_dir": run_dir,
                                       "tasks": task_ids, "n_runs": len(runs),
                                       "n_failed": len(failures), "failures": failures,
                                       "vram_peak_mib": peak, "unloaded": ok, "ollama_ps": ps})
            if failures:
                print(f"  ⚠ 실패 {len(failures)}회(회차 제외·본문에 정직 표기 대상)")

    # ★원자적 쓰기 — 도중에 끊기면 기존 인덱스가 빈 파일로 남는다(앞 레그 기록 증발).
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    print(f"\n[run_repro] 인덱스 → {out}")
    print(f"[run_repro] GPU 현재 = {gpu_used_mib()} MiB (기준선 {index['gpu_baseline_mib']} MiB)")
    if unload_failed:
        # §5-A: 반납 실패를 정상 종료로 덮지 않는다(다음 작업이 VRAM 잔존 위에서 시작하는 걸 막는다).
        print("🚨 모델 언로드 실패 — VRAM 잔존 확인 후 수동 정리 필요(`ollama ps` / `ollama stop`).",
              file=sys.stderr)
        return 3
    print("    ★남은 일: repro_score.py 채점 → ollama serve 종료 → nvidia-smi 복귀 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
