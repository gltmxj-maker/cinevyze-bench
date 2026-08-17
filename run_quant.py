#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_quant.py — Lane A 실측 드라이버: 같은 모델에서 **양자화만** 바꿔 VRAM·속도·정확도 비교.

★축 분리: 07-20 'GPU 사다리'는 *모델 크기* 축, 07-21은 *모델 간* 축. 이건 **양자화 축**(크기·계열 고정).

역할 분담(위조 방지):
  - run.yaml 은 `tool_test_harness`가 **직접** 기록한다(generated_by=tool_test_harness = 문자 그대로 사실).
    이 드라이버는 run.yaml 을 손대지 않는다 — 오케스트레이션 + VRAM 샘플링 + 집계만 한다.
  - 집계 산출 `quant_results.json` 은 별도 파일(07-20 ladder_results.json 선례).

§5-A 자원 규율(하드):
  - 로드 전 `model_fit_gate.py` GO/NO-GO — NO-GO면 그 양자화는 **실행하지 않고** 사유만 기록(지어낸 수치 0).
  - 모델 한 번에 하나(OLLAMA_MAX_LOADED_MODELS=1·NUM_PARALLEL=1) · 상주 금지(OLLAMA_KEEP_ALIVE=0).
  - 설치는 SSD 단일 폴더(홈 디스크 보호) · 끝나면 서버 종료 + VRAM 복귀 확인.

사용:
  python run_quant.py --out ./out/<post>/runs
  python run_quant.py --out ... --repeats 2 --dry-run      # 게이트·태그만 점검(실행 X)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

# ── 환경은 harness/adapter import 전에 박는다(어댑터가 os.environ을 읽어 서버를 띄운다) ──
SSD_MODELS = os.environ.get("QUANT_MODELS_DIR", "./models/ollama-quant")
os.environ.setdefault("OLLAMA_MODELS", SSD_MODELS)
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
os.environ["OLLAMA_NUM_PARALLEL"] = "1"
os.environ["OLLAMA_KEEP_ALIVE"] = "0"       # ★VRAM 상주 금지

OLLAMA_BIN = os.path.expanduser("~/.local/ollama/bin/ollama")
BASE_MODEL = "qwen2.5"
TASKS = ["TXT-01", "TXT-02", "TXT-04"]

# 레지스트리 매니페스트에서 firsthand 확인한 가중치 크기(2026-07-26) — 게이트 입력값.
QUANTS = [
    {"tag": "7b-instruct-q4_K_M", "weights_gb": 4.68, "label": "Q4_K_M"},
    {"tag": "7b-instruct-q8_0",   "weights_gb": 8.10, "label": "Q8_0"},
    {"tag": "7b-instruct-fp16",   "weights_gb": 15.24, "label": "FP16"},
]

# TXT-02 채점: 생성된 dedup_sort 를 실제로 실행해 검증(TESTSET '[정량] 실제 실행 성공/실패')
TXT02_CHECK = (
    "\nimport json,sys\n"
    "try:\n"
    "    r = dedup_sort([3,1,2,3,1])\n"
    "except Exception as e:\n"
    "    print(json.dumps({'ok': False, 'err': type(e).__name__ + ': ' + str(e)})); sys.exit(0)\n"
    "print(json.dumps({'ok': list(r) == [3,2,1], 'got': list(r)}))\n"
)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_used_mib():
    """GPU0 사용량(MiB). 실패는 None(예외 전파 금지 — 폴러 스레드가 죽으면 측정이 조용히 틀어진다)."""
    try:
        r = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10)
        if r.returncode != 0:
            return None
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramPoller(threading.Thread):
    """실행 중 VRAM 사용량을 0.5초 간격 샘플링 → 최대치. keep_alive=0이라 로드 구간이 짧아 촘촘히 본다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.errors = 0          # ★샘플링 실패 횟수 — 0이 아니면 peak을 믿지 말 것
        self._stop = threading.Event()

    def run(self):
        # ★예외로 스레드가 조용히 죽으면 peak이 과소 측정된다(거짓 GREEN) → 삼키되 카운트는 남긴다.
        while not self._stop.is_set():
            try:
                v = gpu_used_mib()
            except Exception:
                v, self.errors = None, self.errors + 1
            if v is not None:
                self.samples.append(v)
            time.sleep(0.5)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)


def fit_gate(weights_gb):
    r = sh([sys.executable, "model_fit_gate.py", "--gb", str(weights_gb)], timeout=60)
    return r.returncode, (r.stdout or "").strip()


def ensure_server():
    # ★이미 떠 있는 서버는 우리가 심은 OLLAMA_MAX_LOADED_MODELS·KEEP_ALIVE를 안 물고 있을 수 있다.
    #   그 상태의 측정은 조건이 다르므로 조용히 진행하지 않고 경고한다(측정 무결성).
    if sh([OLLAMA_BIN, "list"], timeout=15).returncode == 0:
        print("    [!] 이미 실행 중인 ollama 서버를 사용합니다 — 자원 규율 env가 적용되지 않았을 수 있음"
              "(정확한 조건이 필요하면 서버를 내리고 다시 실행).", flush=True)
        return True
    subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if sh([OLLAMA_BIN, "list"], timeout=10).returncode == 0:
            return True
        time.sleep(1)
    return False


def pull(model):
    print(f"    [pull] {model} → {os.environ['OLLAMA_MODELS']}", flush=True)
    try:
        r = subprocess.run([OLLAMA_BIN, "pull", model], text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"    [pull] 타임아웃(1시간) — {model} 중단", flush=True)
        return False
    return r.returncode == 0


def unload(model):
    sh([OLLAMA_BIN, "stop", model], timeout=60)
    time.sleep(2)
    ps = sh([OLLAMA_BIN, "ps"], timeout=30).stdout
    return model not in ps, ps.strip()


def tokps_from_logs(run_dir):
    """호출로그(harness가 기록)에서 tok/s·eval_count 파싱 — 드라이버가 숫자를 만들지 않는다."""
    out = []
    for fn in sorted(os.listdir(run_dir)):
        if not fn.endswith("-invocation.log"):
            continue
        txt = open(os.path.join(run_dir, fn), encoding="utf-8").read()
        d = {"log": fn}
        for key in ("task", "elapsed_s", "tok_per_s", "eval_count", "prompt_eval_count"):
            m = re.search(rf"^{key}: (.+)$", txt, re.M)
            if m:
                d[key] = m.group(1).strip()
        out.append(d)
    return out


def grade_txt02(run_dir, runs):
    """TXT-02 출력에서 파이썬 코드블록을 뽑아 실제 실행 → [3,2,1] 반환 여부(정량)."""
    results = []
    for r in runs:
        if r["task"] != "TXT-02":
            continue
        try:
            with open(os.path.join(run_dir, r["output_file"]), encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            results.append({"output_file": r["output_file"], "ok": False, "err": f"출력파일 없음: {e}"})
            continue
        blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
        code = next((b for b in blocks if "def dedup_sort" in b), None)
        if code is None:
            results.append({"output_file": r["output_file"], "ok": False,
                            "err": "코드블록에서 dedup_sort 정의를 못 찾음"})
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code + TXT02_CHECK)
            path = f.name
        try:
            p = sh([sys.executable, path], timeout=30)
            try:
                verdict = json.loads((p.stdout or "").strip().splitlines()[-1])
                if not isinstance(verdict, dict):
                    verdict = {"ok": False, "err": f"비-객체 출력: {verdict!r}"[:300]}
            except Exception:
                verdict = {"ok": False, "err": (p.stderr or p.stdout or "no output")[:300]}
        except Exception as e:                      # 타임아웃·OS 오류로 채점 전체가 죽지 않게
            verdict = {"ok": False, "err": f"{type(e).__name__}: {e}"[:300]}
        finally:
            os.unlink(path)
        verdict["output_file"] = r["output_file"]
        results.append(verdict)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="집계 산출 폴더(글 폴더의 runs/)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.environ["OLLAMA_MODELS"], exist_ok=True)

    baseline = gpu_used_mib()
    print(f"[baseline] GPU used = {baseline} MiB")

    report = {"base_model": BASE_MODEL, "tasks": TASKS, "repeats": a.repeats,
              "baseline_vram_mib": baseline, "models_dir": os.environ["OLLAMA_MODELS"],
              "quants": []}

    # 1) 게이트 먼저 전량 판정 — NO-GO는 다운로드조차 하지 않는다(디스크·시간 낭비 방지)
    for q in QUANTS:
        code, text = fit_gate(q["weights_gb"])
        q["gate_exit"] = code
        q["gate_verdict"] = "GO" if code == 0 else ("NO-GO" if code == 1 else "판단불가")
        q["gate_text"] = text
        print(f"[gate] {q['label']:8s} weights={q['weights_gb']}GB → {q['gate_verdict']}")

    if a.dry_run:
        with open(os.path.join(a.out, "quant_gate_dryrun.json"), "w", encoding="utf-8") as fh:
            json.dump(report | {"quants": QUANTS}, fh, ensure_ascii=False, indent=2)
        print("[dry-run] 게이트 판정만 기록하고 종료.")
        return

    if not ensure_server():
        print("[중단] ollama 서버 기동 실패", file=sys.stderr)
        sys.exit(2)

    import tool_test_harness as harness   # 환경 확정 후 import
    import testset

    for q in QUANTS:
        entry = {k: q[k] for k in ("tag", "label", "weights_gb", "gate_exit", "gate_verdict", "gate_text")}
        model = f"{BASE_MODEL}:{q['tag']}"
        entry["model"] = model

        if q["gate_exit"] != 0:
            entry["executed"] = False
            entry["skip_reason"] = "model_fit_gate NO-GO — 이 카드(16GB)엔 안 들어감. 실행 안 함(추정치 기재 금지)."
            print(f"[skip] {q['label']}: 게이트 NO-GO → 실행 안 함")
            report["quants"].append(entry)
            continue

        print(f"\n=== {q['label']} ({model}) ===")
        if not pull(model):
            entry["executed"] = False
            entry["skip_reason"] = "ollama pull 실패"
            report["quants"].append(entry)
            continue

        tasks = [(t, testset.build_prompt(t)) for t in TASKS for _ in range(a.repeats)]
        # ★양자화마다 baseline 재측정 — 직전 모델이 덜 내려갔으면 delta가 부풀려진다(적대검증 채택)
        pre = gpu_used_mib()
        entry["pre_run_vram_mib"] = pre
        poller = VramPoller()
        poller.start()
        run_dir = runs = None
        try:
            run_dir, runs = harness.run_testset(f"ollama:{model}", tasks, timeout=600)
        except Exception as e:                      # 측정 실패도 정직 기록(숫자를 지어내지 않는다)
            entry["executed"] = False
            entry["skip_reason"] = f"harness 예외: {type(e).__name__}: {e}"[:300]
            print(f"    [!] {model} 실행 실패: {e}")
        finally:
            poller.stop()
            # ★§5-A: 예외가 나도 반드시 반납한다(unload가 try 밖에 있으면 모델이 VRAM에 남는다)
            ok, ps = unload(model)
            entry["unloaded"] = ok
            entry["ollama_ps_after"] = ps

        peak = max(poller.samples) if poller.samples else None
        entry["vram_peak_mib"] = peak
        entry["vram_samples_n"] = len(poller.samples)
        entry["vram_sample_errors"] = poller.errors      # >0 이면 peak을 신뢰하지 말 것
        base_for_delta = pre if pre is not None else baseline
        entry["vram_delta_mib"] = (peak - base_for_delta) if (peak is not None and base_for_delta is not None) else None

        if runs is not None:
            entry["executed"] = True
            entry["run_dir"] = run_dir
            entry["run_yaml"] = os.path.join(run_dir, "run.yaml")
            entry["invocations"] = tokps_from_logs(run_dir)
            entry["txt02_execution"] = grade_txt02(run_dir, runs)
            entry["elapsed_by_task"] = {}
            for r in runs:
                entry["elapsed_by_task"].setdefault(r["task"], []).append(r["elapsed_s"])

        print(f"    [unload] {model} → {'OK' if entry['unloaded'] else '⚠ 잔존'} · VRAM peak={peak} MiB")
        report["quants"].append(entry)
        # ★중간 저장 — 뒤쪽 양자화에서 죽어도 앞 결과가 통째로 날아가지 않게(적대검증 채택)
        with open(os.path.join(a.out, "quant_results.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    report["final_vram_mib"] = gpu_used_mib()
    path = os.path.join(a.out, "quant_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n[+] 집계 → {path}")
    print(f"[+] 종료 시 GPU used = {report['final_vram_mib']} MiB (baseline {baseline})")
    print("    ★남은 일: ollama serve 종료 + nvidia-smi 복귀 확인 + SSD 모델폴더 삭제(cleanup)")


if __name__ == "__main__":
    main()
