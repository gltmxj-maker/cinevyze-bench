#!/usr/bin/env python3
"""Option-order bias benchmark against local Ollama.

두 안을 주고 고르게 한 뒤 **순서만 뒤집어** 다시 묻는다.
축 = 제시 순서 2(AB/BA) x 완화 장치 2팔(그냥 고르기 / 장단점 먼저) x 케이스 12종.

★라벨 대조군(`--label-scheme reversed`) — 본런은 위쪽이 항상 ①이라 '위쪽'과 '①'이 붙어 있다.
   위쪽에 ②를 붙인 대조군을 따로 돌려 **자리 편향인지 라벨 편향인지** 가른다.
   대조군 없이 "순서 편향"이라고 발표하지 않는다.

This script is the write-origin for run.yaml: 매 시행의 프롬프트 전문·원응답·API 메타데이터를
raw/에 저장하고 그 기록에서만 run.yaml을 만든다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from orderbias_score import ARMS, LABEL_SCHEMES, ORDERS, score_run_dir, validate_cases

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("orderbias_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-orderbias-20260808"
PILOT_PER_KIND = 2
LABELS = ("①", "②")


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_cases(data)
    return data


def render_case(data: dict[str, Any], arm: str, case: dict[str, Any], order: str,
                label_scheme: str) -> tuple[str, dict[str, Any]]:
    """프롬프트 한 장과 그 배치 기록을 함께 만든다.

    order 는 어느 안이 위로 가는지, label_scheme 은 위쪽에 어떤 라벨이 붙는지를 정한다.
    이 둘을 따로 두지 않으면 자리와 라벨을 한꺼번에 바꾼 비교가 된다.
    """
    first_option = "x" if order == "AB" else "y"
    second_option = "y" if order == "AB" else "x"
    first_text = case[f"option_{first_option}"]
    second_text = case[f"option_{second_option}"]
    if label_scheme == "normal":
        first_label, second_label = LABELS[0], LABELS[1]
    else:
        first_label, second_label = LABELS[1], LABELS[0]

    lines = [data["prompt"]["preamble"], f"기준: {case['criterion']}"]
    if case.get("context"):
        lines.append("")
        lines.append(case["context"])
    lines.extend(["", f"{first_label} {first_text}", f"{second_label} {second_text}", "",
                  data["prompt"]["arms"][arm]])
    placement = {
        "order": order,
        "label_scheme": label_scheme,
        "pos1_option": first_option,
        "pos1_label": first_label,
        "pos2_option": second_option,
        "pos2_label": second_label,
    }
    return "\n".join(lines), placement


def api_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("API 응답 최상위 값이 JSON 객체가 아님")
    return decoded


def model_metadata(base_url: str, model: str, timeout: int) -> dict[str, Any]:
    try:
        return api_json(base_url.rstrip("/") + "/api/show", {"model": model}, timeout)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"모델 메타데이터 조회 실패: {exc}") from exc


def generate(base_url: str, model: str, prompt: str, num_ctx: int,
             timeout: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None, dict[str, Any]]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
               "options": {"num_ctx": num_ctx}}
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            response = api_json(base_url.rstrip("/") + "/api/generate", payload, timeout)
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": None})
            return response, attempts, None, payload
        except urllib.error.HTTPError as exc:
            error = f"HTTPError {exc.code}: {exc.reason}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            transient = exc.code in {408, 425, 429} or exc.code >= 500
            if not transient or attempt == 2:
                return None, attempts, error, payload
            time.sleep(1)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - started, 3), "error": error})
            if attempt == 2:
                return None, attempts, error, payload
            time.sleep(1)
    return None, attempts, attempts[-1]["error"], payload


def _existing(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        keys.add(json.loads(path.read_text(encoding="utf-8"))["key"])
    return keys


def _next_id(run_dir: Path) -> int:
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        maximum = max(maximum, int(row.get("run_id", 0)))
    return maximum


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compare_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Read back from the scored aggregate — never hand-written."""
    aggregate_path = run_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[dict[str, Any]] = []

    def push(metric: str, name: str, entry: dict[str, Any]) -> None:
        rows.append({"metric": metric, "arm": name, "value": entry["flip_rate"],
                     "flips": entry["flips"], "pairs": entry["pairs_parsed"],
                     "first_pick_rate": entry["first_pick_rate"],
                     "label1_pick_rate": entry["label1_pick_rate"],
                     "accuracy": entry["accuracy"], "parse_rate": entry["parse_rate"]})

    overall = aggregate.get("overall") or {}
    if overall:
        push("flip_rate_overall", "all", overall)
    for arm, entry in (aggregate.get("by_arm") or {}).items():
        push("flip_rate_by_arm", arm, entry)
    for model, entry in (aggregate.get("by_model") or {}).items():
        push("flip_rate_by_model", model, entry)
    for kind, entry in (aggregate.get("by_kind") or {}).items():
        push("flip_rate_by_kind", kind, entry)
    for case, entry in (aggregate.get("by_case") or {}).items():
        push("flip_rate_by_case", case, entry)
    control = (aggregate.get("label_control") or {}).get("overall") or {}
    if control and control.get("runs"):
        push("flip_rate_label_control", "reversed_labels", control)
    return rows


def write_run_yaml(run_dir: Path, metadata: dict[str, Any], data: dict[str, Any], num_ctx: int) -> None:
    entries = []
    models: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        models.add(row["model"])
        entries.append({
            "key": row["key"],
            "arm": row["arm"],
            "model": row["model"],
            "case_id": row["case_id"],
            "order": row["order"],
            "label_scheme": row["label_scheme"],
            "repeat": row["repeat"],
            "output_file": row["transcript_file"],
            "response_file": row["response_file"],
            "log_file": str(path.relative_to(run_dir)),
            "elapsed_s": row.get("elapsed_s"),
            "num_ctx": row.get("num_ctx"),
            "prompt_eval_count": (row.get("api_metrics") or {}).get("prompt_eval_count"),
            "eval_count": (row.get("api_metrics") or {}).get("eval_count"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": ("SGR-ORDERBIAS — 두 안을 제시해 하나를 고르게 한 뒤 제시 순서만 뒤집어 다시 묻고, "
                   "실질 안 기준으로 선택이 뒤집히는 비율을 결정론 판정한다. "
                   "축 = 제시 순서 2(AB/BA) x 완화 장치 2팔(그냥 고르기 / 장단점 먼저) x 케이스 12종 x 반복. "
                   "정답이 있는 6종은 정확도를 함께 재고, 정답이 없는 6종은 일관성만 잰다. "
                   "위쪽 라벨을 뒤바꾼 대조군을 따로 돌려 자리 편향과 라벨 편향을 가른다. "
                   "마지막 줄에서 답을 못 뽑은 회차는 분모에서 뺀다."),
        "access": "local",
        "model": sorted(models)[0] if models else None,
        "models": sorted(models),
        "generated_by": "run_orderbias_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "num_ctx": num_ctx,
        "axis": "presentation_order_x_arm",
        "arms": list(ARMS),
        "orders": list(ORDERS),
        "label_schemes": list(LABEL_SCHEMES),
        "cases": [case["id"] for case in data["cases"]],
        "case_kinds": {case["id"]: case["kind"] for case in data["cases"]},
        "synthetic_data": "선택지 12쌍 전부 자체 작성. 정답이 있는 6종은 정답 근거(why)를 케이스에 박아 둔다.",
        "compare": _compare_rows(run_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "ollama_models": os.environ.get("OLLAMA_MODELS"),
            "ollama_max_loaded_models": os.environ.get("OLLAMA_MAX_LOADED_MODELS"),
            "ollama_num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL"),
        },
        "model_metadata": metadata,
        "runs": entries,
    }
    target_path = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target_path)


def _write_records(run_dir: Path, run_id: int, record: dict[str, Any], prompt_text: str,
                   response: dict[str, Any] | None, response_text: str) -> dict[str, Any]:
    stem = f"{run_id:03d}"
    response_rel = f"raw/{stem}-response.txt"
    transcript_rel = f"raw/{stem}-output.txt"
    (run_dir / response_rel).write_text(response_text, encoding="utf-8")
    header = "\n".join(f"{key}: {value}" for key, value in record.items()
                       if key in ("key", "arm", "model", "case_id", "order", "label_scheme",
                                  "repeat", "num_ctx"))
    (run_dir / transcript_rel).write_text(
        f"run_id: {run_id}\n{header}\n\n[PROMPT]\n{prompt_text}\n\n[RAW RESPONSE]\n{response_text}\n"
        f"\n[API METADATA]\n{json.dumps(response or {}, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8")
    record.update({
        "run_id": run_id,
        "response_file": response_rel,
        "transcript_file": transcript_rel,
        "api_metrics": {key: (response or {}).get(key) for key in (
            "done_reason", "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration"
        )} if response else None,
    })
    (run_dir / f"raw/{stem}-gen.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _pilot_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """정답 있는 것과 없는 것을 섞어 고른다 — 한 갈래만 보면 채점기 결함을 못 잡는다."""
    picked: list[dict[str, Any]] = []
    for kind in ("objective", "subjective"):
        picked.extend([case for case in cases if case["kind"] == kind][:PILOT_PER_KIND])
    return picked


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    if args.dry_run:
        case = data["cases"][0]
        prompt_ab, placement_ab = render_case(data, "plain", case, "AB", args.label_scheme)
        prompt_ba, _ = render_case(data, "reason_first", case, "BA", args.label_scheme)
        print(json.dumps({
            "cases": [c["id"] for c in data["cases"]],
            "planned_calls_per_model": len(ARMS) * len(data["cases"]) * len(ORDERS) * args.repeats,
            "placement": placement_ab,
            "sample_prompt_AB_plain": prompt_ab,
            "sample_prompt_BA_reason_first": prompt_ba,
        }, ensure_ascii=False, indent=2))
        return 0

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    # 채점기는 이 스냅샷을 읽는다 — 케이스 파일이 나중에 바뀌어도 채점 기준은 실행 시점에 고정된다.
    (run_dir / "cases_snapshot.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    cases = _pilot_cases(data["cases"]) if args.mode == "pilot" else data["cases"]
    repeats = 1 if args.mode == "pilot" else args.repeats

    if args.mode == "pilot":
        (run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot", "cases": len(cases)}, ensure_ascii=False), encoding="utf-8")
    else:
        decision_path = run_dir / "pilot_decision.json"
        if not decision_path.exists():
            print("파일럿 판정 없음 — 같은 --run-dir에서 --mode pilot을 먼저 완료하세요.")
            return 3
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not decision.get("proceed"):
            print(f"파일럿 중단조건 발동: {decision.get('reasons')}")
            return 3

    metadata_raw = model_metadata(args.base_url, args.model, args.timeout)
    (run_dir / "model_metadata").mkdir(exist_ok=True)
    (run_dir / "model_metadata" / f"{args.model.replace(':', '_')}.json").write_text(
        json.dumps({"modified_at": metadata_raw.get("modified_at"), "details": metadata_raw.get("details"),
                    "parameters": metadata_raw.get("parameters")}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    existing = _existing(run_dir)
    run_id = _next_id(run_dir)
    plan = []
    for arm in ARMS:
        for case in cases:
            for order in ORDERS:
                for repeat in range(1, repeats + 1):
                    key = f"{arm}/{args.model}/{case['id']}/{order}/{args.label_scheme}/r{repeat}"
                    if key not in existing:
                        plan.append((key, arm, case, order, repeat))
    print(f"model={args.model} mode={args.mode} labels={args.label_scheme} "
          f"num_ctx={args.num_ctx} pending={len(plan)}", flush=True)

    for index, (key, arm, case, order, repeat) in enumerate(plan, 1):
        run_id += 1
        prompt_text, placement = render_case(data, arm, case, order, args.label_scheme)
        response, attempts, infra_error, payload = generate(
            args.base_url, args.model, prompt_text, args.num_ctx, args.timeout)
        response_value = response.get("response") if response is not None else None
        if response is not None and not isinstance(response_value, str):
            infra_error = infra_error or "invalid_api_response: response 필드가 문자열이 아님"
        response_text = response_value if isinstance(response_value, str) else ""
        record = {
            "key": key,
            "arm": arm,
            "model": args.model,
            "case_id": case["id"],
            "kind": case["kind"],
            "repeat": repeat,
            "num_ctx": args.num_ctx,
            **placement,
            "payload": {k: v for k, v in payload.items() if k != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(run_dir, run_id, record, prompt_text, response, response_text)
        print(f"[{index}/{len(plan)}] {key} {record['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    rows, aggregate = score_run_dir(run_dir)
    metadata = {}
    metadata_dir = run_dir / "model_metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.glob("*.json")):
            metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    write_run_yaml(run_dir, metadata, data, args.num_ctx)
    print(json.dumps({key: aggregate[key] for key in ("overall", "by_arm", "infra_errors")},
                     ensure_ascii=False, indent=2))
    print(f"rows_total={len(rows)}")
    if args.mode == "pilot" and (run_dir / "pilot_decision.json").exists():
        decision = json.loads((run_dir / "pilot_decision.json").read_text(encoding="utf-8"))
        if not decision["proceed"]:
            print(f"파일럿 중단조건: {decision['reasons']}")
            return 3
    return 0 if aggregate["infra_errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--label-scheme", choices=LABEL_SCHEMES, default="normal",
                        help="reversed = 위쪽에 ②를 붙이는 라벨 대조군")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
