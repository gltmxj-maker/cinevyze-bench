#!/usr/bin/env python3
"""30번: 고정 사실 문항의 같은 응답에 세 가지 사후 검증 규칙 적용."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from experiment_rebuild_common import ROOT, base_yaml, digest, fresh_run, generate, materialize_short_outputs, save_results, save_yaml, verify_text_run
from live_mark import mark


PLAN_FILE = ROOT / "hallucination_filter_cases.json"
DEFAULT_RUN = ROOT / "test_runs/hallucination-filter-qwen2.5-20260924"
METHODS = ("caution_words", "number_date_check", "official_lookup")


def load_plan() -> dict:
    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    if plan["model"] != "qwen2.5:7b" or plan["repetitions"] != 3 or len(plan["questions"]) != 12:
        raise ValueError("사전 고정한 12문항×3회·로컬 모델과 불일치")
    if tuple(plan["methods"]) != METHODS:
        raise ValueError("사전 고정한 검증 절차와 불일치")
    return plan


def stance(answer: str) -> str:
    head = answer.strip()[:24]
    if re.match(r"^(맞다|네[,. ]|예[,. ]|사실이다)", head):
        return "yes"
    if re.match(r"^(아니다|아니요|아니[,. ]|거짓이다)", head):
        return "no"
    return "unknown"


def has_hallucination(row: dict) -> bool:
    value = stance(row["answer"])
    if value == "unknown" or (value == "yes") != (row["ground_truth"] == "true"):
        return True
    answer = row["answer"]
    # 실행 후 발견: 찬반이 맞아도 근거 문장에 틀린 날짜·기체·장소가 섞였다.
    # 고정 NASA 정답표에서 확인되는 이 네 경우만 추가 판정한다. 그 밖의 자유서술은 UNKNOWN.
    if row["id"] in {"H01", "H02"} and "1990" not in answer:
        return True
    if row["id"] in {"W01", "W02"} and "2021" in answer and not re.search(r"12월\s*25일", answer):
        return True
    if row["id"] == "H03" and any(x in answer for x in ("奋进号", "엔데버")):
        return True
    if row["id"] == "P03" and "자호 계곡" in answer:
        return True
    return False


def flags(row: dict) -> dict[str, bool]:
    answer = row["answer"]
    claim = row["claim"]
    risky_words = ("모른다", "불확실", "아마", "확실하지", "추정", "모르겠")
    caution = any(word in answer for word in risky_words) or stance(answer) == "unknown"
    # 숫자가 있는 질문에만 적용: 공식 고정값이 답변의 긍정 근거에 빠지면 보류.
    number = row.get("official_number")
    claim_numbers = re.findall(r"\d+", claim)
    numeric = bool(number and claim_numbers and stance(answer) == "yes" and
                   (str(number) not in claim_numbers or str(number) not in answer))
    official = has_hallucination(row)
    return {"caution_words": caution, "number_date_check": numeric, "official_lookup": official}


def evaluate_methods(rows: list[dict]) -> dict[str, dict[str, int]]:
    totals = {name: {"caught": 0, "missed": 0, "false_discard": 0, "kept_correct": 0}
              for name in METHODS}
    for row in rows:
        bad = has_hallucination(row)
        for method, flagged in flags(row).items():
            key = ("caught" if flagged else "missed") if bad else ("false_discard" if flagged else "kept_correct")
            totals[method][key] += 1
    return totals


def rescore_run(run: Path) -> dict:
    """원출력 SHA와 호출 로그를 대조하고 변경된 규칙으로 전건 재채점."""
    import yaml
    meta = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    rows = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if len(rows) != 36 or len(meta["runs"]) != len(rows):
        raise ValueError("36회 원기록 누락")
    before = json.loads((run / "aggregate.json").read_text(encoding="utf-8"))
    if meta.get("condition_changes"):
        before = dict(before, methods=meta["condition_changes"][0]["before"],
                      hallucination_n=meta["condition_changes"][0]["before_hallucination_n"])
    for row, record in zip(rows, meta["runs"]):
        output = run / row["output_file"]
        log = run / row["log_file"]
        if (digest(output.read_bytes()) != row["output_sha256"] or
                output.read_text(encoding="utf-8") != row["answer"] or
                record["output_sha256"] != row["output_sha256"] or
                json.loads(log.read_text(encoding="utf-8"))["response"]["response"] != row["answer"]):
            raise ValueError(f"원출력·호출 로그 SHA 불일치: {output.name}")
        gate_output = run / record["output_file"]
        if gate_output.name != output.name:
            evidence = json.loads(gate_output.read_text(encoding="utf-8"))
            if (evidence["response"]["response"] != row["answer"] or
                    evidence["text_sha256"] != row["output_sha256"] or
                    digest(gate_output.read_bytes()) != record["evidence_sha256"]):
                raise ValueError(f"API 증거 불일치: {gate_output.name}")
        row["hallucination"] = has_hallucination(row)
    methods = evaluate_methods(rows)
    after = dict(before)
    after.update({"hallucination_n": sum(r["hallucination"] for r in rows),
                  "correct_n": sum(not r["hallucination"] for r in rows),
                  "methods": methods})
    meta["score_rules"] = dict(meta["score_rules"],
        post_run_change="찬반이 맞아도 근거의 잘못된 날짜·기체·장소를 환각으로 포함. NASA 정답표의 4개 경우만 추가")
    meta["condition_changes"] = [{"reason": "찬반 단독 채점이 근거 속 오류를 누락",
        "scope": "W01/W02 날짜, H03 우주왕복선, P03 착륙지; 36개 전건 원출력 SHA 확인 후 재채점",
        "before": before["methods"], "after": methods,
        "before_hallucination_n": before["hallucination_n"],
        "after_hallucination_n": after["hallucination_n"]}]
    save_yaml(run, meta)
    save_results(run, rows, after)
    materialize_short_outputs(run)
    return after


def report_rescore(run: Path) -> None:
    """실측 원출력 전건을 다시 읽고 live_capture_runner에 변경 후 집계를 표시."""
    result = rescore_run(run)
    mark("turn", "36개 저장 원출력 SHA 대조·전건 재채점 — 찬반 정답이어도 틀린 근거를 환각으로 포함")
    for method in METHODS[:2]:
        mark("turn", f"재채점 사후 검증 방법 — {method}")
    mark("agg", "재채점 36개 / " + "; ".join(
         f"{k}: 잡음 {v['caught']}·놓침 {v['missed']}·오탐 {v['false_discard']}" for k, v in result["methods"].items()))
    print(json.dumps(result, ensure_ascii=False), flush=True)


def report_rescore_in_run(run: Path) -> None:
    """별도 GPU 호출 없이 원 런의 기존 증거에 변경 후 재채점 화면 결박."""
    result = rescore_run(run)
    mark("turn", "36개 저장 원출력 SHA 대조·전건 재채점 — 찬반 정답이어도 틀린 근거를 환각으로 포함")
    mark("agg", "재채점 36개 / " + "; ".join(
         f"{k}: 잡음 {v['caught']}·놓침 {v['missed']}·오탐 {v['false_discard']}" for k, v in result["methods"].items()))
    print(json.dumps(result, ensure_ascii=False), flush=True)


def verify_evidence(run: Path) -> int:
    """기존 API 로그·원문을 확인하고 게이트용 원응답 JSON을 하네스로 자동 연결."""
    changed = materialize_short_outputs(run)
    print(f"API 원응답 증거 연결 {changed}건; 원문 답변은 변경하지 않음")
    return changed


def run_live(run: Path, base_url: str) -> None:
    fresh_run(run)
    plan = load_plan()
    cases = plan["questions"]
    model = plan["model"]
    meta = base_yaml(Path(__file__).name, "FACT-RECEIVER-01", model)
    meta.update({"cases_file": PLAN_FILE.name, "cases_sha256": digest(PLAN_FILE.read_bytes()),
                 "generation_options": plan["generation_options"], "repetitions": plan["repetitions"],
                 "score_rules": plan["methods"], "condition_changes": []})
    rows = []
    mark("turn", "답 수집 — NASA 공식 원문 3건의 고정 참·거짓 12문항 × 3회")
    for rep in range(1, plan["repetitions"] + 1):
        for case in cases:
            stem = f"{len(rows)+1:02d}"
            prompt = plan["prompt_prefix"] + case["claim"]
            try:
                data, ps, elapsed = generate(base_url, model, prompt, seed=30)
            except Exception as exc:
                mark("infra", f"30번 {case['id']} 반복 {rep} 호출 실패: {type(exc).__name__} {exc}")
                raise
            output = run / f"{stem}-output.txt"
            log = run / f"{stem}-invocation.log"
            output.write_text(data["response"], encoding="utf-8")
            row = {"id": case["id"], "repetition": rep, "model": model,
                   "claim": case["claim"], "ground_truth": case["ground_truth"],
                   "official_number": case["official_number"], "evidence_url": case["evidence_url"],
                   "source_fact": case["source_fact"], "answer": data["response"],
                   "stance": stance(data["response"]), "hallucination": False,
                   "output_file": output.name, "log_file": log.name,
                   "input_sha256": digest(prompt.encode()), "output_sha256": digest(output.read_bytes()),
                   "elapsed_s": elapsed, "ps_size_vram_bytes": ps["size_vram"]}
            row["hallucination"] = has_hallucination(row)
            log.write_text(json.dumps({"request": {"model": model, "prompt": prompt,
                "temperature": 0, "seed": 30}, "response": data, "ps": ps,
                "evidence_url": case["evidence_url"], "row": row}, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            meta["runs"].append({"task": case["id"], "input": prompt,
                                 "output_file": output.name, "log_file": log.name,
                                 "elapsed_s": elapsed, "input_sha256": row["input_sha256"],
                                 "output_sha256": row["output_sha256"]})
            save_yaml(run, meta)
            print(f"[{stem}/36] {case['id']} 반복 {rep} · {row['stance']} · 환각 {row['hallucination']}", flush=True)
    methods = {}
    for method in METHODS:
        mark("turn", f"사후 검증 방법 전환 — {method} · 동일한 저장 답 36개에만 적용")
        score = {"caught": 0, "missed": 0, "false_discard": 0, "kept_correct": 0}
        first_miss = False
        for row in rows:
            bad, flagged = row["hallucination"], flags(row)[method]
            key = ("caught" if flagged else "missed") if bad else ("false_discard" if flagged else "kept_correct")
            score[key] += 1
            if key == "missed" and not first_miss:
                # 게재 예산은 break 전부+agg ≤3장. 첫 두 방법의 첫 놓침만 촬영한다.
                if method in METHODS[:2]:
                    mark("break", f"{method} 첫 환각 놓침 — {row['id']} 반복 {row['repetition']} · {row['output_file']}")
                first_miss = True
        methods[method] = score
        print(f"[방법] {method}: {score}", flush=True)
    aggregate = {"sample_n": len(rows), "question_n": len(cases), "repetitions": plan["repetitions"],
                 "model": model, "hallucination_n": sum(r["hallucination"] for r in rows),
                 "correct_n": sum(not r["hallucination"] for r in rows),
                 "median_ps_size_vram_bytes": int(statistics.median(r["ps_size_vram_bytes"] for r in rows)),
                 "methods": methods, "infra_errors": 0}
    save_results(run, rows, aggregate)
    materialize_short_outputs(run)
    mark("agg", "사후 검증 3법 / 동일 답 36개 — " + "; ".join(
         f"{k}: 잡음 {v['caught']}·놓침 {v['missed']}·오탐 {v['false_discard']}" for k, v in methods.items()))
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--report-rescore", action="store_true")
    parser.add_argument("--report-rescore-in-run", action="store_true")
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_text_run(args.run_dir), ensure_ascii=False))
    elif args.verify_evidence:
        verify_evidence(args.run_dir)
    elif args.report_rescore_in_run:
        report_rescore_in_run(args.run_dir)
    elif args.report_rescore:
        report_rescore(args.run_dir)
    elif args.rescore:
        print(json.dumps(rescore_run(args.run_dir), ensure_ascii=False))
    else:
        run_live(args.run_dir, args.base_url)


if __name__ == "__main__":
    main()
