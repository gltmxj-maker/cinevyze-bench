"""29~32 측정의 원출력, 해시, 집계 파일 기록 도구."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_yaml(run: Path, value: dict) -> None:
    target = run / "run.yaml"
    temp = run / "run.yaml.tmp"
    temp.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temp, target)


def base_yaml(writer: str, method: str, model: str, *, access: str = "local") -> dict:
    return {"tool": "ollama" if model != "u2net" else "rembg",
            "date": dt.date.today().isoformat(), "method": method, "access": access,
            "model": model, "generated_by": writer, "harness_version": "1.0",
            "tos_confirmed": True, "tos_source_url": "로컬 오픈웨이트/오픈소스 자체 구동",
            "runs": [], "compare": [], "disclosure": True}


def save_results(run: Path, rows: list[dict], aggregate: dict) -> None:
    save_json(run / "results.json", rows)
    save_json(run / "aggregate.json", aggregate)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def api(base_url: str, endpoint: str, payload: dict | None = None, *, timeout: int = 240) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + endpoint, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Ollama {endpoint}: 객체 응답 아님")
    return value


def generate(base_url: str, model: str, prompt: str, *, images: list[str] | None = None,
             seed: int = 29, timeout: int = 240) -> tuple[dict, dict, float]:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "5m",
               "options": {"temperature": 0, "seed": seed}}
    if images:
        payload["images"] = images
    started = time.monotonic()
    value = api(base_url, "/api/generate", payload, timeout=timeout)
    elapsed = round(time.monotonic() - started, 3)
    if value.get("model") != model or not str(value.get("response") or "").strip():
        raise ValueError(f"Ollama {model}: 모델 불일치 또는 빈 출력")
    loaded = api(base_url, "/api/ps")
    ps = next((r for r in loaded.get("models", []) if r.get("name") == model), None)
    if not ps or not isinstance(ps.get("size_vram"), int):
        raise ValueError(f"Ollama /api/ps: {model} VRAM 측정 없음")
    return value, ps, elapsed


def fresh_run(run: Path) -> None:
    if run.exists() and any(p.name != "live" for p in run.iterdir()):
        raise FileExistsError(f"기존 런을 덮지 않습니다: {run}")
    run.mkdir(parents=True, exist_ok=True)


def materialize_short_outputs(run: Path, minimum: int = 200) -> int:
    """짧은 응답의 실제 API 원문 JSON을 게이트 출력으로 결박한다.

    사람이 텍스트를 패딩하지 않는다. 기존 응답 원문·SHA·로그 일치부터 확인한다.
    """
    meta = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    changed = 0
    for record in meta["runs"]:
        output = run / record["output_file"]
        if output.stat().st_size >= minimum:
            continue
        raw = output.read_bytes()
        if digest(raw) != record["output_sha256"]:
            raise ValueError(f"원응답 SHA 불일치: {output.name}")
        log = json.loads((run / record["log_file"]).read_text(encoding="utf-8"))
        response = log["response"]
        if response["response"] != raw.decode("utf-8"):
            raise ValueError(f"API 로그와 원응답 불일치: {output.name}")
        evidence = run / output.name.replace("-output.txt", "-api-response.json")
        evidence.write_text(json.dumps({"response": response, "text_output_file": output.name,
                             "text_sha256": digest(raw), "source_log": record["log_file"]},
                             ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if evidence.stat().st_size < minimum:
            raise ValueError(f"API 응답 원본도 {minimum}B 미만: {evidence.name}")
        record["output_file"] = evidence.name
        record["evidence_sha256"] = digest(evidence.read_bytes())
        changed += 1
    if changed:
        save_yaml(run, meta)
    return changed


def verify_text_run(run: Path) -> dict:
    """원문, API 로그, 선언 해시, 게이트 증거를 전건 대조."""
    meta = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    rows = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if len(rows) != len(meta["runs"]):
        raise ValueError("회차 수 불일치")
    for row, record in zip(rows, meta["runs"]):
        output = run / row["output_file"]
        log = json.loads((run / row["log_file"]).read_text(encoding="utf-8"))
        raw = output.read_bytes()
        if digest(raw) != row["output_sha256"] or row["output_sha256"] != record["output_sha256"]:
            raise ValueError(f"출력 SHA 불일치: {output.name}")
        if log["response"]["response"] != raw.decode("utf-8"):
            raise ValueError(f"API 응답 불일치: {output.name}")
        request = log["request"]
        input_text = request.get("prompt") or request.get("question")
        if not input_text:
            raise ValueError(f"질문 누락: {output.name}")
        if "image_sha256" in row:
            image = Path(row["image"])
            if digest(image.read_bytes()) != row["image_sha256"]:
                raise ValueError(f"입력 이미지 SHA 불일치: {image}")
            if request.get("question") != row["question"]:
                raise ValueError(f"질문 로그 불일치: {output.name}")
        elif digest(input_text.encode()) != row["input_sha256"]:
            raise ValueError(f"입력 SHA 불일치: {output.name}")
        gate_output = run / record["output_file"]
        if gate_output.name != output.name:
            evidence = json.loads(gate_output.read_text(encoding="utf-8"))
            if (evidence["response"]["response"] != raw.decode("utf-8") or
                    evidence["text_sha256"] != row["output_sha256"] or
                    digest(gate_output.read_bytes()) != record["evidence_sha256"]):
                raise ValueError(f"API 증거 불일치: {gate_output.name}")
    return {"verified": len(rows), "generated_by": meta["generated_by"], "status": "PASS"}
