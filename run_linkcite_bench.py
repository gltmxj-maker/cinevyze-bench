#!/usr/bin/env python3
"""Citation-link survival benchmark against local Ollama.

"출처도 같이 적어줘"라고 시킨 뒤, 받아 적은 URL을 **실제로 눌러본다.**
축 = 지시 방식 3팔(그냥 요구 / 확실한 것만+도피구 / URL 금지) x 질문 6종.

★이 측정이 재는 대상은 **웹 접근 없이 답하는 AI**다. 로컬 모델이라 검색을 못 한다.
   검색이 붙은 챗봇으로 일반화하지 않는다(원고 한계 섹션 강제).

This script is the write-origin for run.yaml: 매 시행의 프롬프트 전문·원응답·API 메타데이터를 raw/에
저장하고, 링크 조회 결과는 url_probe.json 에 박제한 뒤 그 기록에서만 run.yaml을 만든다.
네트워크 조회는 이 실행기에서만 일어나고 채점기는 박제본만 읽는다 — 같은 실행을 몇 번 채점해도
같은 숫자가 나온다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from linkcite_score import ARMS, extract_urls, score_run_dir, validate_cases

HARNESS_VERSION = "1.0"
DEFAULT_CASES = Path(__file__).with_name("linkcite_bench_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "ollama-linkcite-20260808"
PILOT_QUESTIONS = 2
PROBE_UA = "cinevyze-linkcheck/1.0 (+https://www.cinevyze.com; research; contact via site)"
PROBE_READ_BYTES = 65536
_TITLE_RE = re.compile(rb"<title[^>]*>(.{0,400}?)</title>", re.IGNORECASE | re.DOTALL)
_STRIP_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_META_CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)
TEXT_HEAD_CHARS = 1500


def _charset_of(body: bytes, header_charset: str | None) -> str:
    """★본런 2026-08-08 — 한국 공공 사이트는 아직 EUC-KR 이 많은데 HTTP 헤더에 charset 을 안 준다.
    UTF-8로 밀어 읽으면 본문이 깨져서 '페이지를 찾을 수 없습니다' 안내가 글자로 안 잡히고,
    그 결과 죽은 링크가 살아 있는 링크로 집계된다. 문서 안 meta charset 까지 본다."""
    if header_charset:
        return header_charset
    match = _META_CHARSET_RE.search(body[:2048])
    if match:
        return match.group(1).decode("ascii", "ignore")
    return "utf-8"


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_cases(data)
    return data


def build_prompt(data: dict[str, Any], arm: str, question: dict[str, Any]) -> str:
    prompt = data["prompt"]
    return (f"{prompt['preamble']}\n{prompt['arms'][arm]}\n{prompt['tail']}\n\n"
            f"질문: {question['text']}")


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


# --------------------------------------------------------------------------- 링크 조회


def _title_of(body: bytes, charset: str | None) -> str | None:
    match = _TITLE_RE.search(body)
    if not match:
        return None
    raw = match.group(1)
    for encoding in (charset, "utf-8", "cp949", "latin-1"):
        if not encoding:
            continue
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        return None
    return re.sub(r"\s+", " ", text).strip()[:200]


def _text_head(body: bytes, charset: str | None) -> str | None:
    """본문 앞부분을 태그 없이 남긴다 — 제목이 멀쩡해도 본문이 오류 안내인 화면을 가리기 위해서다."""
    for encoding in (charset, "utf-8", "cp949"):
        if not encoding:
            continue
        try:
            text = body.decode(encoding, errors="ignore")
            break
        except LookupError:
            continue
    else:
        return None
    text = _STRIP_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()[:TEXT_HEAD_CHARS] or None


def _wire_url(url: str) -> str:
    """★T2 적대검증 2026-08-08 — 한글이 든 주소를 그대로 보내면 urllib 가 UnicodeEncodeError 로 죽는다.
    그러면 조회가 아예 안 됐는데도 '연결 실패=죽은 링크'로 집계돼, 우리 클라이언트의 한계가
    AI 탓으로 둔갑한다(실측 2건: `me.go.kr/폐기물관리/…`, `josa.or.kr/…/1종보통.jsp`).
    경로·질의·조각을 퍼센트 인코딩하고, 호스트에 비ASCII가 있으면 IDNA 로 바꾼다."""
    parts = urlsplit(url)
    netloc = parts.netloc
    try:
        netloc.encode("ascii")
    except UnicodeEncodeError:
        try:
            netloc = netloc.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            return url
    return urlunsplit((parts.scheme, netloc,
                       quote(parts.path, safe="/%~:@!$&'()*+,;="),
                       quote(parts.query, safe="/%~:@!$&'()*+,;=?"),
                       quote(parts.fragment, safe="/%~:@!$&'()*+,;=?")))


def _dns_ok(host: str) -> bool:
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False
    except OSError:
        # DNS 서버 자체가 잠깐 안 될 수도 있다 — 그건 '없는 도메인'이 아니다.
        return True


def probe_url(url: str, timeout: int) -> dict[str, Any]:
    """링크 하나를 실제로 조회한다. 판정은 하지 않고 관측값만 남긴다."""
    host = urlsplit(url).hostname or ""
    record: dict[str, Any] = {"url": url, "probed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    if not host:
        record.update({"status": None, "error": "no_host"})
        return record
    if not _dns_ok(host):
        record.update({"dns_fail": True, "status": None, "error": "dns_nxdomain"})
        return record
    request = urllib.request.Request(_wire_url(url), headers={
        "User-Agent": PROBE_UA,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "ko,en;q=0.8",
    })
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(PROBE_READ_BYTES)
                record.update({
                    "status": response.status,
                    "final_url": response.geturl(),
                    "title": _title_of(body, _charset_of(body, response.headers.get_content_charset())),
                    "text_head": _text_head(body, _charset_of(body, response.headers.get_content_charset())),
                    "content_type": response.headers.get("Content-Type"),
                    "error": None,
                })
                return record
        except urllib.error.HTTPError as exc:
            record.update({"status": exc.code, "final_url": exc.url, "error": f"HTTPError {exc.code}"})
            return record
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            record.update({"status": None, "error": f"{type(exc).__name__}: {reason}"})
            if isinstance(getattr(exc, "reason", None), socket.gaierror):
                record["dns_fail"] = True
                return record
            if attempt == 2:
                return record
            time.sleep(1)
    return record


def probe_all(run_dir: Path, timeout: int, delay: float) -> dict[str, dict[str, Any]]:
    """raw/ 에 쌓인 응답에서 URL을 모아 아직 안 본 것만 조회한다(중단·재개 가능).

    같은 호스트를 연달아 때리지 않는다 — 남의 서버에 예의를 지키는 쪽이 우리 숫자에도 낫다
    (연타로 429를 받으면 멀쩡한 링크가 'blocked' 로 빠진다).
    """
    probe_path = run_dir / "url_probe.json"
    probes: dict[str, dict[str, Any]] = {}
    if probe_path.exists():
        probes = json.loads(probe_path.read_text(encoding="utf-8"))

    wanted: list[str] = []
    seen: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        if not response_path.exists():
            continue
        for url in extract_urls(response_path.read_text(encoding="utf-8")):
            if url not in probes and url not in seen:
                seen.add(url)
                wanted.append(url)

    if not wanted:
        print(f"[probe] 새로 조회할 URL 없음 (누적 {len(probes)}건)", flush=True)
        return probes

    # ★호스트마다 대문도 한 번 본다 — 없는 경로에 200 + 대문을 얹어 주는 사이트가 있어서,
    #   대문 제목을 모르면 지어낸 주소가 '살아 있는 근거'로 집계된다.
    root_path = run_dir / "root_probe.json"
    roots: dict[str, dict[str, Any]] = {}
    if root_path.exists():
        roots = json.loads(root_path.read_text(encoding="utf-8"))
    hosts = {}
    for url in wanted:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host and host not in roots:
            hosts.setdefault(host, f"{parts.scheme}://{parts.netloc}/")
    for index, (host, root_url) in enumerate(sorted(hosts.items()), 1):
        record = probe_url(root_url, timeout)
        roots[host] = {"root_url": root_url, "status": record.get("status"),
                       "title": record.get("title"), "text_head": record.get("text_head"),
                       "dns_fail": record.get("dns_fail")}
        print(f"[root {index}/{len(hosts)}] {record.get('status')} {host} :: {record.get('title')}", flush=True)
        root_path.write_text(json.dumps(roots, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(delay)

    last_hit: dict[str, float] = {}
    for index, url in enumerate(wanted, 1):
        host = (urlsplit(url).hostname or "").lower()
        gap = time.monotonic() - last_hit.get(host, 0.0)
        if host in last_hit and gap < delay:
            time.sleep(delay - gap)
        record = probe_url(url, timeout)
        last_hit[host] = time.monotonic()
        probes[url] = record
        verdict = "dns_fail" if record.get("dns_fail") else record.get("status")
        print(f"[probe {index}/{len(wanted)}] {verdict} {url}", flush=True)
        temporary = run_dir / "url_probe.json.tmp"
        temporary.write_text(json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, probe_path)
    return probes


# --------------------------------------------------------------------------- 실행


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
    for arm, entry in (aggregate.get("by_arm") or {}).items():
        rows.append({"metric": "broken_rate_by_arm", "arm": arm, "value": entry["broken_rate"],
                     "broken": entry["broken"], "total": entry["urls_checkable"],
                     "alive": entry["alive"], "bare_domain": entry["bare_domain"],
                     "urls_per_run": entry["urls_per_run"], "abstain_rate": entry["abstain_rate"],
                     "url_rate": entry["url_rate"], "blocked": entry["blocked"]})
    for model, entry in (aggregate.get("by_model") or {}).items():
        rows.append({"metric": "broken_rate_by_model", "arm": model, "value": entry["broken_rate"],
                     "broken": entry["broken"], "total": entry["urls_checkable"],
                     "urls_per_run": entry["urls_per_run"]})
    for question, entry in (aggregate.get("by_question") or {}).items():
        rows.append({"metric": "broken_rate_by_question", "arm": question, "value": entry["broken_rate"],
                     "broken": entry["broken"], "total": entry["urls_checkable"]})
    overall = aggregate.get("overall") or {}
    if overall:
        rows.append({"metric": "broken_rate_overall", "arm": "all", "value": overall["broken_rate"],
                     "broken": overall["broken"], "total": overall["urls_checkable"],
                     "alive": overall["alive"], "bare_domain": overall["bare_domain"],
                     "blocked": overall["blocked"]})
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
            "question_id": row["question_id"],
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
    probe_path = run_dir / "url_probe.json"
    probes = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
    payload = {
        "tool": "ollama",
        "date": dt.date.today().isoformat(),
        "method": ("SGR-LINKCITE — 웹 접근이 없는 로컬 모델에게 한국 제도 질문 6종의 답과 '출처'를 함께 "
                   "요구한 뒤, 응답에 적힌 URL을 실제 HTTP로 조회해 생존을 결정론 판정한다. "
                   "축 = 지시 방식 3팔(그냥 요구 / 확실한 것만+'출처 없음' 도피구 / URL 금지) x 질문 6종 x 반복. "
                   "401·403·429는 우리 조회가 막힌 것이므로 분모에서 뺀다. 경로 없는 대문 주소는 살아 있어도 "
                   "근거가 아니므로 별도 칸으로 센다. 검색이 붙은 챗봇으로 일반화하지 않는다."),
        "access": "local",
        "model": sorted(models)[0] if models else None,
        "models": sorted(models),
        "generated_by": "run_linkcite_bench.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)",
        "keep_alive": 0,
        "request_parallelism": 1,
        "num_ctx": num_ctx,
        "axis": "instruction_arm_x_question",
        "arms": list(ARMS),
        "questions": [question["id"] for question in data["questions"]],
        "link_probe": {
            "user_agent": PROBE_UA,
            "urls_probed": len(probes),
            "hosts_root_probed": len(json.loads((run_dir / "root_probe.json").read_text(encoding="utf-8")))
                                 if (run_dir / "root_probe.json").exists() else 0,
            "read_bytes": PROBE_READ_BYTES,
            "note": ("조회 결과는 url_probe.json·root_probe.json 에 박제 — 채점은 그 박제본만 읽는다"
                     "(재채점 재현성). 대문 제목을 따로 받아 두는 이유는 없는 경로에 200 + 대문을 "
                     "돌려주는 사이트를 '살아 있는 근거'로 세지 않기 위해서다."),
        },
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
                       if key in ("key", "arm", "model", "question_id", "repeat", "num_ctx"))
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


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    if args.dry_run:
        question = data["questions"][0]
        print(json.dumps({
            "questions": [q["id"] for q in data["questions"]],
            "planned_calls_per_model": len(ARMS) * len(data["questions"]) * args.repeats,
            "sample_prompt": build_prompt(data, "permission", question),
        }, ensure_ascii=False, indent=2))
        return 0

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    # 채점기는 이 스냅샷을 읽는다 — 케이스 파일이 나중에 바뀌어도 채점 기준은 실행 시점에 고정된다.
    (run_dir / "cases_snapshot.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.probe_only:
        probe_all(run_dir, args.probe_timeout, args.probe_delay)
        rows, aggregate = score_run_dir(run_dir)
        write_run_yaml(run_dir, {}, data, args.num_ctx)
        print(json.dumps(aggregate["overall"], ensure_ascii=False, indent=2))
        return 0

    questions = data["questions"][:PILOT_QUESTIONS] if args.mode == "pilot" else data["questions"]
    repeats = 1 if args.mode == "pilot" else args.repeats

    if args.mode == "pilot":
        (run_dir / "pilot_marker.json").write_text(
            json.dumps({"mode": "pilot", "questions": len(questions)}, ensure_ascii=False), encoding="utf-8")
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
        for question in questions:
            for repeat in range(1, repeats + 1):
                key = f"{arm}/{args.model}/{question['id']}/r{repeat}"
                if key not in existing:
                    plan.append((key, arm, question, repeat))
    print(f"model={args.model} mode={args.mode} num_ctx={args.num_ctx} pending={len(plan)}", flush=True)

    for index, (key, arm, question, repeat) in enumerate(plan, 1):
        run_id += 1
        prompt_text = build_prompt(data, arm, question)
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
            "question_id": question["id"],
            "topic": question["topic"],
            "repeat": repeat,
            "num_ctx": args.num_ctx,
            "payload": {k: v for k, v in payload.items() if k != "prompt"},
            "attempts": attempts,
            "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
            "infra_error": infra_error,
        }
        _write_records(run_dir, run_id, record, prompt_text, response, response_text)
        print(f"[{index}/{len(plan)}] {key} {record['elapsed_s']}s"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    if not args.no_probe:
        probe_all(run_dir, args.probe_timeout, args.probe_delay)

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
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--probe-timeout", type=int, default=15)
    parser.add_argument("--probe-delay", type=float, default=1.0, help="같은 호스트 연속 조회 간 최소 간격(초)")
    parser.add_argument("--no-probe", action="store_true", help="생성만 하고 링크 조회는 나중에")
    parser.add_argument("--probe-only", action="store_true", help="생성 없이 링크 조회+재채점만")
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
