#!/usr/bin/env python3
"""Deterministic scorer for the citation-link survival benchmark.

AI가 근거로 적어준 URL을 실제로 눌러보고, 그 결과만으로 판정한다. 판정은 다섯 갈래다.

  dns_fail    호스트 이름 자체가 세상에 없다 — 지어냈다는 가장 깨끗한 증거
  dead        연결 실패 · 타임아웃 · 4xx · 5xx
  blocked     401/403/429 — 우리 조회가 막힌 것. 링크가 죽었다는 뜻이 아니므로 분모에서 뺀다.
              (막힌 것을 죽은 것으로 세면 남의 방화벽을 AI 탓으로 돌리게 된다.)
  tls_error   인증서 검증 실패 — 주소는 실재하고 서버도 응답한다. 역시 분모에서 뺀다.
  redirect_loop 자기 자신으로 계속 되돌리는 3xx — 쿠키·스크립트를 요구하는 사이트라 우리 조회로는
              판정할 수 없다(브라우저에서는 열린다). 역시 분모에서 뺀다.
  soft_404    2xx인데 깊은 경로가 루트로 튕겼거나, 제목·본문이 '이 페이지는 없다'는 안내인 것
              — 살아 있는 척하는 죽은 링크. 한국 공공 사이트는 404를 잘 안 주므로 이 칸이 핵심이다.
  bare_domain 애초에 경로가 없는 대문 주소만 준 것 — 열리기는 하지만 근거가 아니다
  alive       위 어디에도 안 걸린 2xx

회차 단위 판정도 따로 둔다. URL을 아예 안 쓴 회차를 URL 분모에 섞으면
"URL을 적게 준 팔"이 "정확한 팔"로 둔갑한다.

  abstained   URL 없이 '출처 없음'류를 명시 — 도피구를 실제로 썼다
  no_url      URL도 없고 기권 문구도 없음

사람이 점수를 적어 넣을 자리는 없다. 네트워크 조회 결과는 실행 시점에 url_probe.json 으로 박제되고,
채점은 그 박제본만 읽는다(같은 실행을 몇 번 채점해도 같은 숫자가 나온다).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ARMS = ("bare", "permission", "no_url")
VERDICTS = ("alive", "bare_domain", "soft_404", "dead", "dns_fail", "blocked", "tls_error",
            "redirect_loop", "unprobed")
# 자체서명·만료 인증서는 '지어낸 주소'의 증거가 아니라 그 사이트의 성질이다. 분모에서 뺀다.
_TLS_ERROR_RE = re.compile(r"(SSL|CERTIFICATE_VERIFY_FAILED|SSLCertVerificationError|CERTIFICATE)", re.IGNORECASE)

# ★모델은 출처를 거의 항상 마크다운 링크로 적는다(파일럿 2026-08-08: 4/4).
#   `[https://a](https://a)` 를 한 덩어리로 긁으면 URL이 통째로 오염돼 멀쩡한 링크가 전부 죽은 링크가 된다.
#   그래서 마크다운 링크를 먼저 떼어 괄호 안 주소만 취하고, 남은 자리에서 맨 URL을 긁는다.
#   ★T2 적대검증 2026-08-08 — `[^)\s]+` 로 잡으면 URL 안의 괄호(위키백과 `훈민정음_(책)`)에서 잘려
#   멀쩡한 주소가 죽은 링크가 된다. 공백만 배제하고 탐욕 매칭 → 남는 괄호는 _clean 이 정리한다.
_MD_LINK_RE = re.compile(r"\[[^\]\n]*\]\(\s*(https?://[^\s]*)\s*\)", re.IGNORECASE)
# 대괄호는 URL에 안 쓰이므로 배제하고, 소괄호는 남긴다(위키백과 '훈민정음_(책)' 같은 실제 주소가 있다).
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'`\\\[\]]+", re.IGNORECASE)
_TRAILING = ".,;:!?]}>”’'\"*_"
# ★T2 적대검증 2026-08-08 — "자료(https://a.go.kr/x_(y))에서" 처럼 짝이 맞는 괄호로 끝난 뒤
#   조사가 붙는 경우. 괄호 깊이 절단은 이걸 못 잡아 주소에 '에서'가 남는다.
#   한글 경로 자체(`/폐기물분류`)는 건드리면 안 되므로 '닫는 괄호 바로 뒤'로 범위를 좁힌다.
_PARTICLE_AFTER_PAREN_RE = re.compile(r"(?<=\))[가-힣]+$")
# '출처 없음'을 실제로 쓴 회차. permission 팔이 준 도피구를 썼는지 본다.
_ABSTAIN_RE = re.compile(
    r"(출처\s*없음|출처가\s*없|출처를\s*찾을\s*수\s*없|출처를\s*알\s*수\s*없|"
    r"정확한\s*출처를\s*제시할\s*수\s*없|확실하지\s*않아|확실한\s*출처가\s*없|URL\s*없음)")
# ★본런 2026-08-08 — 한국 공공 사이트는 없는 경로에 404 대신 200 + '오류 안내 화면'을 준다.
#   제목만 보면 "기상청"·"대한민국 국회 홈페이지"처럼 멀쩡해 보여서, 지어낸 주소가 '살아 있는 근거'로
#   집계됐다(육안 대조에서 잡음). 그래서 제목과 본문 앞부분을 함께 본다.
#   여기 적힌 문구는 전부 '이 페이지는 없다/못 열었다'는 뜻으로만 쓰이는 표현이다.
_NOT_FOUND_RE = re.compile(
    r"(404|not\s*found|page\s*not\s*found|error\s*404|에러\s*페이지|에러페이지|"
    r"찾을\s*수\s*없|페이지를\s*찾지\s*못|존재하지\s*않는\s*페이지|삭제되었거나|"
    r"서비스\s*이용에\s*불편을\s*드려|서비스\s*장애가\s*계속|잘못된\s*접근|"
    r"올바르지\s*않은\s*(접근|경로)|접근\s*권한이\s*없)", re.IGNORECASE)
# 제목이 통째로 이것뿐이면 오류 화면이다(자바스크립트 alert 로 넘기는 정부 사이트가 있다).
_ERROR_ONLY_TITLES = {"error", "alert", "에러", "오류"}
# 제목 **끝**에 붙은 것만 잡는다 — "정책브리핑 - Error" 처럼 사이트 이름 뒤에 오류를 달아 주는 형태.
# 앞머리까지 잡으면 "오류 처리 가이드라인" 같은 멀쩡한 문서를 죽은 링크로 세게 된다.
_ERROR_TITLE_RE = re.compile(r"[\s\-–—|:·]\s*(error|에러|오류)\s*$", re.IGNORECASE)


def _clean(url: str) -> str:
    # ★본런 2026-08-08 회귀 가드 — 모델은 "누리집(https://a.go.kr)에서 확인" 처럼 괄호 안에 주소를 넣고
    #   조사를 붙여 쓴다. 안 닫힌 괄호에서 자르지 않으면 `https://a.go.kr)에서` 가 호스트명이 되어
    #   멀쩡한 도메인이 '없는 도메인(dns_fail)'으로 집계된다.
    depth = 0
    for index, char in enumerate(url):
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                url = url[:index]
                break
            depth -= 1
    url = url.rstrip(_TRAILING)
    return _PARTICLE_AFTER_PAREN_RE.sub("", url)


def extract_urls(text: str) -> list[str]:
    """응답에서 URL을 뽑아 중복 제거(등장 순서 유지).

    같은 링크를 두 번 적은 것은 근거 두 개가 아니다 — 한 번으로 센다.
    """
    source = text or ""
    found: list[tuple[int, str]] = []
    masked = list(source)
    for match in _MD_LINK_RE.finditer(source):
        found.append((match.start(1), match.group(1)))
        for index in range(match.start(), match.end()):
            masked[index] = " "
    for match in _BARE_URL_RE.finditer("".join(masked)):
        found.append((match.start(), match.group(0)))

    seen: set[str] = set()
    urls: list[str] = []
    for _, raw in sorted(found, key=lambda item: item[0]):
        url = _clean(raw)
        if len(url) < len("http://a.b"):
            continue
        if not urlsplit(url).netloc:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def has_path(url: str) -> bool:
    parts = urlsplit(url)
    return bool(parts.path.strip("/") or parts.query)


def classify(url: str, probe: dict[str, Any] | None) -> str:
    """조회 결과 하나를 판정. 우선순위 = 죽음 > 차단 > soft_404 > 대문주소 > 생존."""
    if probe is None:
        return "unprobed"
    if probe.get("dns_fail"):
        return "dns_fail"
    status = probe.get("status")
    if status is None:
        if _TLS_ERROR_RE.search(probe.get("error") or ""):
            # 인증서 문제로 우리 조회가 실패한 것이지 주소가 없는 것이 아니다.
            return "tls_error"
        return "dead"                      # 연결 실패·타임아웃
    if status in (401, 403, 429):
        return "blocked"
    if status >= 400:
        return "dead"
    if status >= 300:
        # ★본런 2026-08-08 — 리다이렉트를 다 따라가고도 3xx면 자기 자신으로 되도는 것이다.
        #   국토교통부·법무부처럼 쿠키/스크립트를 요구하는 사이트가 그렇다(브라우저에서는 열린다).
        #   이걸 '죽은 링크'로 세면 우리 클라이언트의 한계를 AI 탓으로 돌리게 된다.
        return "redirect_loop"
    title = (probe.get("title") or "").strip()
    if title.lower() in _ERROR_ONLY_TITLES:
        return "soft_404"
    if _ERROR_TITLE_RE.search(title):
        return "soft_404"
    if _NOT_FOUND_RE.search(title) or _NOT_FOUND_RE.search(probe.get("text_head") or ""):
        return "soft_404"
    final_url = probe.get("final_url") or url
    if has_path(url) and not has_path(final_url):
        # 깊은 경로를 요청했는데 대문으로 튕겼다 = 그 문서는 없다.
        return "soft_404"
    if not has_path(url):
        return "bare_domain"
    return "alive"


def title_echo(url: str, probe: dict[str, Any] | None, roots: dict[str, dict[str, Any]]) -> bool:
    """열리기는 하는데 사이트 대문과 제목이 똑같은 경우.

    한국 정부·공공 사이트 상당수는 없는 경로를 404로 돌려주지 않고 200에 대문을 얹어 준다
    (파일럿 2026-08-08에 실제로 나왔다). 그런 응답을 '살아 있는 근거'로 세면 지어낸 주소가
    멀쩡한 출처로 둔갑한다. 판정(verdict)을 덮어쓰지는 않고 의심 표시만 따로 남긴다 —
    본문 제목이 같다는 것만으로 죽었다고 단정할 수는 없기 때문이다.
    """
    if probe is None or not has_path(url):
        return False
    title = (probe.get("title") or "").strip()
    if not title:
        return False
    host = (urlsplit(probe.get("final_url") or url).hostname or "").lower()
    root = roots.get(host) or {}
    root_title = (root.get("title") or "").strip()
    return bool(root_title) and title == root_title


def score_generation(invocation: dict[str, Any], response_text: str,
                     probes: dict[str, dict[str, Any]],
                     roots: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    text = response_text or ""
    stripped = text.strip()
    urls = extract_urls(text)
    abstain = bool(_ABSTAIN_RE.search(text))

    if invocation.get("infra_error"):
        outcome = "infra"
    elif not stripped:
        outcome = "blank"
    elif urls:
        outcome = "scored"
    elif abstain:
        outcome = "abstained"
    else:
        outcome = "no_url"

    roots = roots or {}
    details = [{"url": url, "verdict": classify(url, probes.get(url)),
                "root_title_echo": title_echo(url, probes.get(url), roots)} for url in urls]
    counts = {verdict: sum(1 for item in details if item["verdict"] == verdict) for verdict in VERDICTS}
    checkable = (len(details) - counts["blocked"] - counts["unprobed"] - counts["tls_error"]
                 - counts["redirect_loop"])
    return {
        "outcome": outcome,
        "scored": outcome == "scored",
        "urls": len(details),
        "urls_checkable": checkable,
        "abstain_phrase": abstain,
        "response_chars": len(stripped),
        "verdicts": counts,
        "root_title_echo": sum(1 for item in details if item["root_title_echo"]),
        "alive_root_echo": sum(1 for item in details
                               if item["root_title_echo"] and item["verdict"] == "alive"),
        "url_detail": details,
        "response_head": stripped[:160].replace("\n", " "),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: _summarize(items) for name, items in sorted(groups.items())}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["outcome"] not in ("infra",)]
    counts = {verdict: sum(row["verdicts"][verdict] for row in usable) for verdict in VERDICTS}
    urls = sum(row["urls"] for row in usable)
    checkable = sum(row["urls_checkable"] for row in usable)
    tls_error = counts["tls_error"]
    redirect_loop = counts["redirect_loop"]
    broken = counts["dns_fail"] + counts["dead"] + counts["soft_404"]
    usable_alive = counts["alive"]
    runs_with_url = sum(1 for row in usable if row["urls"])
    abstained = sum(1 for row in usable if row["outcome"] == "abstained")
    no_url = sum(1 for row in usable if row["outcome"] == "no_url")
    return {
        "runs": len(rows),
        "runs_usable": len(usable),
        "infra": sum(1 for row in rows if row["outcome"] == "infra"),
        "blank": sum(1 for row in usable if row["outcome"] == "blank"),
        "runs_with_url": runs_with_url,
        "abstained": abstained,
        "no_url": no_url,
        "abstain_rate": round(abstained / len(usable), 4) if usable else None,
        "url_rate": round(runs_with_url / len(usable), 4) if usable else None,
        "urls": urls,
        "urls_per_run": round(urls / len(usable), 3) if usable else None,
        "urls_checkable": checkable,
        "blocked": counts["blocked"],
        "tls_error": tls_error,
        "redirect_loop": redirect_loop,
        "unprobed": counts["unprobed"],
        "broken": broken,
        "broken_rate": round(broken / checkable, 4) if checkable else None,
        "alive": usable_alive,
        "alive_rate": round(usable_alive / checkable, 4) if checkable else None,
        "bare_domain": counts["bare_domain"],
        # alive 로 셌지만 사이트 대문과 제목이 같은 것 — '살아 있는 근거'에서 따로 떼어 본다.
        "alive_root_echo": sum(row.get("alive_root_echo", 0) for row in usable),
        "by_verdict": counts,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, dict[str, int]] = {}
    for row in rows:
        for item in row["url_detail"]:
            host = urlsplit(item["url"]).netloc.lower()
            bucket = domains.setdefault(host, {"n": 0, "broken": 0, "alive": 0, "blocked": 0})
            bucket["n"] += 1
            if item["verdict"] in ("dns_fail", "dead", "soft_404"):
                bucket["broken"] += 1
            elif item["verdict"] == "alive":
                bucket["alive"] += 1
            elif item["verdict"] in ("blocked", "tls_error", "redirect_loop"):
                bucket["blocked"] += 1
    return {
        "overall": _summarize(rows),
        "by_arm": _group(rows, "arm"),
        "by_model": _group(rows, "model"),
        "by_question": _group(rows, "question_id"),
        "by_domain": dict(sorted(domains.items(), key=lambda kv: (-kv[1]["n"], kv[0]))),
        "infra_errors": sum(1 for row in rows if row["outcome"] == "infra"),
    }


def pilot_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        return {"proceed": False, "reasons": ["생성 0건"], "generations": 0, "infra_errors": 0}
    infra = sum(1 for row in rows if row["outcome"] == "infra")
    blank = sum(1 for row in rows if row["outcome"] == "blank")
    if infra / len(rows) > 0.10:
        reasons.append(f"infra_error_rate={infra}/{len(rows)}")
    if blank / len(rows) > 0.10:
        reasons.append(f"백지 응답 {blank}/{len(rows)}")
    # URL을 요구한 두 팔에서 URL이 전혀 안 나오면 잴 것이 없다.
    demanded = [row for row in rows if row["arm"] in ("bare", "permission")
                and row["outcome"] not in ("infra", "blank")]
    if demanded and not any(row["urls"] for row in demanded):
        reasons.append(f"URL 요구 팔 {len(demanded)}회 전부 URL 0개 — 링크 생존율을 잴 분모가 없다")
    probed = sum(row["urls"] - row["verdicts"]["unprobed"] for row in rows)
    total_urls = sum(row["urls"] for row in rows)
    if total_urls and probed / total_urls < 0.90:
        reasons.append(f"조회 안 된 URL {total_urls - probed}/{total_urls} — 네트워크 점검 필요")
    blocked = sum(row["verdicts"]["blocked"] for row in rows)
    if total_urls and blocked / total_urls > 0.50:
        reasons.append(f"차단 응답 {blocked}/{total_urls} — 우리 조회가 막혀 판정 불가")
    return {"proceed": not reasons, "reasons": reasons, "generations": len(rows), "infra_errors": infra}


def validate_cases(data: dict[str, Any]) -> None:
    """케이스셋부터 검증한다 — 팔 사이에 질문이 다르면 팔 차이가 아니라 질문 차이를 재게 된다."""
    prompt = data.get("prompt") or {}
    arms = prompt.get("arms") or {}
    if set(arms) != set(ARMS):
        raise ValueError(f"arms는 {ARMS} 셋이어야 한다: {sorted(arms)}")
    for name, instruction in arms.items():
        if not instruction.strip():
            raise ValueError(f"빈 팔 지시문: {name}")
    if "url" not in arms["no_url"].lower() and "URL" not in arms["no_url"]:
        raise ValueError("no_url 팔 지시문에 URL 금지 문구가 없다")
    questions = data.get("questions") or []
    if len(questions) < 3:
        raise ValueError("질문이 3종 미만이면 질문 하나의 특성을 결과로 착각하게 된다")
    seen: set[str] = set()
    for question in questions:
        qid = question.get("id")
        if not qid or qid in seen:
            raise ValueError(f"중복/빈 question id: {qid}")
        seen.add(qid)
        if not (question.get("text") or "").strip():
            raise ValueError(f"{qid} 질문 본문 없음")
        if not question.get("topic"):
            raise ValueError(f"{qid} topic 없음")
        if "http" in question["text"]:
            # 질문에 URL이 들어 있으면 모델이 그걸 베껴 적고 우리는 그걸 '생성한 출처'로 센다.
            raise ValueError(f"{qid} 질문에 URL이 들어 있다 — 채점이 오염된다")


_COLUMNS = ["run_id", "arm", "model", "question_id", "repeat", "outcome", "urls", "urls_checkable",
            "alive", "alive_root_echo", "bare_domain", "soft_404", "dead", "dns_fail", "blocked",
            "tls_error", "redirect_loop", "abstain_phrase", "response_chars", "elapsed_s", "infra_error",
            "response_head"]


def score_run_dir(run_dir: Path, cases_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases_path = cases_path or (run_dir / "cases_snapshot.json")
    validate_cases(json.loads(cases_path.read_text(encoding="utf-8")))
    probe_path = run_dir / "url_probe.json"
    probes: dict[str, dict[str, Any]] = {}
    if probe_path.exists():
        probes = json.loads(probe_path.read_text(encoding="utf-8"))
    root_path = run_dir / "root_probe.json"
    roots: dict[str, dict[str, Any]] = {}
    if root_path.exists():
        roots = json.loads(root_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "raw").glob("*-gen.json")):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        response_path = run_dir / invocation["response_file"]
        if response_path.exists():
            response = response_path.read_text(encoding="utf-8")
        else:
            # ★T2 적대검증 2026-08-08 — 파일 결손을 '모델이 빈 응답을 냈다'로 세면 안 된다.
            response = ""
            invocation = {**invocation, "infra_error": invocation.get("infra_error") or "response_file 없음"}
        scored = score_generation(invocation, response, probes, roots)
        rows.append({
            "run_id": invocation["run_id"],
            "key": invocation["key"],
            "arm": invocation["arm"],
            "model": invocation["model"],
            "question_id": invocation["question_id"],
            "repeat": invocation["repeat"],
            "elapsed_s": invocation.get("elapsed_s"),
            "infra_error": invocation.get("infra_error"),
            **scored,
        })

    (run_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in _COLUMNS}
            for verdict in ("alive", "bare_domain", "soft_404", "dead", "dns_fail", "blocked",
                            "tls_error", "redirect_loop"):
                flat[verdict] = row["verdicts"][verdict]
            writer.writerow(flat)

    aggregate = aggregate_results(rows)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows and (run_dir / "pilot_marker.json").exists() and not (run_dir / "pilot_decision.json").exists():
        decision = pilot_decision(rows)
        (run_dir / "pilot_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["pilot_decision"] = decision
    return rows, aggregate


def self_test() -> None:
    # --- URL 추출 ---
    text = ("근거는 https://www.weather.go.kr/w/index.do 입니다. "
            "그리고 [법령](https://law.go.kr/lsInfoP.do?lsiSeq=1234), "
            "https://www.weather.go.kr/w/index.do 를 다시 봐도 같습니다.")
    urls = extract_urls(text)
    assert urls == ["https://www.weather.go.kr/w/index.do",
                    "https://law.go.kr/lsInfoP.do?lsiSeq=1234"], urls
    # 문장 끝 마침표·괄호가 URL에 붙어 들어오면 멀쩡한 링크를 죽은 링크로 오판한다.
    assert extract_urls("보세요: https://example.test/a/b.") == ["https://example.test/a/b"]
    assert extract_urls("(https://example.test/a)") == ["https://example.test/a"]
    assert extract_urls("URL 없습니다") == []
    # ★파일럿 2026-08-08 회귀 가드 — 모델이 실제로 쓰는 형태는 이것이다.
    #   `](` 를 안 끊으면 주소가 통째로 오염돼 멀쩡한 링크가 전부 죽은 링크로 집계된다.
    assert extract_urls("출처: [https://www.weather.go.kr/w/bgm/index.do](https://www.weather.go.kr/w/bgm/index.do)") \
        == ["https://www.weather.go.kr/w/bgm/index.do"]
    assert extract_urls("근거는 [기상청 고시](https://kma.go.kr/a/b?x=1) 입니다") \
        == ["https://kma.go.kr/a/b?x=1"]
    # 주소 안의 짝 맞는 괄호는 살린다 — 잘라내면 살아 있는 주소가 죽은 주소가 된다.
    assert extract_urls("https://ko.wikipedia.org/wiki/훈민정음_(책)") \
        == ["https://ko.wikipedia.org/wiki/훈민정음_(책)"]
    # ★본런 2026-08-08 회귀 가드 — 괄호 안 주소 + 조사. 안 자르면 멀쩡한 도메인이 dns_fail 로 집계된다.
    assert extract_urls("국가유산청 누리집(https://www.cha.go.kr)에서 확인하세요") \
        == ["https://www.cha.go.kr"]
    # 한글 경로는 모델이 실제로 쓰는 형태다 — 살려 둔다.
    assert extract_urls("출처: https://www.me.go.kr/폐기물관리/분류") \
        == ["https://www.me.go.kr/폐기물관리/분류"]
    # ★T2 적대검증 2026-08-08 회귀 가드 — 마크다운 링크 안의 괄호가 잘리면 멀쩡한 주소가 죽은 링크가 된다.
    assert extract_urls("[훈민정음](https://ko.wikipedia.org/wiki/훈민정음_(책))") \
        == ["https://ko.wikipedia.org/wiki/훈민정음_(책)"]
    # ★T2 회귀 가드 — 짝 맞는 괄호 뒤 조사. 한글 경로 자체는 살려야 한다.
    assert extract_urls("자료(https://a.go.kr/x_(y))에서 확인") == ["https://a.go.kr/x_(y)"]
    assert extract_urls("출처: https://www.me.go.kr/폐기물관리/폐기물분류") \
        == ["https://www.me.go.kr/폐기물관리/폐기물분류"]
    # 마크다운과 맨 URL이 섞여도 문서 등장 순서를 지킨다.
    assert extract_urls("먼저 https://a.test/1 그리고 [글](https://b.test/2)") \
        == ["https://a.test/1", "https://b.test/2"]

    # --- 판정 ---
    assert classify("https://a.test/doc", {"dns_fail": True}) == "dns_fail"
    assert classify("https://a.test/doc", {"status": 404}) == "dead"
    assert classify("https://a.test/doc", {"status": None}) == "dead"
    assert classify("https://a.test/doc", {"status": 403}) == "blocked"
    # ★본런 2026-08-08 회귀 가드 — 자기 자신으로 되도는 3xx 는 우리 조회의 한계지 죽은 링크가 아니다.
    assert classify("https://a.test/doc", {"status": 307, "final_url": "https://a.test/doc"}) == "redirect_loop"
    # ★파일럿 2026-08-08 회귀 가드 — 자체서명 인증서 사이트를 '지어낸 주소'로 세면 안 된다.
    assert classify("https://a.test/doc", {"status": None,
                                           "error": "URLError: [SSL: CERTIFICATE_VERIFY_FAILED] ..."}) == "tls_error"
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/doc",
                                           "title": "고시 전문"}) == "alive"
    # 열리기는 하는데 제목이 '찾을 수 없음' — 살아 있는 척하는 죽은 링크
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/doc",
                                           "title": "404 Not Found"}) == "soft_404"
    # ★본런 2026-08-08 회귀 가드 — 제목은 멀쩡한데 본문이 오류 안내인 정부 사이트 화면.
    assert classify("https://a.test/doc", {
        "status": 200, "final_url": "https://a.test/doc", "title": "기상청",
        "text_head": "기상청 이 누리집은 대한민국 공식 전자정부 누리집입니다. 서비스 이용에 불편을 드려 죄송합니다."
    }) == "soft_404"
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/doc",
                                           "title": "Alert"}) == "soft_404"
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/doc",
                                           "title": "대한민국 국회 - 에러페이지"}) == "soft_404"
    # ★본런 2026-08-08 회귀 가드 — 사이트 이름 뒤에 오류를 붙여 주는 형태.
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/doc",
                                           "title": "정책브리핑 - Error"}) == "soft_404"
    # 제목 앞머리에 '오류'가 있는 멀쩡한 문서는 살려 둔다 — 여기까지 잡으면 헛것을 센다.
    assert classify("https://a.test/doc", {
        "status": 200, "final_url": "https://a.test/doc", "title": "오류 처리 가이드라인",
        "text_head": "이 문서는 시스템 오류 처리 절차를 설명합니다."}) == "alive"
    # 멀쩡한 본문은 그대로 alive 다.
    assert classify("https://a.test/doc", {
        "status": 200, "final_url": "https://a.test/doc", "title": "폭염 특보 기준",
        "text_head": "폭염주의보는 최고체감온도 33도 이상인 상태가 이틀 이상 지속될 것으로 예상될 때 발표합니다."
    }) == "alive"
    # 깊은 경로를 요청했는데 대문으로 튕김
    assert classify("https://a.test/doc", {"status": 200, "final_url": "https://a.test/",
                                           "title": "기상청"}) == "soft_404"
    # 애초에 대문 주소만 준 것은 살아 있어도 근거가 아니다
    assert classify("https://a.test/", {"status": 200, "final_url": "https://a.test/",
                                        "title": "기상청"}) == "bare_domain"
    assert classify("https://a.test/doc", None) == "unprobed"

    # ★대문 제목 되울림 — 없는 경로에 200 + 대문을 얹어 주는 사이트(파일럿에서 실제로 나왔다).
    #   판정은 alive 로 두되 의심 표시를 따로 남긴다.
    roots = {"a.test": {"title": "기상청"}}
    deep = {"status": 200, "final_url": "https://a.test/timetask/none", "title": "기상청"}
    assert classify("https://a.test/timetask/none", deep) == "alive"
    assert title_echo("https://a.test/timetask/none", deep, roots) is True
    # 대문 주소 자체는 되울림이 아니다(원래 대문이다).
    assert title_echo("https://a.test/", {"status": 200, "title": "기상청"}, roots) is False
    # 제목이 다르면 되울림이 아니다.
    assert title_echo("https://a.test/real", {"status": 200, "title": "폭염 고시"}, roots) is False
    echo_row = score_generation({}, "출처: https://a.test/timetask/none",
                                {"https://a.test/timetask/none": deep}, roots)
    assert echo_row["verdicts"]["alive"] == 1 and echo_row["alive_root_echo"] == 1, echo_row
    assert _summarize([echo_row])["alive_root_echo"] == 1

    probes = {
        "https://a.test/live": {"status": 200, "final_url": "https://a.test/live", "title": "고시"},
        "https://a.test/gone": {"status": 404},
        "https://nope.test/x": {"dns_fail": True},
        "https://b.test/": {"status": 200, "final_url": "https://b.test/", "title": "대문"},
        "https://c.test/wall": {"status": 403},
    }
    row = score_generation({}, "근거: https://a.test/live 와 https://a.test/gone 와 https://nope.test/x", probes)
    assert row["outcome"] == "scored" and row["urls"] == 3, row
    assert row["verdicts"]["alive"] == 1 and row["verdicts"]["dead"] == 1 and row["verdicts"]["dns_fail"] == 1
    assert row["urls_checkable"] == 3

    tls_row = score_generation({}, "근거: https://tls.test/x", {"https://tls.test/x": {
        "status": None, "error": "URLError: [SSL: CERTIFICATE_VERIFY_FAILED] self-signed"}})
    assert tls_row["urls"] == 1 and tls_row["urls_checkable"] == 0, tls_row
    assert _summarize([tls_row])["broken_rate"] is None, _summarize([tls_row])
    loop_row = score_generation({}, "근거: https://loop.test/x",
                                {"https://loop.test/x": {"status": 307}})
    assert loop_row["urls_checkable"] == 0 and _summarize([loop_row])["redirect_loop"] == 1, loop_row

    # 차단된 링크는 분모에서 빠진다 — 남의 방화벽을 AI 탓으로 돌리지 않는다.
    blocked_row = score_generation({}, "근거: https://c.test/wall 와 https://a.test/gone", probes)
    assert blocked_row["urls"] == 2 and blocked_row["urls_checkable"] == 1, blocked_row
    summary = _summarize([blocked_row])
    assert summary["broken"] == 1 and summary["broken_rate"] == 1.0, summary

    abstained = score_generation({}, "확실한 출처가 없어 출처 없음으로 적습니다.", probes)
    assert abstained["outcome"] == "abstained" and not abstained["scored"], abstained
    silent = score_generation({}, "국민 안전을 위한 제도입니다.", probes)
    assert silent["outcome"] == "no_url" and not silent["abstain_phrase"], silent
    blank = score_generation({}, "   ", probes)
    assert blank["outcome"] == "blank"
    infra = score_generation({"infra_error": "boom"}, "", probes)
    assert infra["outcome"] == "infra"

    # --- 집계 ---
    rows = [
        {**row, "arm": "bare", "model": "m1", "question_id": "Q1", "repeat": 1},
        {**abstained, "arm": "permission", "model": "m1", "question_id": "Q1", "repeat": 1},
        {**score_generation({}, "https://b.test/", probes), "arm": "bare", "model": "m1",
         "question_id": "Q2", "repeat": 1},
    ]
    agg = aggregate_results(rows)
    assert agg["overall"]["urls"] == 4, agg["overall"]
    assert agg["overall"]["broken"] == 2 and agg["overall"]["broken_rate"] == 0.5, agg["overall"]
    # 대문 주소는 죽지도 살지도 않은 제3의 칸이다 — alive 로 세면 근거인 척하게 된다.
    assert agg["overall"]["bare_domain"] == 1 and agg["overall"]["alive"] == 1, agg["overall"]
    assert agg["by_arm"]["permission"]["abstain_rate"] == 1.0
    assert agg["by_arm"]["bare"]["urls_per_run"] == 2.0, agg["by_arm"]["bare"]
    assert agg["by_domain"]["a.test"]["n"] == 2

    stop = pilot_decision([dict(silent, arm="bare"), dict(silent, arm="permission")])
    assert not stop["proceed"] and any("URL 요구 팔" in r for r in stop["reasons"]), stop
    healthy = [dict(row, arm="bare"), dict(abstained, arm="permission")]
    assert pilot_decision(healthy)["proceed"], pilot_decision(healthy)

    # --- 케이스 검증 ---
    good = {
        "prompt": {"preamble": "p", "arms": {"bare": "출처 URL도 함께", "permission": "확실한 URL만",
                                             "no_url": "URL은 쓰지 마세요"}},
        "questions": [{"id": f"Q{i}", "topic": "t", "text": "질문"} for i in (1, 2, 3)],
    }
    validate_cases(good)
    for broken_case, needle in (
        ({**good, "questions": good["questions"][:2]}, "3종 미만"),
        ({**good, "questions": [dict(good["questions"][0], text="https://a.test 를 보고")] * 3}, "URL이 들어 있다"),
        ({**good, "prompt": {"arms": {"bare": "a", "permission": "b"}}}, "arms는"),
    ):
        try:
            validate_cases(broken_case)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"검증이 통과시켰다: {needle}")
    print("linkcite_score self-test OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path, help="케이스 JSON만 검증")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.validate:
        validate_cases(json.loads(args.validate.read_text(encoding="utf-8")))
        print(f"{args.validate} 케이스 검증 OK")
        return 0
    if not args.run_dir:
        parser.error("--run-dir / --self-test / --validate 중 하나 필요")
    _, aggregate = score_run_dir(args.run_dir, args.cases)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
