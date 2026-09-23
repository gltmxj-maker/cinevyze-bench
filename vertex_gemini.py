#!/usr/bin/env python3
"""Vertex AI Gemini REST 호출(2026-09-23) — 18·19번 재제작 하네스 공용.

- 인증 = 환경변수 `VERTEX_SA_FILE`(서비스 계정 키 경로 · 저장소 밖) → google-auth 로 토큰 발급.
- 리전 = global(2026-09-23 probe: gemini-3.1-pro-preview 는 global 200 · us-central1 404).
- ★생각 토큰이 출력 상한을 먼저 먹는다(3.8 Flash probe 에서 64 상한 → MAX_TOKENS 잘림).
  그래서 상한을 넉넉히 두고, 매 회차 finishReason·thoughtsTokenCount 를 돌려준다 — 잘린 답을
  오답으로 세지 않도록 호출자가 infra 로 분리한다.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_CREDS = None


def _token() -> tuple[str, str]:
    global _CREDS
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    if _CREDS is None:
        path = os.environ.get("VERTEX_SA_FILE")
        if not path or not os.path.exists(path):
            raise RuntimeError("VERTEX_SA_FILE 이 없거나 파일이 없습니다")
        _CREDS = service_account.Credentials.from_service_account_file(path, scopes=[SCOPE])
    if not _CREDS.valid:
        _CREDS.refresh(Request())
    return _CREDS.token, _CREDS.project_id


def endpoint(model: str, method: str = "generateContent", location: str = "global") -> str:
    _, project = _token()
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:{method}")


def generate(model: str, prompt: str, *, thinking_level: str | None = None, max_output_tokens: int = 32768,
             timeout: int = 300, location: str = "global") -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """generateContent 1회(429·5xx 는 최대 3회 재시도). 반환 = (응답 JSON, 시도 기록, infra 오류)."""
    gen_cfg: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
    if thinking_level:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": thinking_level}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
    attempts: list[dict[str, Any]] = []
    backoff = (5, 15, 30)  # 429(할당량)·5xx 는 기다렸다 다시 — 최대 4회. 시도마다 기록이 남는다.
    for attempt in range(1, len(backoff) + 2):
        t0 = time.monotonic()
        retryable = True
        try:
            tok, _ = _token()
            req = urllib.request.Request(endpoint(model, location=location),
                                         data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                         headers={"Authorization": f"Bearer {tok}",
                                                  "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": None})
            return data, attempts, None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            err = f"HTTP {exc.code}: {detail}"
            retryable = exc.code == 429 or exc.code >= 500
        except RuntimeError as exc:  # 키 파일 없음 같은 설정 오류 — 기다려도 안 풀린다
            err = f"{type(exc).__name__}: {exc}"
            retryable = False
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            err = f"{type(exc).__name__}: {exc}"
        attempts.append({"attempt": attempt, "elapsed_s": round(time.monotonic() - t0, 3), "error": err})
        if not retryable or attempt > len(backoff):
            return None, attempts, err
        time.sleep(backoff[attempt - 1])
    raise AssertionError("unreachable")


def limit_child() -> None:
    """생성 코드 실행용 자식 프로세스 자원 상한(preexec_fn). 격리 샌드박스가 아니라 폭주 방지다 —
    CPU 120초 · 주소공간 2GB · 파일 크기 200MB."""
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (200 << 20, 200 << 20))


def request_body_shape(thinking_level: str | None, max_output_tokens: int) -> dict[str, Any]:
    """run.yaml 에 남길 요청 설정(프롬프트 제외)."""
    cfg: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
    if thinking_level:
        cfg["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {"generationConfig": cfg, "temperature": "미지정(모델 기본값)"}


def extract(resp: dict[str, Any] | None) -> dict[str, Any]:
    """응답에서 본문 텍스트·종료 사유·토큰 사용량을 뽑는다(생각 요약 파트는 본문에서 뺀다)."""
    if not resp:
        return {"text": "", "finish_reason": None, "usage": None, "model_version": None}
    cands = resp.get("candidates") or []
    cand = cands[0] if cands else {}
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return {"text": text, "finish_reason": cand.get("finishReason"), "usage": resp.get("usageMetadata"),
            "model_version": resp.get("modelVersion")}
