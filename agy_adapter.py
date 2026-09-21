"""
agy_adapter.py — Gemini 호출 1원화 어댑터 (Antigravity `agy` CLI · 구독 OAuth).

★출처: app-factory `scripts/lib/agy_adapter.py`를 all-blog로 가져옴(시네 2026-06-21
  "제미나이는 앱팩토리 안티그래비티 CLI 쓰면 돼, 그거 가져와"). vendoring = 자체완결
  (app-factory 내부 의존 안 만듦). 원본 firsthand 검증 2026-06-10 + all-blog 환경
  firsthand 재확인 2026-06-21(`agy ... -p "..." --print-timeout` → "PROBE OK" exit 0).

레일 B(애드센스) Lane A '자동 실측'의 LLM/툴 접근층:
  - Gemini 자체를 TESTSET 태스크로 실행 = 진짜 1차 실측(Lane A for Gemini).
  - 멀티모달(--add-dir)로 이미지 입력 → 이미지 AI 결과 비교에도 사용.
  - API 키·토큰 과금 없음(구독 OAuth). 제약 = rate-limit(exit 53)·미구성(exit 41).

불변식:
  - 묵묵 통과 0: 비정상 종료·빈출력·timeout은 GeminiError로 승격(빈 결과를 정상인 척 반환 금지).
  - 어댑터는 실측 신뢰의 기반 — 예외를 삼키지 않는다(호출자가 skip/warn으로 매핑).

호출 경로(firsthand):
  agy --model "Gemini 3.1 Pro (High)" -p "<프롬프트>" --print-timeout <Ns>
  agy --add-dir <frames-dir> --model ... -p "..."   # 멀티모달(이미지 입력)
bin 탐색: ANTIGRAVITY_CLI_BIN env → shutil.which("agy").
"""
import os
import re
import shutil
import subprocess
import tempfile

DEFAULT_MODEL = "Gemini 3.1 Pro (High)"

# agy 종료코드 의미(firsthand recon): 53=rate-limit · 41=provider-not-configured.
_EXIT_RATE_LIMIT = 53
_EXIT_NOT_CONFIGURED = 41

# ANSI 이스케이프 제거(색상·커서·private-mode '?' 포함).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class GeminiError(RuntimeError):
    """
    agy 호출 실패. kind로 호출자가 상태에 매핑:
      "missing"        → skip  (bin 부재 — 정직한 미검증)
      "not_configured" → skip  (agy exit 41 — provider 미구성)
      "rate_limit"     → warn  (agy exit 53 — 실측 미수행, 차단 아님)
      "nonzero"        → warn  (기타 비정상 종료)
      "empty"          → warn  (exit 0이나 빈 출력)
      "timeout"        → warn  (print-timeout/subprocess timeout)
    """

    def __init__(self, kind, message, exit_code=None):
        super().__init__(message)
        self.kind = kind
        self.exit_code = exit_code


def _find_bin():
    """agy 실행 파일 경로. ANTIGRAVITY_CLI_BIN env 우선, 없으면 PATH의 'agy'."""
    return os.environ.get("ANTIGRAVITY_CLI_BIN") or shutil.which("agy")


def gemini_available():
    """agy bin 존재 여부(호출자가 skip 분기에 사용)."""
    return _find_bin() is not None


def run_gemini(prompt, *, model=DEFAULT_MODEL, timeout=900, add_dir=None):
    """
    agy로 Gemini 호출 → 정상 출력(ANSI strip·비빈) 반환.

    실패는 전부 GeminiError로 승격:
      - bin 부재 → kind="missing"
      - exit 41 → kind="not_configured" · exit 53 → kind="rate_limit" · 기타 비정상 → kind="nonzero"
      - TimeoutExpired → kind="timeout" · 빈 출력 → kind="empty"

    add_dir 지정 시 멀티모달(이미지 디렉토리) 입력.
    timeout(초)은 agy --print-timeout으로 전달, subprocess는 +30s 여유로 감싼다.
    """
    bin_path = _find_bin()
    if not bin_path:
        raise GeminiError(
            "missing",
            "agy CLI를 찾을 수 없음 (ANTIGRAVITY_CLI_BIN 또는 PATH에 'agy' 필요)",
        )

    cmd = [bin_path, "--model", model]
    if add_dir:
        cmd += ["--add-dir", add_dir]
    cmd += ["-p", prompt, "--print-timeout", f"{timeout}s"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired as e:
        raise GeminiError("timeout", f"agy 타임아웃({timeout}s 초과): {e}") from e

    if result.returncode != 0:
        rc = result.returncode
        err = (result.stderr or result.stdout or "").strip()[:300] or "출력 없음"
        if rc == _EXIT_NOT_CONFIGURED:
            raise GeminiError("not_configured", f"agy provider 미구성 (exit 41): {err}", rc)
        if rc == _EXIT_RATE_LIMIT:
            raise GeminiError("rate_limit", f"agy rate-limit (exit 53): {err}", rc)
        raise GeminiError("nonzero", f"agy 비정상 종료 (exit {rc}): {err}", rc)

    clean = _ANSI_RE.sub("", result.stdout).strip()
    if not clean:
        raise GeminiError("empty", "agy 응답이 비어 있음 (exit 0이나 stdout 공백)")
    return clean
