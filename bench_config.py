"""bench_config — 공개 저장소용 최소 설정.

원본 저장소의 운영 설정 모듈은 공개하지 않는다. 하네스가 실제로 쓰는
심볼만 여기 둔다: 출력 폴더와 파일명 안전화.
"""
from __future__ import annotations

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADSENSE_TEST_RUNS_DIR = os.environ.get("BENCH_RUNS_DIR", os.path.join(BASE_DIR, "runs"))


def safe_name(s: str) -> str:
    """파일명으로 쓸 수 있게 정규화."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-").lower() or "unnamed"
