# -*- coding: utf-8 -*-
"""실촬영 마커 — 하네스가 「지금이 그 순간」을 한 줄로 알린다.

왜 필요한가(2026-09-16 실측): 촬영기(`live_capture_runner.py`)는 시계로만 찍는다. 금지어 벤치
첫 실행의 터미널 기록 전체에서 `위반|실패|오류|ERROR` 에 걸린 줄은 **4줄뿐이고 전부
`"infra_errors": 0`**(=오류 0건이라는 줄)이었다. 실제 위반 5회는 화면에 흔적이 없다 —
`[31/48] B05/ban5/back elapsed=4.7s` 만 흐른다. 위반인지는 **채점해야** 알고 채점은 하네스만 한다.
그래서 「이슈 생기면 찍어라」 같은 범용 문자열로는 영원히 못 찍는다.

규약: 종류는 닫힌 어휘로 고정하고, 설명은 그 실험의 말로 쓴다.

    [LIVE:break] 첫 위반 — B02/ban5/front · 금지어 "죄송" 이 그대로 나왔다
    [LIVE:agg]   집계 완료 — 위반 5건 / 지시회차 33회

종류(KINDS):
  break — 재고 있는 성질이 깨진 순간(위반·오답·누락·방어 뚫림·죽은 링크·유출)
  turn  — 조건이 바뀌는 지점(팔·모델·단계 전환)
  agg   — 집계 결과가 화면에 뜨는 순간
  infra — 측정 자체가 흔들린 순간(타임아웃·재시도·빈 응답)

마커는 stdout 한 줄일 뿐이라 측정값을 바꾸지 않는다.
"""
import sys

KINDS = ("break", "turn", "agg", "infra")


def mark(kind, text):
    """`[LIVE:<kind>] <text>` 한 줄을 즉시 흘린다(촬영기가 이 줄을 보고 그 화면을 찍는다)."""
    if kind not in KINDS:
        raise ValueError(f"모르는 마커 종류: {kind!r} (허용 {KINDS})")
    print(f"[LIVE:{kind}] {text}", flush=True)
    sys.stdout.flush()
