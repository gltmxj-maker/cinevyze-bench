#!/usr/bin/env python3
"""Render publication charts from the format benchmark aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"

FORMAT_ORDER = ("json", "csv", "markdown")
FORMAT_LABELS = {"json": "JSON", "csv": "CSV", "markdown": "마크다운 표"}


def set_font() -> None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SECONDARY, length=0, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def metric(aggregate: dict, output_format: str, name: str) -> tuple[float, float, float, int, int]:
    row = aggregate["formats"][output_format][name]
    return row["rate"] * 100, row["wilson_low"] * 100, row["wilson_high"] * 100, row["success"], row["total"]


def plot_success(aggregate: dict, output: Path) -> None:
    x = np.arange(len(FORMAT_ORDER))
    width = 0.32
    parse = [metric(aggregate, fmt, "parse") for fmt in FORMAT_ORDER]
    final = [metric(aggregate, fmt, "success") for fmt in FORMAT_ORDER]

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=SURFACE)
    style_axis(ax)
    parse_values = [row[0] for row in parse]
    final_values = [row[0] for row in final]
    parse_err = np.array([[max(0.0, row[0] - row[1]) for row in parse],
                          [max(0.0, row[2] - row[0]) for row in parse]])
    final_err = np.array([[max(0.0, row[0] - row[1]) for row in final],
                          [max(0.0, row[2] - row[0]) for row in final]])

    bars_parse = ax.bar(x - width / 2, parse_values, width, color=BLUE, label="엄격 파싱 성공",
                        yerr=parse_err, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": SECONDARY}, zorder=3)
    bars_final = ax.bar(x + width / 2, final_values, width, color=ORANGE, label="파싱+스키마+값 모두 성공",
                        yerr=final_err, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": SECONDARY}, zorder=3)

    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_xticks(x, [FORMAT_LABELS[fmt] for fmt in FORMAT_ORDER], fontsize=13, color=INK)
    ax.set_ylabel("성공률 (각 형식 60회)", fontsize=11, color=SECONDARY, labelpad=12)
    ax.set_title("형식은 읽혔지만, 값까지 맞은 비율은 달랐습니다", loc="left", fontsize=22,
                 fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.01, "gemma3:4b · 한국어 60사례 × 3형식 · 막대 오차선은 95% Wilson 구간",
            transform=ax.transAxes, fontsize=11, color=SECONDARY, va="bottom")
    ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=10)

    for bars, rows in ((bars_parse, parse), (bars_final, final)):
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4.5,
                    f"{row[3]}/{row[4]}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=INK)

    fig.text(0.075, 0.03, "출처: cinevyze 자체 실측 · 2026-07-29 · 자동 복구 없이 응답 전체를 엄격 파싱",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.96))
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_failures(aggregate: dict, output: Path) -> None:
    value_failures = []
    structural_failures = []
    for fmt in FORMAT_ORDER:
        counts = aggregate["formats"][fmt]["failure_counts"]
        value_failures.append(counts.get("value", 0))
        structural_failures.append(sum(count for reason, count in counts.items() if reason != "value"))

    y = np.arange(len(FORMAT_ORDER))
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=SECONDARY, length=0, labelsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    bars_value = ax.barh(y, value_failures, color=AQUA, height=0.42, label="값 오류", zorder=3)
    bars_structure = ax.barh(y, structural_failures, left=value_failures, color=VIOLET,
                             height=0.42, label="구조·행·스키마 오류", zorder=3)
    ax.set_yticks(y, [FORMAT_LABELS[fmt] for fmt in FORMAT_ORDER], fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xticks(range(0, 41, 5))
    ax.set_xlabel("실패 시행 수 (각 형식 60회)", fontsize=11, color=SECONDARY, labelpad=12)
    ax.set_title("마크다운 표의 추가 실패는 대부분 ‘형식 이탈’이었습니다", loc="left",
                 fontsize=22, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.01, "JSON·CSV는 주로 값이 틀렸고, 마크다운 표는 JSON으로 바꾸거나 행을 덧붙이는 실패가 15건",
            transform=ax.transAxes, fontsize=11, color=SECONDARY, va="bottom")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.96), ncols=2, fontsize=10)

    for bars, values in ((bars_value, value_failures), (bars_structure, structural_failures)):
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        str(value), ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if value >= 3 else INK)
    totals = [a + b for a, b in zip(value_failures, structural_failures)]
    for index, total in enumerate(totals):
        ax.text(total + 0.7, index, f"총 {total}건", va="center", fontsize=10.5, color=SECONDARY)

    fig.text(0.075, 0.03, "구조 실패 = syntax·row_count·schema·wrapper_noise 합계. 부분 추출이나 자동 복구는 하지 않음.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.96))
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    set_font()
    plot_success(aggregate, args.output_dir / "format-parse-success.png")
    plot_failures(aggregate, args.output_dir / "format-failure-types.png")
    print(args.output_dir / "format-parse-success.png")
    print(args.output_dir / "format-failure-types.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
