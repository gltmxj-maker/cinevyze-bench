"""SDXL 일러스트 실측의 제한적인 자동 지표. 미관·저작권·실사용 적합성은 판정하지 않는다."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from statistics import mean

from PIL import Image


def score_image(path: Path, case: dict) -> dict:
    """가장자리 네 모서리 패치의 흰 픽셀 비율만 측정한다."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        patch = max(1, min(width, height) // 16)
        boxes = ((0, 0, patch, patch), (width - patch, 0, width, patch),
                 (0, height - patch, patch, height),
                 (width - patch, height - patch, width, height))
        pixels = [pixel for box in boxes for pixel in image.crop(box).get_flattened_data()]
    fraction = sum(min(pixel) >= 240 for pixel in pixels) / len(pixels)
    requested = bool(case.get("white_background_requested"))
    threshold = float(case.get("minimum_white_corner_fraction", 0.5))
    return {
        "width": width, "height": height,
        "white_corner_fraction": round(fraction, 4),
        "white_background_proxy_pass": fraction >= threshold if requested else None,
    }


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    elapsed = [row["elapsed_s"] for row in ok if isinstance(row.get("elapsed_s"), (int, float))]
    background = [row for row in ok if row.get("white_background_proxy_pass") is not None]
    return {
        "attempted_runs": len(rows),
        "successful_images": len(ok),
        "infra_errors": sum(row.get("status") == "infra_error" for row in rows),
        "elapsed_mean_s": round(mean(elapsed), 3) if elapsed else None,
        "elapsed_min_s": min(elapsed) if elapsed else None,
        "elapsed_max_s": max(elapsed) if elapsed else None,
        "white_background_proxy_checked": len(background),
        "white_background_proxy_failures": sum(row["white_background_proxy_pass"] is False for row in background),
        "bytes_min": min((row["bytes"] for row in ok), default=None),
        "bytes_max": max((row["bytes"] for row in ok), default=None),
        "by_task": {task: {"attempted": len(group),
                           "successful": sum(row.get("status") == "ok" for row in group),
                           "infra_errors": sum(row.get("status") == "infra_error" for row in group)}
                    for task in sorted({row["task"] for row in rows})
                    for group in [[row for row in rows if row["task"] == task]]},
    }


def score_run_dir(run_dir: Path) -> dict:
    """원본 rows와 PNG를 전건 대조해 aggregate.json을 새로 쓴다."""
    rows = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cases = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
    by_task = {case["task"]: case for case in cases["cases"]}
    rescored = []
    for row in rows:
        item = dict(row)
        if item["status"] == "ok":
            path = run_dir / item["output_file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            item.update(score_image(path, by_task[item["task"]]))
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"PNG 크기 불일치: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"PNG SHA256 불일치: {path}")
        rescored.append(item)
    (run_dir / "results.json").write_text(json.dumps(rescored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = aggregate(rescored)
    (run_dir / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
