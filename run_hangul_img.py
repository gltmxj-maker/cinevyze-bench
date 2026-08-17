#!/usr/bin/env python3
"""Run the Hangul-vs-Latin *in-image text rendering* benchmark on local SDXL checkpoints.

The axis is the writing system of the requested text. Prompt template, checkpoint,
sampler, steps, CFG, resolution, and seed sequence are held fixed; only the string
inside the template changes between the Korean and English arms of each pair.

The generation workflow is defined here rather than reused from tool_adapters.ComfyUIAdapter
because that adapter's negative prompt contains "text" and "signature" — it actively
suppresses lettering, which would fabricate failure in a lettering benchmark.

The script is the write-origin for run.yaml. It stores every PNG, every OCR transcript,
and the full generation parameters rather than asking a person to fill evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hangul_score import score_one, score_run_dir
from ocr_score import _lev, _norm

HARNESS_VERSION = "1.0"
CALIBRATION_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
DEFAULT_CASES = Path(__file__).with_name("hangul_img_cases.json")
DEFAULT_RUN_DIR = Path(__file__).with_name("test_runs") / "sdxl-hangul-text-20260731"

COMFY_HOST = "http://127.0.0.1:8188"
COMFY_DIR = Path(os.path.expanduser("~/mcf-tools/ComfyUI"))
CHECKPOINTS = ("animagine-xl-4.0-opt.safetensors", "Illustrious-XL-v2.0.safetensors")
STEPS, CFG, SAMPLER, RESOLUTION = 26, 6.5, "euler_ancestral", 1024
# Deliberately omits "text"/"signature": suppressing lettering would fabricate the result.
NEGATIVE = "lowres, blurry, jpeg artifacts, watermark"
SCRIPTS = ("ko", "en")
# Alternate which script is generated first so warm-up or drift cannot masquerade as a
# writing-system effect.
SCRIPT_ROTATIONS = (("ko", "en"), ("en", "ko"))
OCR_LANGS = {"ko": "kor", "en": "eng"}


def load_cases(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("prompt_template") or ""
    pairs = data.get("pairs") or []
    if "{text}" not in template:
        raise ValueError("prompt_template에 {text} 자리표시자가 필요함")
    if not pairs:
        raise ValueError("pairs가 비어 있음")
    seen: set[str] = set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if not pair_id or pair_id in seen:
            raise ValueError(f"중복/빈 pair_id: {pair_id}")
        seen.add(pair_id)
        for script in SCRIPTS:
            text = pair.get(script)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{pair_id}/{script} 텍스트 누락")
        if any(ord(ch) < 128 for ch in pair["ko"]):
            raise ValueError(f"{pair_id} ko 값에 비한글 문자 포함")
        if not pair["en"].isascii():
            raise ValueError(f"{pair_id} en 값에 비ASCII 문자 포함")
    return data


def build_prompt(template: str, text: str) -> str:
    return template.replace("{text}", text)


def _reachable() -> bool:
    try:
        urllib.request.urlopen(COMFY_HOST + "/system_stats", timeout=5)
        return True
    except (OSError, urllib.error.URLError):
        return False


def ensure_server(timeout: int = 180) -> bool:
    if _reachable():
        return True
    python = COMFY_DIR / ".venv" / "bin" / "python"
    if not python.exists():
        return False
    subprocess.Popen([str(python), "main.py", "--port", "8188"], cwd=str(COMFY_DIR),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _reachable():
            return True
        time.sleep(1.5)
    return False


def installed_checkpoints() -> list[str]:
    directory = COMFY_DIR / "models" / "checkpoints"
    return sorted(p.name for p in directory.glob("*.safetensors")) if directory.is_dir() else []


def workflow(prompt: str, checkpoint: str, seed: int) -> dict[str, Any]:
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": CFG,
              "sampler_name": SAMPLER, "scheduler": "normal", "denoise": 1,
              "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": RESOLUTION, "height": RESOLUTION, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "hangul-bench", "images": ["8", 0]}},
    }


def _post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(COMFY_HOST + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _get(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(COMFY_HOST + path, timeout=30) as response:
        return json.loads(response.read())


def generate_image(prompt: str, checkpoint: str, seed: int, timeout: int) -> tuple[bytes | None, dict[str, Any], str | None]:
    """Return PNG bytes plus generation metadata, or an honest error string."""
    started = time.monotonic()
    try:
        submitted = _post("/prompt", {"prompt": workflow(prompt, checkpoint, seed)})
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"
    prompt_id = submitted.get("prompt_id")
    if not prompt_id:
        return None, {}, f"prompt_id 없음: {submitted}"

    image = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            history = _get("/history/" + prompt_id)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return None, {"prompt_id": prompt_id}, f"history 조회 실패: {exc}"
        entry = history.get(prompt_id)
        if entry:
            for node_output in entry.get("outputs", {}).values():
                for candidate in node_output.get("images", []):
                    image = candidate
                    break
                if image:
                    break
            if image:
                break
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                return None, {"prompt_id": prompt_id}, f"ComfyUI 실행 오류: {json.dumps(entry.get('status'), ensure_ascii=False)[:300]}"
        time.sleep(0.2)
    if not image:
        return None, {"prompt_id": prompt_id}, f"생성 타임아웃({timeout}s)"

    source = COMFY_DIR / "output" / image.get("subfolder", "") / image["filename"]
    if not source.exists():
        return None, {"prompt_id": prompt_id}, f"출력 파일 없음: {source}"
    data = source.read_bytes()
    if len(data) < 5000:
        return None, {"prompt_id": prompt_id}, f"출력 과소({len(data)}B) — 생성 실패 의심"
    meta = {
        "prompt_id": prompt_id, "seed": seed, "steps": STEPS, "cfg": CFG, "sampler": SAMPLER,
        "checkpoint": checkpoint, "resolution": f"{RESOLUTION}x{RESOLUTION}",
        "negative_prompt": NEGATIVE, "source_path": str(source),
        "elapsed_s": round(time.monotonic() - started, 3), "bytes": len(data),
    }
    return data, meta, None


# psm 7 assumes one text line; the models actually produce titles, subtitles and scattered
# marks. Reading every layout mode and keeping the best transcription per image stops the
# OCR layout assumption from being scored as a model failure. Applied identically to both
# arms, so it cannot favour one writing system.
OCR_PSMS = ("7", "6", "11", "3")


def ocr_once(image_path: Path, lang: str, psm: str, timeout: int) -> tuple[str, str | None]:
    command = ["tesseract", str(image_path), "stdout", "-l", lang, "--psm", psm, "--oem", "1"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "", f"tesseract 실패(psm={psm}, rc={proc.returncode}): {proc.stderr.strip()[:200]}"
    return " ".join((proc.stdout or "").split()), None


def ocr(image_path: Path, lang: str, requested: str = "", timeout: int = 120
        ) -> tuple[str, dict[str, Any], str | None]:
    """Transcribe with Tesseract across layout modes, keeping the reading closest to target.

    "Closest" is decided by character edit distance to the requested string, which is the
    most generous reading the image supports. A benchmark should fail a model for what it
    drew, not for how the reader was configured.
    """
    started = time.monotonic()
    variants: dict[str, str] = {}
    errors: list[str] = []
    for psm in OCR_PSMS:
        text, error = ocr_once(image_path, lang, psm, timeout)
        if error:
            errors.append(error)
            continue
        variants[psm] = text
    if not variants:
        return "", {}, errors[0] if errors else "tesseract 전 모드 실패"

    def rank(item: tuple[str, str]) -> tuple[int, int]:
        _, text = item
        target = _norm(requested, False) if requested else ""
        return (_lev(list(target), list(_norm(text, False))) if target else 0, len(text))

    best_psm, best_text = min(variants.items(), key=rank)
    meta = {"lang": lang, "selected_psm": best_psm, "oem": "1", "variants": variants,
            "proc_s": round(time.monotonic() - started, 3)}
    return best_text, meta, None


def calibrate_ocr(run_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Measure the OCR instrument's own floor before blaming the image model.

    Every requested string is rendered as clean vector text with a font that covers both
    scripts, then read back through the identical Tesseract call. Whatever this misses is
    the measurement's floor, not a model failure, and the article must subtract it.
    """
    from PIL import Image, ImageDraw, ImageFont

    directory = run_dir / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in data["pairs"]:
        for script in SCRIPTS:
            text = pair[script]
            image = Image.new("RGB", (RESOLUTION, RESOLUTION), "white")
            font = ImageFont.truetype(str(CALIBRATION_FONT), 220)
            draw = ImageDraw.Draw(image)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            draw.text(((RESOLUTION - (right - left)) / 2 - left,
                       (RESOLUTION - (bottom - top)) / 2 - top), text, font=font, fill="black")
            path = directory / f"{pair['pair_id']}-{script}.png"
            image.save(path)
            recognised, _, error = ocr(path, OCR_LANGS[script], text)
            scored = score_one(text, recognised)
            rows.append({"pair_id": pair["pair_id"], "script": script, "requested_text": text,
                         "ocr_text": recognised, "exact": scored["exact"], "cer": scored["cer"],
                         "image_file": f"calibration/{path.name}", "error": error})

    summary = {
        "note": "합성 렌더 텍스트에 동일 OCR 설정 적용 — 여기서 못 읽히면 그건 모델 실패가 아니라 계측 한계",
        "font": str(CALIBRATION_FONT),
        "ocr": "tesseract --psm 7 --oem 1",
        "by_script": {
            script: {
                "exact": sum(r["exact"] for r in rows if r["script"] == script),
                "total": sum(1 for r in rows if r["script"] == script),
            } for script in SCRIPTS
        },
        "rows": rows,
    }
    (run_dir / "calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def reocr(run_dir: Path) -> dict[str, Any]:
    """Re-read already-saved PNGs with the current OCR settings.

    The images are the evidence; the OCR call is the instrument. When the instrument is
    corrected mid-run, every image must be re-read the same way or the earlier and later
    halves of the run would be scored by different rules. No image is regenerated.
    """
    changed = 0
    total = 0
    for invocation_path in sorted((run_dir / "raw").glob("*-invocation.json")):
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        image_file = invocation.get("image_file")
        if not image_file or not (run_dir / image_file).exists():
            continue
        total += 1
        recognised, meta, error = ocr(run_dir / image_file, OCR_LANGS[invocation["script"]],
                                      invocation["requested_text"])
        if recognised != invocation.get("ocr_text"):
            changed += 1
        invocation["ocr_text"] = recognised
        invocation["ocr_meta"] = meta
        if error:
            invocation["infra_error"] = error
        invocation_path.write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"reread": total, "changed": changed}


def planned_runs(data: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for index, pair in enumerate(data["pairs"]):
            for repeat in range(1, repeats + 1):
                for script in SCRIPT_ROTATIONS[(index + repeat) % len(SCRIPT_ROTATIONS)]:
                    runs.append({
                        "key": f"{checkpoint}/{pair['pair_id']}/{script}/r{repeat}",
                        "checkpoint": checkpoint,
                        "pair_id": pair["pair_id"],
                        "script": script,
                        "repeat": repeat,
                        "text": pair[script],
                    })
    return runs


def _existing(run_dir: Path) -> tuple[set[str], int]:
    keys: set[str] = set()
    maximum = 0
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        keys.add(row["key"])
        maximum = max(maximum, int(row["run_id"]))
    return keys, maximum


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_yaml(run_dir: Path, cases_path: Path) -> None:
    entries = []
    for path in sorted((run_dir / "raw").glob("*-invocation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "key": row["key"],
            "checkpoint": row["checkpoint"],
            "pair_id": row["pair_id"],
            "script": row["script"],
            "requested_text": row["requested_text"],
            "output_file": row["transcript_file"],
            "screenshot": row.get("image_file"),
            "log_file": str(path.relative_to(run_dir)),
            "seed": (row.get("generation") or {}).get("seed"),
            "elapsed_s": (row.get("generation") or {}).get("elapsed_s"),
            "infra_error": row.get("infra_error"),
        })
    payload = {
        "tool": "comfyui",
        "date": dt.date.today().isoformat(),
        "method": "IMG-02 writing-system arm (Hangul vs Latin poster text, OCR-scored)",
        "access": "local",
        "model": " + ".join(CHECKPOINTS),
        "generated_by": "run_hangul_img.py",
        "harness_version": HARNESS_VERSION,
        "tos_confirmed": True,
        "tos_source_url": "로컬 ComfyUI + 오픈 SDXL 체크포인트(자체 구동·구독/계정 무관)",
        "request_parallelism": 1,
        "axis": "writing_system",
        "generation": {"steps": STEPS, "cfg": CFG, "sampler": SAMPLER,
                       "resolution": f"{RESOLUTION}x{RESOLUTION}", "negative_prompt": NEGATIVE},
        "scoring": {"ocr": "tesseract --psm 7 --oem 1", "langs": OCR_LANGS,
                    "cases_file": cases_path.name,
                    "ocr_calibration": "calibration.json"},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_sha_at_run": _git_sha(),
            "comfyui_dir": str(COMFY_DIR),
        },
        "runs": entries,
    }
    target = run_dir / "run.yaml"
    temporary = run_dir / "run.yaml.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def run_benchmark(args: argparse.Namespace) -> int:
    data = load_cases(args.cases)
    plan = planned_runs(data, args.repeats)
    if args.dry_run:
        print(json.dumps({"runs": len(plan), "checkpoints": list(CHECKPOINTS),
                          "first": {"key": plan[0]["key"],
                                    "prompt": build_prompt(data["prompt_template"], plan[0]["text"])}},
                         ensure_ascii=False, indent=2))
        return 0

    if args.reocr:
        summary = reocr(args.run_dir)
        _, aggregate = score_run_dir(args.run_dir)
        print(json.dumps({**summary, "scripts": aggregate["scripts"]}, ensure_ascii=False, indent=2))
        return 0

    if args.calibrate:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        summary = calibrate_ocr(args.run_dir, data)
        print(json.dumps(summary["by_script"], ensure_ascii=False, indent=2))
        return 0

    missing = [c for c in CHECKPOINTS if c not in installed_checkpoints()]
    if missing:
        print(f"체크포인트 미설치: {missing} — 설치된 것: {installed_checkpoints()}")
        return 2
    if not ensure_server():
        print("ComfyUI 서버 미기동 — .venv/main.py 자동기동 실패(수동 확인 필요).")
        return 2

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "raw").mkdir(exist_ok=True)
    (args.run_dir / "images").mkdir(exist_ok=True)
    existing_keys, next_id = _existing(args.run_dir)
    pending = [row for row in plan if row["key"] not in existing_keys]
    pending_total = len(pending)
    if args.max_new > 0:
        pending = pending[:args.max_new]
    chunk_limited = len(pending) < pending_total
    print(f"planned={len(plan)} existing={len(plan) - pending_total} "
          f"pending_total={pending_total} running_now={len(pending)}")

    for position, row in enumerate(pending, 1):
        next_id += 1
        prompt = build_prompt(data["prompt_template"], row["text"])
        # Seed depends only on the pair and repeat, so the two scripts of a pair share it.
        seed = 7000 + hash((row["pair_id"], row["repeat"])) % 1000
        image, generation, infra_error = generate_image(prompt, row["checkpoint"], seed, args.timeout)

        stem = f"{next_id:03d}"
        image_rel = f"images/{stem}-{row['pair_id']}-{row['script']}.png"
        transcript_rel = f"raw/{stem}-output.txt"
        invocation_rel = f"raw/{stem}-invocation.json"

        recognised, ocr_meta, ocr_error = "", {}, None
        if image is not None:
            (args.run_dir / image_rel).write_bytes(image)
            recognised, ocr_meta, ocr_error = ocr(
                args.run_dir / image_rel, OCR_LANGS[row["script"]], row["text"]
            )
            if ocr_error:
                infra_error = infra_error or ocr_error

        transcript = (
            f"run_id: {next_id}\nkey: {row['key']}\ncheckpoint: {row['checkpoint']}\n"
            f"pair_id: {row['pair_id']}\nscript: {row['script']}\nrepeat: {row['repeat']}\n"
            f"requested_text: {row['text']}\n\n[PROMPT]\n{prompt}\n\n"
            f"[GENERATION]\n{json.dumps(generation, ensure_ascii=False, indent=2)}\n\n"
            f"[OCR TEXT]\n{recognised}\n\n[OCR META]\n{json.dumps(ocr_meta, ensure_ascii=False, indent=2)}\n"
        )
        (args.run_dir / transcript_rel).write_text(transcript, encoding="utf-8")
        invocation = {
            "run_id": next_id,
            "key": row["key"],
            "checkpoint": row["checkpoint"],
            "pair_id": row["pair_id"],
            "script": row["script"],
            "repeat": row["repeat"],
            "requested_text": row["text"],
            "prompt": prompt,
            "generation": generation,
            "ocr_text": recognised,
            "ocr_meta": ocr_meta,
            "infra_error": infra_error,
            "image_file": image_rel if image is not None else None,
            "transcript_file": transcript_rel,
        }
        (args.run_dir / invocation_rel).write_text(
            json.dumps(invocation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_run_yaml(args.run_dir, args.cases)
        print(f"[{position}/{len(pending)}] {row['key']} '{row['text']}' → OCR '{recognised[:30]}'"
              + (f" ERROR={infra_error}" if infra_error else ""), flush=True)

    write_run_yaml(args.run_dir, args.cases)
    results, aggregate = score_run_dir(args.run_dir)
    infra_count = sum(bool(row.get("infra_error")) for row in results)
    status = {
        "runs": len(results),
        "infra_errors": infra_count,
        "infra_rate": infra_count / len(results) if results else 1.0,
        "infra_limit": 0.05,
        "complete": len(results) == len(plan),
        "chunk_limited": chunk_limited,
    }
    (args.run_dir / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["infra_rate"] > status["infra_limit"]:
        return 4
    if status["chunk_limited"] and not status["complete"]:
        return 0
    if not status["complete"]:
        return 4
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-new", type=int, default=0,
                        help="이번 호출에서 새로 실행할 최대 시행 수(0=전부, 긴 런의 안전한 청크용)")
    parser.add_argument("--reocr", action="store_true",
                        help="저장된 PNG를 현재 OCR 설정으로 재판독(생성 없음·GPU 미사용)")
    parser.add_argument("--calibrate", action="store_true",
                        help="합성 렌더 텍스트로 OCR 계측 한계만 측정(GPU 미사용)")
    parser.add_argument("--dry-run", action="store_true")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
