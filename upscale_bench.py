#!/usr/bin/env python
"""업스케일 비교 벤치 — tested 글의 표(PSNR/SSIM/시간)를 받치는 재현가능 측정 아티팩트 생성.

★voice=tested 무결성(적대검증 2026-06-27 GLM/MiMo 지적): PSNR/SSIM·warm 추론시간이
임시 스크립트가 아니라 *커밋된 재현 스크립트*에서 나오고, 결과를 run 디렉터리에
comparison.json으로 남겨 추적 가능해야 한다. (하네스 run.yaml = 업스케일 실행 증거,
이 스크립트 = 그 출력에 대한 PSNR/SSIM 분석 + cold/warm 타이밍 분리.)

ComfyUI .venv(cu130·CUDA)에서 실행:
  ~/mcf-tools/ComfyUI/.venv/bin/python upscale_bench.py
"""
import time, json, os, sys
import numpy as np
from PIL import Image
import torch
from spandrel import ModelLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus/ups")
OUTDIR = os.path.join(BASE, "test_runs/upscale-apisr4x-20260627")
APISR4X = os.environ.get("APISR4X_PATH", "./models/4x_APISR_GRL_GAN_generator.pth")
IMGS = [("ups01", "마스코트(플랫·볼드)"), ("ups02", "차트(선·디테일)"), ("ups03", "복잡한 일러스트")]
DOWNSCALE = "Lanczos"   # 원본 1024 → 256 입력 만들 때 쓴 방식(재현성·적대검증 DeepSeek 지적)


def metrics(orig_np, img):
    a = np.asarray(img)
    return round(psnr(orig_np, a, data_range=255), 2), round(ssim(orig_np, a, channel_axis=2, data_range=255), 4)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"

    # cold = fresh 프로세스에서 모델 로드 시간(첫 장 체감). 여기선 이 프로세스의 로드시간 측정.
    t0 = time.monotonic()
    model = ModelLoader().load_from_file(APISR4X).to(dev).eval()
    model_load_s = round(time.monotonic() - t0, 2)

    def apisr(low):
        arr = np.asarray(low).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.monotonic()
        with torch.no_grad():
            o = model(ten)
        if dev == "cuda": torch.cuda.synchronize()
        dt = round((time.monotonic() - t) * 1000, 1)   # ms
        o = o.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((o * 255).round().astype(np.uint8)), dt

    rows = []
    for sid, kind in IMGS:
        orig = Image.open(f"{CORPUS}/{sid}-orig1024.png").convert("RGB")
        low = Image.open(f"{CORPUS}/{sid}-input256.png").convert("RGB")
        o = np.asarray(orig)
        # Lanczos baseline (CPU·즉시)
        t = time.monotonic(); lanc = low.resize((1024, 1024), Image.LANCZOS); lanc_ms = round((time.monotonic()-t)*1000, 1)
        lp, ls = metrics(o, lanc)
        # APISR 4x (warm GPU 추론)
        ap, ap_ms = apisr(low)
        ap_p, ap_s = metrics(o, ap)
        rows.append({"img": sid, "kind": kind,
                     "lanczos": {"psnr": lp, "ssim": ls, "infer_ms": lanc_ms},
                     "apisr4x": {"psnr": ap_p, "ssim": ap_s, "infer_ms": ap_ms}})

    n = len(rows)
    avg = lambda k, m: round(sum(r[m][k] for r in rows)/n, 2 if k != "ssim" else 4)
    summary = {
        "model": "APISR 4x (4x_APISR_GRL_GAN_generator.pth · GRL generator, GAN 학습)",
        "baseline": "PIL Lanczos ×4",
        "device": gpu, "downscale_method": DOWNSCALE,
        "sample_n": n, "sample_note": "SDXL 자가생성 일러스트 3장(애니/일러스트 도메인). 사진·실사 미포함 — 일반화 주의.",
        "first_image_cold_s": model_load_s,   # 첫 장 = 모델 로딩 포함(이 프로세스 기준)
        "warm_infer_ms": {"lanczos_avg": avg("infer_ms", "lanczos"), "apisr4x_avg": avg("infer_ms", "apisr4x")},
        "avg_psnr": {"lanczos": avg("psnr", "lanczos"), "apisr4x": avg("psnr", "apisr4x")},
        "avg_ssim": {"lanczos": avg("ssim", "lanczos"), "apisr4x": avg("ssim", "apisr4x")},
        "rows": rows,
        "method": "원본 1024px → 256px(Lanczos 다운스케일) 입력 → 1024px 복원 → 원본 대비 PSNR/SSIM. 측정시점 2026-06.",
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nsaved", os.path.join(OUTDIR, "comparison.json"))


if __name__ == "__main__":
    main()
