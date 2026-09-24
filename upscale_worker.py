#!/usr/bin/env python
"""업스케일 워커 — ComfyUI .venv(cu130·CUDA)에서만 실행되는 GPU 작업 단위.

all-blog 하네스 파이썬(torch=rocm·GPU 미인식)에서 직접 못 돌리므로,
UpscaleAdapter가 이 워커를 ComfyUI .venv 파이썬으로 subprocess 호출한다(ComfyUIAdapter가
HTTP로 우회하는 것과 동일 취지·access=local·자체 구동·구독/계정 무관).

usage: python upscale_worker.py <input_png> <method> <output_png>
  method ∈ {lanczos, apisr2x, apisr4x}
stdout 마지막 줄: META:{json}  (in/out res·scale·model_file·t_load_s·t_infer_s·device·out_bytes)
"""
import sys, os, json, time
import numpy as np
from PIL import Image

# APISR 오픈 업스케일 모델(MCF가 SSD에 둔 기설치·재설치/복사 0)
MODELS = {
    "apisr4x": os.environ.get("APISR4X_PATH", "./models/4x_APISR_GRL_GAN_generator.pth"),
    "apisr2x": os.environ.get("APISR2X_PATH", "./models/2x_APISR_RRDB_GAN_generator.pth"),
}


def main():
    if len(sys.argv) != 4:
        print("usage: upscale_worker.py <input_png> <method> <output_png>", file=sys.stderr)
        sys.exit(2)
    inp, method, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    img = Image.open(inp).convert("RGB")
    in_w, in_h = img.size

    if method == "lanczos":
        # 고전 베이스라인(×4·CPU·모델 없음)
        t0 = time.monotonic()
        out = img.resize((in_w * 4, in_h * 4), Image.LANCZOS)
        t_infer = round(time.monotonic() - t0, 4)
        t_load, scale, model_file, device = 0.0, 4, "PIL-Lanczos(baseline)", "cpu"
    elif method in MODELS:
        import torch
        from spandrel import ModelLoader
        path = MODELS[method]
        if not os.path.exists(path):
            print(f"모델 파일 없음: {path}", file=sys.stderr); sys.exit(3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.monotonic()
        m = ModelLoader().load_from_file(path).to(device).eval()
        t_load = round(time.monotonic() - t0, 4)
        scale = m.scale
        arr = np.asarray(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.monotonic()
        with torch.no_grad():
            o = m(ten)
        if device == "cuda":
            torch.cuda.synchronize()
        t_infer = round(time.monotonic() - t1, 4)
        o = o.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = Image.fromarray((o * 255).round().astype(np.uint8))
        model_file = os.path.basename(path)
    else:
        print(f"미지원 method: {method} (lanczos/apisr2x/apisr4x)", file=sys.stderr); sys.exit(2)

    out.save(outp)
    meta = {
        "method": method, "model_file": model_file, "scale": int(scale),
        "in_res": f"{in_w}x{in_h}", "out_res": f"{out.size[0]}x{out.size[1]}",
        "t_load_s": t_load, "t_infer_s": t_infer, "device": device,
        "out_bytes": os.path.getsize(outp), "input_path": os.path.abspath(inp),
    }
    print("META:" + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
