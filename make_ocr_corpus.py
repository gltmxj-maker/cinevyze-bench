# -*- coding: utf-8 -*-
"""make_ocr_corpus.py — OCR 실측용 한국어 이미지 코퍼스 결정론 생성기(커밋 재현물).

corpus/ocr/ 는 gitignore(assets 동급)라 이 스크립트가 재현성의 근거다. 동일 참조 문장을
6가지 '화질 조건'(깨끗/작은글씨/저해상도/흐림/노이즈/회전)으로 렌더 → clip-01..06.png + refs.json.
같은 텍스트를 조건만 바꿔 렌더 = CER 차이가 '글자 난이도'가 아니라 '화질 열화' 때문임을 격리.

정답(ground truth) = 우리가 렌더한 그 문자열(정직·자명). CC 라이선스 데이터 아님(자체 생성).
난수(노이즈)는 seed 고정 = 재현 가능. 폰트 = 시스템 Noto Sans CJK(재배포 안 함).

사용: python3 make_ocr_corpus.py    → corpus/ocr/ 재생성(idempotent)
"""
import os, glob, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import bench_config as cfg

OUT = os.path.join(cfg.BASE_DIR, "corpus", "ocr")
SEED = 42

# 실제 문서에 가까운 혼합(한글 + 숫자 2026·1,234 + 영문 OCR·AI + 문장부호). 5줄 ≈ 290B(≥200B 게이트 여유).
REF_LINES = [
    "인공지능 기술이 2026년 들어 매우 빠르게 발전하고 있다.",
    "누구나 무료로 쓸 수 있는 OCR 도구가 이미지 속 한국어 글자를",
    "얼마나 정확하게 읽어내는지 우리가 직접 측정해 보았다.",
    "테스트 문장에는 숫자 1,234와 영문 단어 AI, 그리고 여러",
    "문장 부호가 자연스럽게 섞여 있어 실제 문서와 비슷하다.",
]
REF = " ".join(REF_LINES)

# (id, 조건키, 한글설명) — 난이도 그라디언트(붕괴점을 실제로 보이게). 출력이 비지 않는 선.
CONDITIONS = [
    ("clean",  "깨끗한 인쇄체(기준)"),
    ("small",  "작은 글씨(13px)"),
    ("lowres", "저해상도(0.28x 왕복)"),
    ("blur",   "흐림(가우시안 2.8)"),
    ("noise",  "픽셀 노이즈(σ48)"),
    ("rotate", "기울기(11°)"),
]


def _font(size):
    pats = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/**/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/**/NotoSansCJK*.ttc"]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            # index 1 = KR face(한글 글리프는 어느 face든 동일하나 KR 우선)
            try:
                return ImageFont.truetype(hits[0], size, index=1)
            except Exception:
                return ImageFont.truetype(hits[0], size)
    raise SystemExit("Noto Sans CJK 폰트를 찾을 수 없음 — fc-list로 확인.")


def _render_base(font_size=30, pad=44, line_gap=16):
    """참조 5줄을 흰 배경/검은 글씨로 렌더한 기준 이미지(RGB)."""
    font = _font(font_size)
    tmp = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(tmp)
    widths, h = [], 0
    for ln in REF_LINES:
        bb = d.textbbox((0, 0), ln, font=font)
        widths.append(bb[2] - bb[0])
        h = max(h, bb[3] - bb[1])
    W = max(widths) + pad * 2
    H = pad * 2 + len(REF_LINES) * h + (len(REF_LINES) - 1) * line_gap
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = pad
    for ln in REF_LINES:
        d.text((pad, y), ln, fill="black", font=font)
        y += h + line_gap
    return img


def _degrade(cond, rng):
    if cond == "clean":
        return _render_base(30)
    if cond == "small":
        return _render_base(13, pad=22, line_gap=7)
    if cond == "lowres":
        base = _render_base(30)
        w, h = base.size
        small = base.resize((int(w * 0.28), int(h * 0.28)), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)
    if cond == "blur":
        return _render_base(30).filter(ImageFilter.GaussianBlur(2.8))
    if cond == "noise":
        base = _render_base(30)
        arr = np.asarray(base).astype(np.int16)
        arr += rng.normal(0, 48, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cond == "rotate":
        return _render_base(30).rotate(11, expand=True, fillcolor="white", resample=Image.BICUBIC)
    raise ValueError(cond)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    refs = {}
    for i, (cond, desc) in enumerate(CONDITIONS, 1):
        img = _degrade(cond, rng)
        name = "clip-%02d.png" % i
        img.save(os.path.join(OUT, name))
        refs[name] = {"id": "OCR-%02d" % i, "condition": cond, "desc": desc,
                      "ref": REF, "ref_bytes": len(REF.encode("utf-8")),
                      "img_size": "%dx%d" % img.size}
        print(f"  {name}  {cond:8} {img.size}  ref_bytes={refs[name]['ref_bytes']}")
    with open(os.path.join(OUT, "refs.json"), "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "SOURCE.md"), "w", encoding="utf-8") as f:
        f.write(
            "# corpus/ocr — OCR 실측 이미지(자체 생성·재현물)\n\n"
            "`make_ocr_corpus.py`(seed=%d)가 결정론적으로 렌더. corpus/는 gitignore라 이 스크립트가 SSOT.\n\n"
            "- **정답(ground truth)** = 렌더한 참조 문자열(refs.json['ref']) — 우리가 그린 것이라 자명.\n"
            "- **조건** = 동일 텍스트를 6가지 화질(clean/small/lowres/blur/noise/rotate)로 열화 → 화질 효과 격리.\n"
            "- **폰트** = 시스템 Noto Sans CJK(재배포 안 함). 난수(노이즈)=seed 고정.\n"
            "- **캐비앗**: 렌더 인쇄체 = 실제 사진·손글씨·복잡 배경보다 쉬움(정직 한계).\n\n"
            "참조 문장:\n\n> %s\n" % (SEED, REF))
    print(f"[done] {OUT}  (clips={len(CONDITIONS)}, ref_bytes={len(REF.encode('utf-8'))})")


if __name__ == "__main__":
    main()
