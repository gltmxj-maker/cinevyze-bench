# cinevyze bench — 한국어 LLM 실측 하네스

[cinevyze.com](https://www.cinevyze.com) 에 올린 측정 글들의 **실행 코드와 채점기**입니다.
글에 적힌 숫자를 직접 재현하라고 공개합니다.

- 각 항목은 `run_*.py`(실행)와 `*_score.py`(채점)가 한 쌍입니다.
- 케이스 파일(`*_bench_cases*.json`)에 문항이 들어 있습니다.
- 개인정보 문항은 **합성 데이터**입니다(`.example` / `.test` 예약 TLD).

## 실행

```bash
python3 run_<name>_bench.py --help
python3 <name>_score.py --help
```

모델 경로 등 기계마다 다른 값은 환경변수로 받습니다 — `QUANT_MODELS_DIR`, `APISR4X_PATH`.

## 하네스 ↔ 글

| 항목 | 코드 | 글 |
|---|---|---|
| `automation` | `automation_bench.py` | [해당 글](https://www.cinevyze.com/2026/06/ai-automate-repetitive-excel-task-2026.html) |
| `banword` | `banword_bench_cases.json` · `banword_score.py` · `run_banword_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-banned-word-instruction-compliance-tested-2026_07952645.html) |
| `format` | `format_bench_cases.json` · `format_score.py` · `plot_format_bench.py` · `run_format_bench.py` | [해당 글](https://www.cinevyze.com/2026/07/ai-output-format-json-csv-markdown-parsing-tested-2026.html) |
| `hangul` | `hangul_img_cases.json` · `hangul_score.py` · `run_hangul_img.py` | [해당 글](https://www.cinevyze.com/2026/07/sdxl-hangul-text-in-image-rendering-tested-2026.html) |
| `injection` | `injection_bench_cases.json` · `injection_score.py` · `run_injection_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/prompt-injection-defense-wording-tested-2026.html) |
| `lang` | `lang_score.py` · `run_lang_bench.py` | [해당 글](https://www.cinevyze.com/2026/07/ai-prompt-korean-vs-english-instruction-language-tested-2026.html) |
| `length` | `length_bench_cases.json` · `length_score.py` · `run_length_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-length-instruction-compliance-tested-2026.html) |
| `linkcite` | `linkcite_bench_cases.json` · `linkcite_score.py` · `run_linkcite_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-citation-link-survival-tested-2026.html) |
| `multiturn` | `multiturn_bench_cases.json` · `multiturn_bench_cases_control.json` · `multiturn_score.py` · `run_multiturn_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-multi-turn-instruction-persistence-tested-2026.html) |
| `needle` | `needle_bench_cases.json` · `needle_score.py` · `run_needle_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-long-document-recall-position-tested-2026.html) |
| `numunit` | `numunit_bench_cases.json` · `numunit_score.py` · `run_numunit_bench.py` | *(미발행 또는 미매칭)* |
| `orderbias` | `orderbias_bench_cases.json` · `orderbias_score.py` · `run_orderbias_bench.py` | *(미발행 또는 미매칭)* |
| `pii` | `pii_bench_cases.json` · `pii_score.py` · `run_pii_bench.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-korean-pii-masking-leak-rate-tested-2026.html) |
| `quant` | `run_quant.py` | [해당 글](https://www.cinevyze.com/2026/07/local-ai-quantization-q4-vs-q8-korean-tested-2026.html) |
| `rag` | `rag_bench.py` | [해당 글](https://www.cinevyze.com/2026/07/local-rag-korean-documents-tested-2026.html) |
| `repro` | `repro_score.py` · `run_repro.py` | [해당 글](https://www.cinevyze.com/2026/07/ai-same-prompt-different-answer-reproducibility-tested-2026.html) |
| `selfgrade` | `run_selfgrade_bench.py` · `selfgrade_bench_cases.json` · `selfgrade_score.py` | [해당 글](https://www.cinevyze.com/2026/08/ai-self-grading-reliability-tested-2026.html) |
| `translate` | `translate_bench.py` | [해당 글](https://www.cinevyze.com/2026/06/offline-local-ai-translation-gemma3-2026.html) |
| `upscale` | `upscale_bench.py` | [해당 글](https://www.cinevyze.com/2026/06/ai-image-upscaler-tested-2026.html) |

## 같은 방법론으로 측정한 다른 글

위 표의 글은 이 저장소에 실행 코드가 들어 있습니다.
아래 글들은 같은 원칙(직접 돌리고, 측정하지 않은 것은 측정한 것처럼 쓰지 않는다)으로 썼지만
아직 코드가 이 저장소에 올라와 있지 않습니다.

### AI 이미지·디자인

- [로컬 비전 AI, 사진을 얼마나 '이해'하나 실측 — 한글 메뉴·차트는 척척, 도형 세기는 헛발 (2026년 7월)](https://www.cinevyze.com/2026/07/local-vision-ai-image-understanding-tested-2026.html)
- [AI 썸네일 만들기 — 무료로 어디까지 되나, 그리고 한글은 제대로 박히나 (2026.7)](https://www.cinevyze.com/2026/07/korean-ai-thumbnail-makers-free-vs-hangul-compared-2026_01357532111.html)
- [AI로 PPT 만들기, 어디까지 공짜인가 — 감마·미리캔버스·코파일럿 '무료의 함정' (2026년 7월)](https://www.cinevyze.com/2026/07/ai-ppt-maker-free-vs-paid-gamma-miricanvas-2026.html)
- [AI 영상편집, 뭐가 공짜고 뭐가 돈 나가나 — 캡컷 vs 브루 솔직 정리 (2026)](https://www.cinevyze.com/2026/07/ai-video-editing-free-vs-paid-capcut-vrew-2026.html)
- [Canva AI, 뭐가 공짜고 뭐가 크레딧을 잡아먹나 — 요금·한도 솔직 정리 (2026)](https://www.cinevyze.com/2026/07/canva-ai-credits-free-vs-paid-2026.html)
- [무료로 이미지 배경 제거(누끼) — 로컬 rembg 직접 돌려본 속도·품질 (2026)](https://www.cinevyze.com/2026/07/free-local-background-removal-rembg-test-2026.html)
- [공짜로 내 PC에서 블로그 일러스트 뽑기 — 로컬 AI 이미지 생성(SDXL) 직접 돌려본 실측 (2026.6)](https://www.cinevyze.com/2026/06/local-ai-image-generation-sdxl-test-2026.html)

### AI 생산성·자동화

- [AI 회의록 정리, 무료 한도보다 한국어 지원부터 봐야 합니다 (2026.7)](https://www.cinevyze.com/2026/07/korean-ai-meeting-notes-free-limits-and-privacy-2026.html)
- [무료 로컬 AI, 내 그래픽카드론 몇 B까지 돌까 — qwen2.5 0.5B~14B VRAM·속도·GPU 등급 실측 (2026)](https://www.cinevyze.com/2026/07/local-ai-model-size-vram-gpu-ladder-tested-2026.html)
- [무료 로컬 코드 AI, 전용 모델이 꼭 필요할까 — qwen2.5-coder vs gemma3 내 PC 실측 (2026)](https://www.cinevyze.com/2026/07/local-code-ai-qwen-coder-vs-gemma3-tested-2026.html)
- [딥리서치 AI 비교 — 무료로 어디까지 되나, 그리고 그 보고서를 믿어도 되나 (2026.7)](https://www.cinevyze.com/2026/07/deep-research-ai-compared-free-limits-and-trust-2026.html)
- [AI 콘텐츠 제작 워크플로 — 대본부터 영상까지 어디까지 공짜로? '무료의 함정'과 돈 새는 지점 (2026)](https://www.cinevyze.com/2026/07/ai-content-creation-workflow-free-vs-paid-2026.html)
- [NotebookLM 활용법 — 답을 내 자료에 묶는 AI, 무료로 어디까지 (2026)](https://www.cinevyze.com/2026/07/notebooklm-how-to-use-free-vs-paid-2026.html)
- [Cursor vs GitHub Copilot, 뭘 공짜로 뭘 돈 내고 쓰나 — 2026 '크레딧 과금'으로 바뀐 계산법](https://www.cinevyze.com/2026/07/cursor-vs-copilot-free-vs-paid-2026.html)
- [무료 OCR로 한국어 이미지 글자, 얼마나 읽나 — CPU만으로 정확도 실측 (2026)](https://www.cinevyze.com/2026/07/free-korean-ocr-tesseract-accuracy-tested-2026.html)
- [노션 AI로 업무 자동화, 진짜 되나? — 요금·API 한계까지 솔직 정리 (2026)](https://www.cinevyze.com/2026/07/notion-ai-work-automation-cost-limits-2026.html)
- [Whisper 한국어 받아쓰기, 무료로 어디까지 되나 — CPU만으로 5개 크기 실측 (2026)](https://www.cinevyze.com/2026/07/whisper-korean-stt-cpu-model-size-tested-2026.html)
- [공짜로 내 PC에서 돌리는 AI(gemma3:4b), 유료 클라우드 어디까지 따라잡나 — 같은 일 직접 시켜본 실측 (2026.6)](https://www.cinevyze.com/2026/06/local-ai-gemma3-vs-cloud-test-2026.html)

### AI 글쓰기·챗봇

- [무료 로컬 AI로 맞춤법·문장 교정, 진짜 고쳐줄까 — 오류 심은 초안 실측 (2026)](https://www.cinevyze.com/2026/07/local-free-ai-korean-proofreading-tested-2026.html)
- [AI 문서요약, 클라우드가 무조건 나을까 — 로컬 gemma3 vs Gemini 정확도·속도 실측 (2026)](https://www.cinevyze.com/2026/07/ai-summary-local-gemma3-vs-gemini-accuracy-tested-2026.html)
- [클로드 무료 제한, 진짜 몇 개일까 — 공식 문서엔 그 숫자가 아예 없습니다 (2026.7)](https://www.cinevyze.com/2026/07/claude-free-plan-limits-vs-pro-2026.html)
- [무료 로컬 AI로 블로그 글, 진짜 되나? — 내 PC에서 무료 모델 4개 직접 돌려본 실측 (2026.7)](https://www.cinevyze.com/2026/07/free-local-ai-blog-writing-4-models-tested-2026.html)
- [챗GPT·클로드·제미나이 요금 비교 — '셋 다 20달러'가 아니었습니다 (2026.07)](https://www.cinevyze.com/2026/06/chatgpt-claude-gemini-same-task-comparison-2026.html)
- [Gemini 3.1 Pro한테 코딩·요약·번역 직접 시켜봤다 — 같은 작업 9번 굴려본 실측 (2026.6)](https://www.cinevyze.com/2026/06/gemini-31-pro-9-20266.html)

### AI 활용가이드

- [무료 로컬 AI, 한국 상식 얼마나 맞히나 — 함정질문에 지어내는 놈 실측 (2026)](https://www.cinevyze.com/2026/07/local-free-ai-korean-facts-hallucination-tested-2026.html)
- [AI 답변, 그대로 믿었다간 — 환각 거르는 법(고정 질문 5개로 직접 잡아봤습니다)](https://www.cinevyze.com/2026/07/how-to-catch-verify-ai-hallucinations-2026.html)
- [AI한테 일 제대로 시키는 법 — 프롬프트 4칸 공식(같은 모델에 18번 넣어 재봤습니다)](https://www.cinevyze.com/2026/06/how-to-write-ai-prompts-4-part-framework-2026.html)

## 없는 것

원본 실행 로그(`test_runs/`)와 원고는 포함하지 않습니다. 코드와 문항만으로 재현할 수 있게 맞췄습니다.

## 라이선스

MIT
