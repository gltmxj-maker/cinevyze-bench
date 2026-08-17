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

## 없는 것

원본 실행 로그(`test_runs/`)와 원고는 포함하지 않습니다. 코드와 문항만으로 재현할 수 있게 맞췄습니다.

## 라이선스

MIT
