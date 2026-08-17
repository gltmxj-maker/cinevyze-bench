# corpus — 입력 데이터 폴더

원본 저장소의 실측에 쓴 문서·이미지·음성은 **저작권과 개인정보 때문에 공개하지 않습니다.**
대신 각 하네스가 기대하는 **파일 규격**을 적어 둡니다. 자기 자료를 넣으면 그대로 돌아갑니다.

| 폴더/파일 | 쓰는 하네스 | 규격 |
|---|---|---|
| `corpus/auto/` | `automation_bench.py` | 반복 취합 대상 파일들(엑셀/CSV) |
| `corpus/rag/` | `rag_bench.py` | 검색 대상 문서(txt/pdf) |
| `corpus/rag_questions.json` | `rag_bench.py` | `[{"q": "...", "answer": "..."}]` |
| `corpus/translate/s1-tech.txt` 외 2 | `translate_bench.py` | 번역 원문(기술문서·캐주얼 후기·비즈니스 메일) |
| `corpus/ups/` | `upscale_bench.py` | 업스케일 대상 원본 이미지 |

경로는 `BENCH_CORPUS_DIR` 환경변수로 바꿀 수 있습니다.
