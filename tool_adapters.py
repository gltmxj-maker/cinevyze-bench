# -*- coding: utf-8 -*-
"""tool_adapters.py — Lane A 멀티툴 어댑터 레지스트리(정규화).

★기존 하드코딩(`_GEMINI_TOOLS` + NotImplementedError)을 폐기하고, 툴 추가 = **어댑터 1개 드롭 +
REGISTRY 등록**으로 일반화. 각 어댑터 계약:
  - name        : 레지스트리 키('gemini'·'ollama'·'openai'·'claude')
  - access      : 'cli' | 'local' | 'api'  (run.yaml access·gate 증거 분기. 전부 텍스트=비visual)
  - default_model
  - tos_source  : run.yaml tos_source_url(정직 출처 라벨)
  - available() : bool — 접근 가능 여부. ★False면 호출부가 Lane B로 강등(안 돌려본 걸 돌린 척 금지)
  - run(prompt, model=None, timeout) -> str  : 실제 호출. 빈 출력/실패는 AdapterError로 승격
    ★생성 벽 = 900s 공통(2026-09-11 · 시네 반복지시 「타임아웃 걸지 말던가 여유롭게」).
      liveness 프로브(list·system_stats)만 짧게 둔다 — 그건 작업을 죽이는 벽이 아니다.

★접근 원칙(2026-06-24 firsthand 검증·memory firsthand-experience-automation): **공식 API/CLI/로컬만**.
  소비자 웹앱 로그인·UI 자동화(chatgpt.com·claude.ai 스크래핑)는 ToS 위반·밴 → 어댑터로 만들지 않는다.
  로컬(ollama)=구독/키 무관 무제한 무료 엔진. api=공식 키. cli=공식 구독 클라이언트(agy).
"""
import os
import time
import json
import shutil
import subprocess
import urllib.request


class AdapterError(RuntimeError):
    """어댑터 실패(미설치·키없음·빈출력 등). 호출부가 skip/강등/중단 결정."""


# ─────────────────────────────────────────────────────────────
# Gemini — 기존 agy_adapter(Antigravity 공식 CLI·OAuth 구독·키 불필요) 래핑
# ─────────────────────────────────────────────────────────────
import agy_adapter


class GeminiAdapter:
    name = "gemini"
    access = "cli"
    default_model = agy_adapter.DEFAULT_MODEL
    tos_source = "Antigravity 구독 CLI(공식 클라이언트·OAuth)"

    def available(self):
        return agy_adapter.gemini_available()

    def run(self, prompt, model=None, timeout=900):
        return agy_adapter.run_gemini(prompt, model=model or self.default_model, timeout=timeout)


# ─────────────────────────────────────────────────────────────
# Ollama — 로컬 오픈웨이트 모델(무료·무제한·구독/키/계정 무관). 구독 한계 없는 핵심 엔진.
# ─────────────────────────────────────────────────────────────
class OllamaAdapter:
    name = "ollama"
    access = "local"
    default_model = "gemma3:4b"   # ★로컬에 이미 설치된 모델(firsthand 2026-06-24). 미설치 모델 자동 pull 금지.
    tos_source = "로컬 오픈웨이트 모델(자체 구동·구독/계정 무관)"

    # ★기설치 바이너리 우선 사용(MCF 등이 ~/.local에 깐 경우 PATH에 없을 수 있음 — 확인없는 재설치 금지·시네 규칙 2026-06-24)
    _KNOWN_BINS = (
        os.path.expanduser("~/.local/ollama/bin/ollama"),
        "/usr/local/bin/ollama",
        "/opt/ollama/ollama",
    )

    def __init__(self):
        self.last_meta = {}
        # 디코딩 옵션(temperature·seed·top_k …) — REP 재현성 실측이 이 축을 움직인다(TESTSET §3-C).
        # 기본 {} = /api/generate 에 options 키 자체를 안 보냄 = 모델 Modelfile 기본값(기존 동작 불변).
        self.extra_options = {}

    def _api_host(self):
        h = (os.environ.get("OLLAMA_HOST") or "").strip()
        if not h:
            return "http://127.0.0.1:11434"
        if not h.startswith("http"):
            h = "http://" + h
        return h.rstrip("/")

    def _bin(self):
        b = shutil.which("ollama") or os.environ.get("OLLAMA_BIN")
        if b and os.path.exists(b):
            return b
        for p in self._KNOWN_BINS:
            if os.path.exists(p):
                return p
        return None

    def _models_dir(self):
        return os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")

    def installed_models(self):
        """디스크 매니페스트 직접 스캔 → 서버 미기동에도 설치 모델 인식(확인없는 재설치 방지)."""
        base = os.path.join(self._models_dir(), "manifests")
        out = []
        if not os.path.isdir(base):
            return out
        for root, _dirs, files in os.walk(base):
            for f in files:
                out.append(f"{os.path.basename(root)}:{f}")   # name:tag (예: gemma3:4b)
        return out

    def _serve_up(self, b):
        try:
            r = subprocess.run([b, "list"], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def _ensure_server(self, b):
        """데몬 미기동 시 백그라운드 기동(로컬·idempotent·재설치 아님). 실패는 정직 False(스왈로 X)."""
        if self._serve_up(b):
            return True
        try:
            subprocess.Popen([b, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False
        for _ in range(20):   # ~10s 대기
            if self._serve_up(b):
                return True
            time.sleep(0.5)
        return False

    def available(self):
        # 서버 미기동이어도 '바이너리+설치모델 존재'면 가능(run()이 데몬 기동) — feasibility 현실 반영
        return bool(self._bin()) and bool(self.installed_models())

    def run(self, prompt, model=None, timeout=900):
        b = self._bin()
        if not b:
            raise AdapterError("ollama 바이너리 없음 — 기설치 확인 후 사용(확인없는 재설치 금지·시네 규칙).")
        m = model or self.default_model
        installed = self.installed_models()
        # ★확인없는 자동 다운로드 금지: 미설치 모델은 거부하고 설치된 것만 안내(새 모델은 명시 pull·SSD 경로)
        if installed and not any(x == m or x.split(":")[0] == m for x in installed):
            raise AdapterError(
                f"모델 '{m}' 미설치 — 자동 pull 금지(시네 규칙). 설치된 모델만 사용: {sorted(set(installed))} "
                f"(새 모델은 OLLAMA_MODELS=SSD 경로로 명시 `ollama pull {m}` 후 재시도)."
            )
        if not self._ensure_server(b):
            raise AdapterError("ollama 서버 미기동 — 자동기동 실패. 수동: `ollama serve`.")
        # ★HTTP API(/api/generate·stream=false) 사용 — `ollama run` CLI 캡처는 스트리밍 커서 제어(ANSI `\x1b[K`·
        #   `\x1b[..D`)가 출력에 새어들어 텍스트를 오염시키고(커서 되돌림+재출력=중복), 정확한 tok/s도 못 얻는다(2026-06-28 firsthand).
        # ★전송 스냅샷을 지역변수로 고정한다. last_meta 를 만들 때 self.extra_options 를 다시 읽으면,
        #   응답 대기(수 초~수십 초) 사이에 값이 바뀐 경우 '기록된 조건'과 '실제 보낸 조건'이 어긋난다
        #   — 측정 위조와 구분이 안 되는 종류의 버그다(T2 적대검증 2026-07-29).
        sent_options = dict(self.extra_options or {})
        payload = {"model": m, "prompt": prompt, "stream": False}
        if sent_options:
            payload["options"] = sent_options
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self._api_host() + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            raise AdapterError(f"ollama API 실패(model={m}): {e}")
        out = (resp.get("response") or "").strip()
        if not out:
            raise AdapterError(f"ollama 빈 출력(model={m}) — 모델 손상 의심.")
        ec, ed = resp.get("eval_count"), resp.get("eval_duration")   # 토큰수·생성시간(ns) → tok/s 재현 증거
        self.last_meta = {"eval_count": ec, "eval_duration_ns": ed,
                          "tok_per_s": round(ec / (ed / 1e9), 1) if ec and ed else None,
                          "prompt_eval_count": resp.get("prompt_eval_count"),
                          "total_duration_ns": resp.get("total_duration"),
                          # load_duration = 콜드 로딩(KEEP_ALIVE=0이면 매 호출 발생) — elapsed_s(총소요)에서
                          # 이걸 빼야 '순수 생성' 시간이다. 표에서 섞으면 07-26 배경제거 글 사고 재발.
                          "load_duration_ns": resp.get("load_duration"),
                          # 실제 전송한 디코딩 옵션 = 재현성 실측의 조건 증거(write-origin 로그에 기록됨).
                          # ★self.extra_options 를 다시 읽지 않는다 — 위 sent_options 스냅샷이 SSOT.
                          "options_sent": json.dumps(sent_options, ensure_ascii=False, sort_keys=True)
                                          if sent_options else "(none·model default)"}
        return out


# ─────────────────────────────────────────────────────────────
# OpenAI(공식 API·키). OpenAI-호환 엔드포인트라 base_url 교체로 호환 서비스도 가능.
# ─────────────────────────────────────────────────────────────
class OpenAIAdapter:
    name = "openai"
    access = "api"
    default_model = "gpt-5.4-mini"
    base_url = "https://api.openai.com/v1"
    key_env = "OPENAI_API_KEY"
    tos_source = "OpenAI 공식 API(키·종량제)"

    def available(self):
        return bool(os.environ.get(self.key_env))

    def run(self, prompt, model=None, timeout=900):
        key = os.environ.get(self.key_env)
        if not key:
            raise AdapterError(f"{self.key_env} 없음 — .env에 공식 API 키 필요(웹 로그인 자동화 금지).")
        try:
            from openai import OpenAI
        except ImportError:
            raise AdapterError("openai SDK 미설치 — pip install openai")
        client = OpenAI(api_key=key, base_url=self.base_url)
        r = client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        return (r.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# Anthropic Claude(공식 Messages API·키).
# ─────────────────────────────────────────────────────────────
class ClaudeAdapter:
    name = "claude"
    access = "api"
    default_model = "claude-haiku-4-5"
    key_env = "ANTHROPIC_API_KEY"
    tos_source = "Anthropic 공식 API(키·종량제)"

    def available(self):
        return bool(os.environ.get(self.key_env))

    def run(self, prompt, model=None, timeout=900):
        key = os.environ.get(self.key_env)
        if not key:
            raise AdapterError(f"{self.key_env} 없음 — .env에 공식 API 키 필요(claude.ai 로그인 자동화 금지).")
        try:
            import anthropic
        except ImportError:
            raise AdapterError("anthropic SDK 미설치 — pip install anthropic")
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(
            model=model or self.default_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in r.content
                       if getattr(b, "type", None) == "text").strip()


# ─────────────────────────────────────────────────────────────
# ComfyUI — 로컬 이미지 생성(무료·GPU·modality=image). 텍스트 어댑터와 동일 계약 + run()이 PNG bytes 반환.
#   ★MCF가 깔아둔 ComfyUI(.venv=CUDA torch·RTX 4070 Ti) + SDXL 체크포인트 재사용(재설치 X·기설치 baseline 보호).
#   설치 체크포인트가 일러스트/만화 특화(animagine·Illustrious)라 어댑터·글은 '블로그 일러스트/마스코트' 범위로
#   정직 한정(범용 포토리얼 주장 금지). 미설치/미가용·빈출력이면 run 거부(안 뽑은 걸 뽑은 척 금지).
#   재현 파라미터(seed·steps·sampler·ckpt·prompt_id)는 self.last_meta로 노출 → harness가 log_file에 기록(write-origin).
# ─────────────────────────────────────────────────────────────
class ComfyUIAdapter:
    name = "comfyui"
    access = "local"
    modality = "image"          # ★텍스트 어댑터는 modality 미선언(harness가 getattr default 'text')
    default_model = "animagine-xl-4.0-opt.safetensors"
    tos_source = "로컬 ComfyUI + 오픈 SDXL 체크포인트(자체 구동·구독/계정 무관)"

    HOST = "http://127.0.0.1:8188"
    _DIR = os.path.expanduser("~/mcf-tools/ComfyUI")
    _ALIAS = {"animagine": "animagine-xl-4.0-opt.safetensors",
              "illustrious": "Illustrious-XL-v2.0.safetensors"}
    _STEPS, _CFG, _SAMPLER, _RES = 26, 6.5, "euler_ancestral", 1024

    def __init__(self):
        self._seed = 1000          # 호출마다 +1 (repeats가 같은 프롬프트라도 다른 그림 = 대표성) · 시드 log 기록
        self.last_meta = {}

    def _py(self):
        p = os.path.join(self._DIR, ".venv", "bin", "python")
        return p if os.path.exists(p) else None

    def _ckpt_dir(self):
        return os.path.join(self._DIR, "models", "checkpoints")

    def _ckpts(self):
        d = self._ckpt_dir()
        return [f for f in os.listdir(d) if f.endswith(".safetensors")] if os.path.isdir(d) else []

    def _resolve_ckpt(self, model):
        m = self._ALIAS.get(model or "", model) or self.default_model
        return m if m.endswith(".safetensors") else m + ".safetensors"

    def _reachable(self):
        try:
            urllib.request.urlopen(self.HOST + "/system_stats", timeout=5)
            return True
        except Exception:
            return False

    def available(self):
        # 서버 미기동이어도 .venv + 체크포인트 있으면 run()이 기동 → feasibility 현실 반영(가짜정밀 아님)
        return self._reachable() or (bool(self._py()) and bool(self._ckpts()))

    def _ensure_server(self, timeout=120):
        if self._reachable():
            return True
        py = self._py()
        if not py:
            return False
        try:
            subprocess.Popen([py, "main.py", "--port", "8188"], cwd=self._DIR,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False
        for _ in range(int(timeout / 1.5)):   # ComfyUI 로딩은 ollama보다 느림
            if self._reachable():
                return True
            time.sleep(1.5)
        return False

    def _workflow(self, prompt, ckpt, seed):
        neg = "lowres, bad anatomy, bad hands, blurry, watermark, signature, text, jpeg artifacts, extra limbs"
        return {
            "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": self._STEPS, "cfg": self._CFG,
                  "sampler_name": self._SAMPLER, "scheduler": "normal", "denoise": 1,
                  "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": self._RES, "height": self._RES, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "harness", "images": ["8", 0]}},
        }

    def _post(self, path, data):
        req = urllib.request.Request(self.HOST + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def _get(self, path):
        return json.loads(urllib.request.urlopen(self.HOST + path, timeout=30).read())

    def run(self, prompt, model=None, timeout=900):
        if not self._ensure_server():
            raise AdapterError("ComfyUI 서버 미기동 — .venv/main.py 자동기동 실패(수동 확인 필요).")
        ckpt = self._resolve_ckpt(model)
        if ckpt not in self._ckpts():
            raise AdapterError(f"체크포인트 '{ckpt}' 미설치 — 설치된 것만 사용: {self._ckpts()}")
        seed = self._seed
        self._seed += 1
        try:
            r = self._post("/prompt", {"prompt": self._workflow(prompt, ckpt, seed)})
        except Exception as e:
            raise AdapterError(f"ComfyUI /prompt 실패: {e}")
        pid = r.get("prompt_id")
        if not pid:
            raise AdapterError(f"ComfyUI prompt_id 없음: {r}")
        img, deadline = None, time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = self._get("/history/" + pid)
            if pid in h:
                for _n, o in h[pid].get("outputs", {}).items():
                    for im in o.get("images", []):
                        img = im
                        break
                if img:
                    break
            time.sleep(0.2)   # ★촘촘 폴링 = 정직한 end-to-end 타이밍(1초 폴링은 ±1s 양자화)
        if not img:
            raise AdapterError(f"ComfyUI 생성 타임아웃({timeout}s·ckpt={ckpt}).")
        path = os.path.join(self._DIR, "output", img.get("subfolder", ""), img["filename"])
        if not os.path.exists(path):
            raise AdapterError(f"ComfyUI 출력 파일 없음: {path}")
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 200:
            raise AdapterError(f"ComfyUI 출력 과소({len(data)}B) — 생성 실패 의심(빈 출력 위장 차단).")
        self.last_meta = {"prompt_id": pid, "seed": seed, "steps": self._STEPS, "cfg": self._CFG,
                          "sampler": self._SAMPLER, "checkpoint": ckpt,
                          "resolution": f"{self._RES}x{self._RES}", "source_path": path}
        return data


# ─────────────────────────────────────────────────────────────
# Upscale — 로컬 오픈 업스케일러(APISR GAN·spandrel·modality=image·access=local)
#   ★GPU 작업은 ComfyUI .venv(cu130·CUDA)에서만 가능 → upscale_worker.py를 그 .venv 파이썬으로
#     subprocess 호출(ComfyUIAdapter가 HTTP 서버로 우회하는 것과 동일 취지). run()=PNG bytes 반환.
#   ★prompt = 입력 이미지 경로(testset UPS instruction). model = lanczos/apisr2x/apisr4x.
# ─────────────────────────────────────────────────────────────
class UpscaleAdapter:
    name = "upscale"
    access = "local"
    modality = "image"
    default_model = "apisr4x"
    tos_source = "로컬 오픈 업스케일러(APISR GAN·spandrel 자체 구동·구독/계정 무관)"

    _VENV_PY = os.path.expanduser("~/mcf-tools/ComfyUI/.venv/bin/python")
    _WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upscale_worker.py")
    _MODELS = {
        "apisr4x": os.environ.get("APISR4X_PATH", "./models/4x_APISR_GRL_GAN_generator.pth"),
        "apisr2x": os.environ.get("APISR2X_PATH", "./models/2x_APISR_RRDB_GAN_generator.pth"),
    }
    _METHODS = ("lanczos", "apisr2x", "apisr4x")

    def __init__(self):
        self.last_meta = {}

    def available(self):
        # .venv 파이썬 + 워커 + (APISR 모델) 존재 = feasibility(파일 검사만·in-proc torch import 안 함)
        if not (os.path.exists(self._VENV_PY) and os.path.exists(self._WORKER)):
            return False
        return any(os.path.exists(p) for p in self._MODELS.values())

    def run(self, prompt, model=None, timeout=900):
        method = (model or self.default_model).strip()
        if method not in self._METHODS:
            raise AdapterError(f"미지원 업스케일 method '{method}' — {self._METHODS} 중 하나.")
        in_path = (prompt or "").strip()                  # ★prompt = 입력 이미지 경로
        if not os.path.exists(in_path):
            raise AdapterError(f"업스케일 입력 이미지 없음: {in_path}")
        import tempfile
        fd, out_path = tempfile.mkstemp(suffix=".png", prefix="ups_")
        os.close(fd)
        try:
            try:
                r = subprocess.run([self._VENV_PY, self._WORKER, in_path, method, out_path],
                                   capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise AdapterError(f"업스케일 타임아웃({timeout}s·method={method}).")
            if r.returncode != 0:
                raise AdapterError(f"업스케일 워커 실패(rc={r.returncode}): {r.stderr.strip()[:300]}")
            meta = {}
            for line in r.stdout.splitlines():
                if line.startswith("META:"):
                    meta = json.loads(line[5:])
            if not os.path.exists(out_path):
                raise AdapterError("업스케일 출력 파일 없음 — 생성 실패.")
            with open(out_path, "rb") as f:
                data = f.read()
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
        if len(data) < 200:
            raise AdapterError(f"업스케일 출력 과소({len(data)}B) — 빈 출력 위장 차단.")
        self.last_meta = meta
        return data


# ─────────────────────────────────────────────────────────────
# Rembg — 로컬 오픈 배경제거(누끼·u2net ONNX·access=local·modality=image)
#   ★onnxruntime(CPU)로 all-blog .venv서 in-process 구동(UpscaleAdapter와 달리 subprocess 불필요).
#   모델은 SSD(U2NET_HOME)로 고정(디스크 규율). prompt = 입력 이미지 경로. run()=누끼 PNG(RGBA) bytes.
# ─────────────────────────────────────────────────────────────
class RembgAdapter:
    name = "rembg"
    access = "local"
    modality = "image"
    default_model = "u2net"
    tos_source = "로컬 오픈 배경제거(rembg·u2net ONNX·CPU·구독/계정 무관)"

    _MODEL_DIR = os.environ.get("REMBG_MODEL_DIR", "./models/rembg")   # 단일 일회용 폴더(폴더째 삭제로 cleanup)

    def __init__(self):
        self.last_meta = {}
        self._sess = {}

    def available(self):
        try:
            import rembg  # noqa: F401
        except Exception:
            return False
        return os.path.isdir(self._MODEL_DIR)

    def _session(self, model):
        os.environ.setdefault("U2NET_HOME", self._MODEL_DIR)
        if model not in self._sess:
            from rembg import new_session
            self._sess[model] = new_session(model)
        return self._sess[model]

    def run(self, prompt, model=None, timeout=900):
        import io
        from PIL import Image
        from rembg import remove
        m = (model or self.default_model).strip()
        in_path = (prompt or "").strip()
        if not os.path.exists(in_path):
            raise AdapterError(f"배경제거 입력 이미지 없음: {in_path}")
        sess = self._session(m)
        im = Image.open(in_path).convert("RGB")
        t = time.monotonic()
        out = remove(im, session=sess)            # RGBA(투명 배경)
        dt = time.monotonic() - t
        buf = io.BytesIO(); out.save(buf, format="PNG"); data = buf.getvalue()
        if len(data) < 200:
            raise AdapterError(f"배경제거 출력 과소({len(data)}B) — 빈 출력 위장 차단.")
        # 전경 픽셀 비율(alpha>=16) = 배경이 실제로 제거됐는지 정직 지표
        try:
            hist = out.getchannel("A").histogram()
            tot = im.size[0] * im.size[1]
            fg_ratio = round(sum(hist[16:]) / tot, 3) if tot else None
        except Exception:
            fg_ratio = None
        self.last_meta = {"model": m, "in_path": in_path,
                          "in_size": f"{im.size[0]}x{im.size[1]}",
                          "proc_s": round(dt, 3), "fg_ratio": fg_ratio, "has_alpha": out.mode == "RGBA"}
        return data


# ─────────────────────────────────────────────────────────────
# Whisper — 로컬 오픈 음성인식(STT·faster-whisper/CTranslate2·access=local·modality=text)
#   run()의 prompt = 오디오 파일 절대경로(입력) → 전사 텍스트 반환(텍스트 어댑터 계약과 동일).
#   ★CPU int8(GPU 미사용·하드행 회피·[[gpu-hard-freeze-hardware-risk]]). 크기별 모델 로드 캐시(반복 재로딩 비용 절감).
#   last_meta = 재현/증거(언어·beam·compute·audio_s·load_s·rtf) → harness log_file에 기록(write-origin).
# ─────────────────────────────────────────────────────────────
class WhisperAdapter:
    name = "whisper"
    access = "local"
    default_model = "small"
    tos_source = "로컬 오픈 음성인식(faster-whisper·CTranslate2·CPU·오프라인·구독/계정 무관)"

    # 크기 별칭 → faster-whisper 로드명(turbo = 커뮤니티 CT2 저장소)
    _ALIAS = {
        "tiny": "tiny", "base": "base", "small": "small", "large-v3": "large-v3",
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    }
    _BEAM = 5
    _LANG = "ko"

    def __init__(self):
        self.last_meta = {}
        self._models = {}

    def available(self):
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return False
        return True

    def _model(self, size):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 캐시서만(오프라인·결정론)
        if size not in self._models:
            from faster_whisper import WhisperModel
            t = time.monotonic()
            mdl = WhisperModel(self._ALIAS.get(size, size), device="cpu", compute_type="int8")
            self._models[size] = (mdl, round(time.monotonic() - t, 2))
        return self._models[size]

    def run(self, prompt, model=None, timeout=900):
        size = (model or self.default_model).strip()
        in_path = (prompt or "").strip()
        if not os.path.exists(in_path):
            raise AdapterError(f"STT 입력 오디오 없음: {in_path}")
        mdl, load_s = self._model(size)
        t = time.monotonic()
        segs, info = mdl.transcribe(in_path, language=self._LANG, beam_size=self._BEAM)
        text = "".join(s.text for s in segs).strip()    # 원출력 그대로(환각 구두점도 위장 없이 통과)
        infer_s = round(time.monotonic() - t, 2)
        dur = round(info.duration, 2)
        self.last_meta = {
            "model_size": size, "load_name": self._ALIAS.get(size, size),
            "audio_s": dur, "infer_s": infer_s, "load_s": load_s,
            "device": "cpu", "compute_type": "int8", "beam_size": self._BEAM,
            "language": self._LANG, "rtf": round(infer_s / dur, 3) if dur else None,
        }
        return text


# ─────────────────────────────────────────────────────────────
# Tesseract — 로컬 오픈 OCR(이미지→텍스트·access=local·modality=text)
#   run()의 prompt = 이미지 파일 절대경로(입력) → 인식 텍스트 반환(텍스트 어댑터 계약과 동일).
#   model = tesseract 언어 스펙(-l): "kor" | "kor+eng" | "eng". CPU 전용(GPU 미사용·하드행 무관).
#   설치 0건 = 시스템 tesseract 바이너리 직접 subprocess(pytesseract 불요). psm/oem 고정 = 재현.
#   last_meta = 재현/증거(lang·psm·oem·img_size·proc_s·tess) → harness log_file에 기록(write-origin).
# ─────────────────────────────────────────────────────────────
class TesseractAdapter:
    name = "tesseract"
    access = "local"
    default_model = "kor+eng"
    tos_source = "로컬 오픈 OCR(Tesseract·CPU·오프라인·구독/계정 무관)"

    _PSM = "6"    # 균일 텍스트 블록(문단) 가정
    _OEM = "1"    # LSTM 신경망 엔진(기본)

    def __init__(self):
        self.last_meta = {}
        self._ver = None

    def _version(self):
        if self._ver is None:
            try:
                out = subprocess.run(["tesseract", "--version"], capture_output=True,
                                     text=True, timeout=15)
                self._ver = (out.stdout or out.stderr).splitlines()[0].strip()
            except Exception:
                self._ver = "unknown"
        return self._ver

    def available(self):
        import shutil
        return shutil.which("tesseract") is not None

    def run(self, prompt, model=None, timeout=900):
        lang = (model or self.default_model).strip()
        in_path = (prompt or "").strip()
        if not os.path.exists(in_path):
            raise AdapterError(f"OCR 입력 이미지 없음: {in_path}")
        try:
            from PIL import Image
            with Image.open(in_path) as im:
                img_size = f"{im.size[0]}x{im.size[1]}"
        except Exception:
            img_size = None
        cmd = ["tesseract", in_path, "stdout", "-l", lang, "--psm", self._PSM, "--oem", self._OEM]
        t = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        proc_s = round(time.monotonic() - t, 3)
        if proc.returncode != 0:
            raise AdapterError(f"tesseract 실패(rc={proc.returncode}): {proc.stderr.strip()[:200]}")
        text = (proc.stdout or "").strip()      # 원출력 그대로(오독도 위장 없이 통과)
        self.last_meta = {
            "lang": lang, "psm": self._PSM, "oem": self._OEM,
            "img_size": img_size, "proc_s": proc_s, "tess": self._version(),
        }
        return text


REGISTRY = {a.name: a for a in (GeminiAdapter(), OllamaAdapter(), OpenAIAdapter(),
                                ClaudeAdapter(), ComfyUIAdapter(), UpscaleAdapter(),
                                RembgAdapter(), WhisperAdapter(), TesseractAdapter())}


def resolve(spec):
    """'gemini' | 'ollama:qwen3:8b' | 'openai:gpt-5.4-mini' -> (adapter, model_or_None).

    첫 ':' 기준 분리(ollama 모델명의 'qwen3:8b' 콜론 보존)."""
    if ":" in spec:
        name, model = spec.split(":", 1)
    else:
        name, model = spec, None
    a = REGISTRY.get(name)
    if not a:
        raise AdapterError(f"미등록 툴 '{name}' (등록됨: {list(REGISTRY)}) — 어댑터를 추가하라(추측 금지).")
    return a, model


def available_tools():
    """{name: bool} — 각 어댑터 접근 가능 여부(키/설치/구독 firsthand 확인)."""
    return {n: a.available() for n, a in REGISTRY.items()}
