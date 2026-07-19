"""
script_writer.py — "đạo diễn kịch bản": phân tích cốt truyện, chọn công
thức viral, viết lời bình (narration), và tạo storyboard.

THAY THẾ: trước đây dùng Claude built-in (không cần API call, chạy ngay trong
agent session của Claude Code). Bây giờ dùng Gemma 4 31B qua Cerebras API
(OpenAI-compatible endpoint), gọi qua thư viện `openai`.

Hàm `generate_narration()` giữ nguyên input/output format: nhận semantic blocks
+ task config, trả về danh sách câu narration đã gắn scene_ids nguồn — để các
stage sau (storyboard, render) không bị vỡ.

Thiết kế an toàn: LLM CHỈ được chọn scene_ids từ danh sách block có sẵn, không
được tự bịa timestamp. Timestamp thật (source/output) được tính toán bằng code
Python xác định sau khi có narration, không phải do LLM tự sinh số.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from platform_utils import resolve_torch_device
from progress_utils import print_progress_bar


def _get_client(cfg) -> OpenAI:
    api_key = cfg.get("api.cerebras_api_key", "")
    base_url = cfg.get("api.cerebras_base_url", "https://api.cerebras.ai/v1")
    if not api_key or api_key.startswith("PASTE_"):
        raise ValueError(
            "Chưa cấu hình api.cerebras_api_key trong config.toml. "
            "Điền key vào config.toml (mục [api] cerebras_api_key) — xem README."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat(cfg, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    """Gọi LLM để viết kịch bản — dispatch theo `api.script_backend`:
      - "local" (MẶC ĐỊNH): model nhẹ chạy ngay trên máy/Colab (Qwen3-4B-Instruct-2507),
        không cần API key, không giới hạn context nhỏ, không có rủi ro "nuốt hết
        token vào suy luận nội bộ" như model reasoning.
      - "cerebras": dùng lại API Cerebras như trước (đổi `api.script_backend =
        "cerebras"` trong config.toml nếu muốn quay lại).
    """
    backend = cfg.get("api.script_backend", "local")
    if backend == "local":
        model = _get_local_model(cfg)
        max_new_tokens = max_tokens or cfg.get("processing.script_local_max_new_tokens", 3000)
        return model.chat(system_prompt, user_prompt, max_new_tokens)
    return _chat_cerebras(cfg, system_prompt, user_prompt, max_tokens)


def _chat_cerebras(cfg, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    """Gọi Cerebras (chat completions, OpenAI-compatible).

    max_tokens: nếu None, dùng api.cerebras_max_tokens trong config (hành vi cũ).
    Cho phép override theo từng call vì narration được sinh theo batch nhỏ
    (xem generate_narration) — mỗi batch chỉ cần vài trăm token thay vì
    toàn bộ cerebras_max_tokens, để dành ngân sách token cho input.
    """
    client = _get_client(cfg)
    model = cfg.get("api.cerebras_model", "gemma-4-31b")
    if max_tokens is None:
        max_tokens = cfg.get("api.cerebras_max_tokens", 8000)
    temperature = cfg.get("api.cerebras_temperature", 0.8)

    # BUGFIX: model zai-glm-4.7 (mặc định trong config) là model REASONING —
    # nó tự sinh 1 chuỗi "suy nghĩ nội bộ" (reasoning tokens) TRƯỚC KHI viết
    # câu trả lời thật. Nếu max_tokens của 1 request nhỏ (vd 2000, dùng cho
    # từng batch narration), phần suy nghĩ đó có thể ăn hết toàn bộ ngân sách
    # token -> response.content rỗng -> _extract_json("") báo lỗi khó hiểu
    # "Expecting value: line 1 column 1". Tắt hẳn reasoning cho các lệnh gọi
    # ở đây vì ta chỉ cần JSON có cấu trúc, không cần model "suy nghĩ thành
    # tiếng" — theo đúng khuyến nghị của Cerebras (reasoning_effort="none").
    extra_body: dict[str, Any] = {}
    if cfg.get("api.cerebras_disable_reasoning", True):
        extra_body["reasoning_effort"] = "none"

    # Một API call không có log tải sẵn theo %, nên dùng streaming: mỗi chunk
    # nhận về là 1 tín hiệu tiến độ thật (ước lượng theo token đã nhận /
    # max_tokens), thay vì chỉ báo "vẫn đang chạy..." như Heartbeat.
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        extra_body=extra_body or None,
    )
    chunks: list[str] = []
    approx_tokens = 0
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if not delta:
            continue
        chunks.append(delta)
        approx_tokens += max(1, len(delta) // 4)  # ước lượng ~4 ký tự/token
        print_progress_bar(
            min(approx_tokens, max_tokens), max_tokens,
            prefix="[script_writer] cerebras", suffix=f"~{approx_tokens} token",
        )
    if approx_tokens < max_tokens:
        print_progress_bar(max_tokens, max_tokens, prefix="[script_writer] cerebras", suffix="xong")
    return "".join(chunks)


# =============================================================================
# Backend "local" — model nhẹ chạy ngay trên máy (mặc định Qwen3-4B-Instruct-2507)
# =============================================================================
#
# Lý do đổi mặc định sang local: model reasoning qua Cerebras (zai-glm-4.7) có
# thể "nuốt" hết ngân sách token vào suy luận nội bộ trước khi kịp viết JSON
# (xem BUGFIX trong _chat_cerebras), đặc biệt dễ xảy ra khi narration được
# chia thành nhiều batch nhỏ cho phim dài. Chạy local tránh hẳn vấn đề đó
# (không cần API key, không rate limit, không giới hạn context 8K khắt khe),
# đổi lại cần GPU/CPU đủ mạnh để tải + chạy model — Qwen3-4B-Instruct-2507
# được chọn vì nhẹ (~4B tham số, cùng cỡ với Qwen3-VL-4B đã dùng cho vision
# nên không tốn thêm VRAM đáng kể), nhanh, và viết văn tiếng Việt tốt. Muốn
# quay lại Cerebras: đổi `api.script_backend = "cerebras"` trong config.toml.

_local_model_singleton: "LocalScriptModel | None" = None

# VRAM còn trống (GB) dưới ngưỡng này -> tự động bật quantization 4-bit thay
# vì tải full bf16/fp16. Qwen3-4B ở bf16 tốn ~8GB CHỈ để load weights, chưa
# tính KV cache cho input dài (batch narration có thể ~20K token) -> trên GPU
# < ~12GB free (Colab T4 dùng chung, laptop GPU, v.v.) rất dễ OOM giữa generate().
# Ở 4-bit, cùng model chỉ tốn ~2.5-3GB weights, để dư nhiều VRAM hơn cho KV cache.
_LOW_VRAM_QUANT_THRESHOLD_GB = 12.0
# Dưới ngưỡng này (kể cả sau khi đã quant 4-bit) coi như không đủ để chạy GPU
# ổn định cho ngữ cảnh dài -> rơi về CPU thay vì cố chạy rồi OOM giữa chừng.
_MIN_USABLE_CUDA_VRAM_GB = 3.5


class LocalModelOOMError(RuntimeError):
    """Hết VRAM/RAM khi sinh text bằng model local, kể cả sau khi đã thử giảm
    max_new_tokens + dọn cache. Caller (vd generate_narration) có thể bắt lỗi
    này để CHIA NHỎ batch input rồi thử lại, thay vì để cả pipeline crash."""


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text or isinstance(exc, MemoryError)


class LocalScriptModel:
    """Bọc model + tokenizer chạy local, load một lần và tái sử dụng cho toàn
    bộ hooks + mọi batch narration của 1 project (tránh load lại model nặng
    cho mỗi lệnh gọi).

    Tự thích ứng với phần cứng thật đang chạy (thay vì giả định luôn có GPU
    lớn rảnh VRAM): phát hiện VRAM còn trống lúc load() để tự chọn quantization
    4-bit khi cần, và tính ra "ngân sách" context token an toàn thay vì dùng
    1 con số cố định cho mọi máy (xem `recommended_max_context_tokens`)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.script_local_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = resolve_torch_device(cfg.get("processing.script_local_device", "auto"))
        self.dtype_name = cfg.get("processing.script_local_dtype", "bfloat16")
        self.quantization = cfg.get("processing.script_local_quantization", "auto")  # auto|4bit|8bit|none
        self.temperature = cfg.get("processing.script_local_temperature", 0.7)
        self.top_p = cfg.get("processing.script_local_top_p", 0.8)
        self.top_k = cfg.get("processing.script_local_top_k", 20)
        self.model = None
        self.tokenizer = None
        self._torch = None
        self._quantized = False
        self._free_vram_gb_at_load: float | None = None

    def load(self) -> None:
        # Giảm phân mảnh VRAM (nguyên nhân phổ biến gây OOM dù tổng VRAM còn
        # đủ) — phải set TRƯỚC khi CUDA context được khởi tạo (trước import torch).
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch  # lazy import: chỉ cần khi thực sự dùng backend "local"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            self.dtype_name, torch.bfloat16
        )

        from platform_utils import get_free_vram_gb
        free_vram_gb = get_free_vram_gb() if self.device == "cuda" else None
        self._free_vram_gb_at_load = free_vram_gb

        # Nếu GPU được chọn nhưng gần như không còn VRAM trống -> rơi về CPU
        # ngay từ đầu thay vì cố tải rồi OOM (vd GPU đang bị stage khác giữ chỗ).
        if self.device == "cuda" and free_vram_gb is not None and free_vram_gb < _MIN_USABLE_CUDA_VRAM_GB:
            print(f"[script_writer] CẢNH BÁO: GPU chỉ còn ~{free_vram_gb:.1f}GB VRAM trống "
                  f"(< {_MIN_USABLE_CUDA_VRAM_GB}GB) — chuyển sang chạy CPU để tránh OOM.")
            self.device = "cpu"

        want_quant = self.quantization
        if want_quant == "auto":
            want_quant = (
                "4bit" if self.device == "cuda" and free_vram_gb is not None
                and free_vram_gb < _LOW_VRAM_QUANT_THRESHOLD_GB else "none"
            )

        quant_config = None
        if want_quant in ("4bit", "8bit") and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                if want_quant == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self._quantized = True
            except ImportError:
                print("[script_writer] CẢNH BÁO: thiếu package 'bitsandbytes' nên không bật được "
                      f"quantization {want_quant} dù VRAM thấp (~{free_vram_gb and round(free_vram_gb, 1)}GB "
                      "trống) — cài `pip install bitsandbytes` để giảm VRAM cần thiết. Vẫn tiếp tục "
                      "chạy full-precision, có thể chậm/OOM trên GPU nhỏ.")
                want_quant = "none"

        vram_note = f", VRAM trống ~{free_vram_gb:.1f}GB" if free_vram_gb is not None else ""
        print(f"[script_writer] Loading local model {self.model_name} on {self.device} "
              f"({self.dtype_name}, quantization={want_quant}{vram_note})... (lần đầu sẽ tải model; "
              f"log tải % / tốc độ của huggingface_hub sẽ hiện ngay bên dưới)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
        )

        load_kwargs = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            load_kwargs["dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        except TypeError:
            # transformers cũ hơn không nhận kwarg 'dtype' (chỉ có 'torch_dtype').
            load_kwargs.pop("dtype", None)
            if quant_config is None:
                load_kwargs["torch_dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        except self._torch.cuda.OutOfMemoryError:
            # Không đủ VRAM ngay cả để LOAD model (khác với OOM lúc generate).
            # Thử lần cuối bằng CPU thay vì crash toàn bộ pipeline.
            print("[script_writer] CẢNH BÁO: OOM khi tải model lên GPU — thử lại trên CPU.")
            gc.collect()
            self._torch.cuda.empty_cache()
            self.device = "cpu"
            load_kwargs.pop("device_map", None)
            load_kwargs.pop("quantization_config", None)
            load_kwargs["dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        if self.device in ("cpu", "mps") and quant_config is None:
            self.model.to(self.device)
        self.model.eval()

    def recommended_max_context_tokens(self, configured_default: int) -> int:
        """Ngân sách token INPUT an toàn cho model local, dựa trên VRAM thật
        phát hiện lúc load() thay vì 1 số cố định — máy yếu tự động dùng batch
        nhỏ hơn (nhiều batch hơn nhưng không OOM), máy mạnh vẫn tận dụng được
        context lớn như config yêu cầu."""
        if self.device != "cuda" or self._free_vram_gb_at_load is None:
            # CPU/MPS: không có ranh giới VRAM cứng như CUDA, nhưng vẫn giới
            # hạn hợp lý để tránh generate() quá chậm / cạn RAM hệ thống.
            return min(configured_default, 12000)
        free_gb = self._free_vram_gb_at_load
        # Ước lượng thô: mỗi 1K token input (KV cache, cỡ model 4B) tốn khoảng
        # 0.35-0.5GB VRAM khi CHƯA quantize, ít hơn khi đã 4-bit. Trừ hao sẵn
        # phần weights + generate buffer + margin an toàn trước khi chia cho
        # chi phí/1K token, rồi lấy min với giá trị cấu hình để không vượt
        # quá cái người dùng chủ động đặt.
        weights_reserve_gb = 3.0 if self._quantized else 8.5
        usable_gb = max(0.0, free_gb - weights_reserve_gb - 1.5)  # 1.5GB margin an toàn
        cost_per_1k_tokens_gb = 0.25 if self._quantized else 0.45
        est_tokens = int((usable_gb / cost_per_1k_tokens_gb) * 1000)
        est_tokens = max(1500, est_tokens)  # sàn tối thiểu để vẫn xử lý được scene
        return max(1500, min(configured_default, est_tokens))

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:
                pass

    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        """Sinh text qua chat template. Qwen3-4B-Instruct-2507 (bản "Instruct",
        KHÔNG phải bản "Thinking") không tự chèn suy luận nội bộ vào output
        mặc định, nên không gặp lại lỗi "nuốt hết token vào suy nghĩ" như
        model reasoning phía Cerebras.

        Tự phục hồi khi OOM: dọn cache + thử lại với max_new_tokens nhỏ hơn
        trước khi bó tay và raise LocalModelOOMError (để caller chia nhỏ input
        batch rồi thử lại — xem generate_narration)."""
        torch = self._torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        attempt_tokens = max_new_tokens
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                print_progress_bar(
                    0, 1, prefix="[script_writer] local",
                    suffix=f"đang sinh (max {attempt_tokens} token, input {input_len} token)...",
                )
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=attempt_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                print_progress_bar(1, 1, prefix="[script_writer] local", suffix="xong")
                gen_ids = output_ids[0][input_len:]
                return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as e:  # bắt rộng vì OutOfMemoryError có thể là torch.cuda.* hoặc RuntimeError tuỳ version
                if not _is_oom_error(e):
                    raise
                last_err = e
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if attempt_tokens > 512:
                    attempt_tokens = max(512, attempt_tokens // 2)
                    print(f"[script_writer] CẢNH BÁO: OOM lúc generate — dọn cache và thử lại với "
                          f"max_new_tokens giảm còn {attempt_tokens} (lần {attempt + 1}/3).")
                    continue
                break
        raise LocalModelOOMError(
            f"Hết VRAM/RAM khi sinh text (input {input_len} token) dù đã giảm max_new_tokens "
            f"xuống {attempt_tokens} và dọn cache 3 lần liên tiếp: {last_err}"
        ) from last_err


def _get_local_model(cfg) -> "LocalScriptModel":
    """Load model local 1 lần duy nhất (singleton cấp module) và tái sử dụng
    cho mọi lệnh gọi tiếp theo trong cùng tiến trình (hooks + mọi batch
    narration) — tránh tải lại model nặng nhiều lần."""
    global _local_model_singleton
    if _local_model_singleton is None:
        _local_model_singleton = LocalScriptModel(cfg)
        _local_model_singleton.load()
    return _local_model_singleton


def unload_local_script_model() -> None:
    """Giải phóng model local khỏi VRAM/RAM sau khi stage 'script' xong, để
    nhường chỗ cho các stage sau (tts, render). An toàn khi gọi dù model
    chưa từng được load (vd đang dùng backend 'cerebras')."""
    global _local_model_singleton
    if _local_model_singleton is not None:
        _local_model_singleton.unload()
        _local_model_singleton = None


class ScriptWriterJSONError(RuntimeError):
    """LLM trả về JSON hỏng/bị cắt sau khi đã thử lại — lỗi rõ ràng thay vì
    một json.JSONDecodeError khó hiểu lẫn trong traceback."""


def _extract_json(text: str) -> Any:
    """Trích JSON từ output LLM, chấp nhận việc model bọc trong ```json ... ```."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _chat_json(
    cfg, system_prompt: str, user_prompt: str, *, stage: str, max_retries: int = 1,
    max_tokens_override: int | None = None,
) -> Any:
    """Gọi `_chat()` rồi parse JSON qua `_extract_json()`, có retry khi JSON hỏng/bị cắt
    (vd: output chạm giới hạn token giữa chừng). Nếu vẫn lỗi sau khi retry,
    raise `ScriptWriterJSONError` rõ ràng thay vì để `json.JSONDecodeError` thô lọt
    ra ngoài với traceback khó hiểu.
    """
    last_err: Exception | None = None
    current_max_tokens = max_tokens_override
    for attempt in range(max_retries + 1):
        prompt = user_prompt
        if attempt > 0:
            prompt += (
                "\n\nLƯU Ý: lần trước output của bạn KHÔNG phải JSON hợp lệ (có thể bị "
                "cắt giữa chừng hoặc lẫn text thừa). Lần này trả lời NGẮN GỌN HƠN nếu "
                "cần và CHỈ trả về đúng 1 JSON hợp lệ, không thêm bất kỳ text nào khác."
            )
        raw = _chat(cfg, system_prompt, prompt, max_tokens=current_max_tokens)
        if not raw.strip():
            # Response rỗng hoàn toàn: KHÁC với JSON bị cắt/hỏng thông thường.
            # Nguyên nhân thường gặp nhất (model reasoning như zai-glm-4.7 qua
            # Cerebras): toàn bộ ngân sách token bị "suy nghĩ nội bộ" ăn hết
            # trước khi kịp viết câu trả lời. Tăng ngân sách token cho lần thử
            # kế tiếp thay vì chỉ lặp lại y hệt (sẽ rỗng lần nữa vì cùng nguyên nhân).
            last_err = json.JSONDecodeError("Expecting value", "", 0)
            print(f"[script_writer] CẢNH BÁO: LLM ở stage '{stage}' trả về RỖNG "
                  f"(lần thử {attempt + 1}/{max_retries + 1}) — có thể do ngân sách token "
                  f"bị dùng hết cho suy luận nội bộ trước khi viết câu trả lời.")
            if current_max_tokens:
                current_max_tokens = min(int(current_max_tokens * 2), 16000)
                print(f"[script_writer] Tăng ngân sách token cho lần thử kế tiếp lên ~{current_max_tokens}.")
            continue
        try:
            return _extract_json(raw)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"[script_writer] CẢNH BÁO: JSON từ LLM ở stage '{stage}' bị hỏng/cắt "
                  f"(lần thử {attempt + 1}/{max_retries + 1}): {e}")
    raise ScriptWriterJSONError(
        f"Stage '{stage}': LLM liên tục trả về JSON hỏng/rỗng sau "
        f"{max_retries + 1} lần thử ({last_err}). Nếu đang dùng backend 'cerebras' với "
        f"model reasoning (vd zai-glm-4.7), thử tăng "
        f"'api.cerebras_narration_batch_max_tokens' (hoặc 'api.cerebras_max_tokens') trong "
        f"config.toml, hoặc chuyển hẳn sang backend 'local' ('api.script_backend = \"local\"') "
        f"để tránh vấn đề này. Cũng có thể do lỗi mạng/API tạm thời — thử chạy lại."
    ) from last_err


def generate_hooks(cfg, task_config: dict[str, Any], director_brief: str = "") -> list[dict[str, str]]:
    """
    Sinh 10 câu hook mở đầu theo 4 hướng (phản差+爽点, 荒诞, 悬念, 提问, 数据)
    như mô tả trong skill.md Step 3. Trả về list [{"style": ..., "text": ...}].
    """
    narration_language = task_config.get("narration_language", "Vietnamese")
    system_prompt = (
        "You are a viral short-video scriptwriter for movie/drama commentary channels. "
        f"Write ALL hooks in {narration_language}, regardless of what language the plot "
        f"summary/brief below is written in (translate/adapt as needed) — the narration "
        f"voice-over for this project is {narration_language}, so text in any other "
        f"language cannot be used as-is. "
        "Respond ONLY with a JSON array of 10 objects: "
        '[{"style": "contrast|absurd|suspense|question|data", "text": "...", '
        '"language_used": "..."}]. '
        "No extra text."
    )
    user_prompt = (
        f"Content type: {task_config.get('content_type')}\n"
        f"Genre: {task_config.get('genre')}\n"
        f"Title: {task_config.get('title', '')}\n"
        f"Plot brief:\n{director_brief or task_config.get('plot_summary', '(none provided)')}\n\n"
        "Generate 10 opening hooks (first 1-2 sentences of the commentary): "
        "3 contrast/payoff twists, 3 absurd/dramatic, 2 suspense/conflict, "
        "1 question-style, 1 data+emotion. No slow build-up, no cliché openers."
    )
    return _chat_json(cfg, system_prompt, user_prompt, stage="hooks")


def _estimate_tokens(text: str) -> int:
    """Ước lượng SỐ TOKEN của 1 chuỗi (chars // 3).

    Đây là ước lượng CỐ Ý cao hơn thực tế (an toàn) để chia batch — không
    dùng cho mục đích tính tiền/giới hạn chính xác của API, chỉ để quyết
    định lúc nào cần cắt batch trước khi gọi Cerebras.
    """
    return max(1, len(text) // 3)


def _batch_semantic_blocks(
    compact_blocks: list[dict[str, Any]],
    max_input_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Chia compact_blocks thành nhiều batch sao cho JSON của mỗi batch nằm
    trong ngân sách token cho phép.

    BUGFIX GỐC: trước đây TOÀN BỘ compact_blocks (có thể 1000+ scene với
    phim dài, vd 1609 scene ~ 158.505 ký tự) bị dồn vào 1 request Cerebras
    duy nhất, vượt xa context window thật của model (8192 token) ->
    `openai.BadRequestError: context_length_exceeded`. Giờ chia nhỏ thành
    nhiều batch, mỗi batch được ước lượng để lọt vừa ngân sách token.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for block in compact_blocks:
        block_tokens = _estimate_tokens(json.dumps(block, ensure_ascii=False))
        if current and current_tokens + block_tokens > max_input_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        batches.append(current)
    return batches


def _narration_system_prompt(narration_language: str, is_continuation: bool) -> str:
    continuation_rule = (
        "7. Đây là phần TIẾP THEO của một narration dài hơn đã được sinh trước đó theo "
        "batch (do phim quá dài để đưa hết vào 1 request). Bạn được cho xem vài câu cuối "
        "của phần trước (mục 'story_so_far') CHỈ để giữ mạch văn/giọng kể liền mạch — "
        "TUYỆT ĐỐI không lặp lại các câu đó, không viết lại phần mở đầu/hook nữa.\n"
        if is_continuation else ""
    )
    return (
        "You are the story director for a viral movie/drama commentary video. "
        "You receive a list of semantic scene blocks (each with a scene_id, time range, "
        "visual summary, dialogue snippets, characters, emotion, tags) extracted from the "
        "source video. Analyze plot roles, emotional arcs, reversals, and conflict. "
        f"Write EVERY narration sentence in {narration_language} — this is the language of "
        f"the TTS voice-over for this project, so text in any other language would be read "
        f"aloud incorrectly. This applies even if the scene blocks, dialogue snippets, hook, "
        f"or director's brief given to you below are in a different language: translate/adapt "
        f"the narrative into {narration_language}, do not mix languages within a sentence. "
        "Write narration sentences that follow a viral formula appropriate to the genre, "
        "matching the target duration and point of view given. "
        "CRITICAL RULES:\n"
        "1. Every sentence must reference one or more scene_id values taken EXACTLY from the "
        "provided list — never invent a scene_id or a timestamp.\n"
        "2. Each sentence is a complete natural clause — do not split or merge to hit a word count.\n"
        "3. Use causal connectors (so/but) between beats, avoid repeating 'then'.\n"
        "4. Make the audience care about the protagonist within the first 2 sentences.\n"
        "5. Escalate stakes each beat; end with an emotional release in the last sentences.\n"
        "6. If no scene fits a beat well, still pick the closest scene and lower match_score, "
        "do not fabricate one.\n"
        f"{continuation_rule}"
        "Respond ONLY with a JSON array of objects with keys: "
        "sentence_id (string, e.g. 'sent_001'), sentence (string), scene_ids (array of strings), "
        "match_reason (string, explain WHY this scene serves the narrative beat), "
        "match_score (number 0-1)."
    )


def _narration_user_prompt(
    task_config: dict[str, Any],
    hook: str | None,
    director_brief: str,
    batch_blocks: list[dict[str, Any]],
    batch_target_duration: float,
    story_so_far: str,
    *,
    is_first: bool,
    is_last: bool,
) -> str:
    parts = [f"Task config: {json.dumps(task_config, ensure_ascii=False)}"]
    if is_first:
        parts.append(f"Selected opening hook: {hook or '(none selected, choose a strong opening yourself)'}")
        parts.append(f"Director's brief (plot research, may be empty): {director_brief or '(none)'}")
    else:
        parts.append(
            f"story_so_far (last sentences of the previous batch, for continuity ONLY — "
            f"do not repeat): {story_so_far or '(none)'}"
        )
    parts.append(
        "Semantic scene blocks (only source of truth for scene_ids/timestamps) — this is "
        f"a SEGMENT of the full film, not the whole thing:\n"
        f"{json.dumps(batch_blocks, ensure_ascii=False)}"
    )
    parts.append(f"Target narration duration for THIS segment: ~{batch_target_duration} seconds.")
    if not is_last:
        parts.append("This is NOT the final segment — do not wrap up the story or add a closing line yet.")
    else:
        parts.append("This IS the final segment — bring the narration to a satisfying emotional close.")
    parts.append("Write the narration for this segment now as the JSON array described in the system prompt.")
    return "\n\n".join(parts)


def generate_narration(
    cfg,
    semantic_blocks: list[dict[str, Any]],
    task_config: dict[str, Any],
    hook: str | None = None,
    director_brief: str = "",
    checkpoint_mgr=None,
) -> list[dict[str, Any]]:
    """
    Hàm chính: sinh lời bình (narration) gắn với scene_ids nguồn.

    Input:
        semantic_blocks: output của semantic_graph.build_semantic_blocks
        task_config: dict {content_type, genre, narration_pov, target_duration_sec, title, ...}
        hook: câu hook đã được chọn (nếu có)
        director_brief: tóm tắt cốt truyện tra cứu được (nếu có)
        checkpoint_mgr: nếu có, mỗi batch narration được lưu micro-checkpoint
            ("narration_batch") để resume được nếu Colab bị ngắt giữa chừng
            (phim dài có thể cần hàng chục batch, mỗi batch tốn 1 API call).

    Output: list[{"sentence_id", "sentence", "scene_ids": [...], "match_reason", "match_score"}]
    LLM chỉ được chọn scene_ids có trong semantic_blocks — không tự bịa timestamp.

    Phim dài (nhiều scene) được chia thành nhiều batch nhỏ để mỗi request gửi
    lên Cerebras luôn nằm trong context window của model (xem
    _batch_semantic_blocks) — trước đây toàn bộ scene bị dồn vào 1 request,
    gây lỗi 'context_length_exceeded' với phim có hàng nghìn scene.
    """
    # Rút gọn semantic_blocks để tiết kiệm token: bỏ dialogue thô dài, giữ tóm tắt.
    compact_blocks = [
        {
            "scene_id": b["scene_id"],
            "start": b["start"],
            "end": b["end"],
            "visual_summary": b["visual_summary"],
            "dialogue_snippets": [d["text"] for d in b["dialogues"][:3]],
            "characters": b["characters"],
            "emotion": b["emotion"],
            "tags": b["tags"],
        }
        for b in semantic_blocks
    ]

    narration_language = task_config.get("narration_language", "Vietnamese")

    # ---- Ngân sách token cho phần scene blocks trong mỗi batch ----
    # Backend-aware: local không bị giới hạn context ngặt nghèo như Cerebras
    # (8192), nên dùng ngân sách lớn hơn nhiều -> ít batch hơn, nhanh hơn, và
    # narration mạch lạc hơn vì model thấy nhiều scene liền một lúc.
    backend = cfg.get("api.script_backend", "local")
    if backend == "local":
        configured_context_tokens = cfg.get("processing.script_local_max_context_tokens", 24000)
        # Model phải được load TRƯỚC khi hỏi ngân sách token an toàn, vì con
        # số này phụ thuộc VRAM thật phát hiện lúc load() (xem
        # LocalScriptModel.recommended_max_context_tokens) — máy yếu tự động
        # nhận batch nhỏ hơn thay vì luôn dùng con số cố định trong config.toml,
        # vốn là nguyên nhân gốc của lỗi CUDA OOM khi chạy trên GPU nhỏ.
        local_model = _get_local_model(cfg)
        max_context_tokens = local_model.recommended_max_context_tokens(configured_context_tokens)
        if max_context_tokens < configured_context_tokens:
            print(f"[script_writer] Giảm ngân sách context từ {configured_context_tokens} xuống "
                  f"{max_context_tokens} token/batch dựa trên VRAM thực tế của máy này (tránh OOM).")
        batch_output_tokens = cfg.get("processing.script_local_max_new_tokens", 3000)
    else:
        max_context_tokens = cfg.get("api.cerebras_max_context_tokens", 8192)
        batch_output_tokens = cfg.get("api.cerebras_narration_batch_max_tokens", 2000)
    base_system_prompt = _narration_system_prompt(narration_language, is_continuation=False)
    fixed_overhead_tokens = (
        _estimate_tokens(base_system_prompt)
        + _estimate_tokens(json.dumps(task_config, ensure_ascii=False))
        + _estimate_tokens(hook or "")
        + _estimate_tokens(director_brief or "")
        + 300  # margin an toàn cho phần khung câu chữ + story_so_far
    )
    max_input_tokens = max(500, max_context_tokens - batch_output_tokens - fixed_overhead_tokens)

    batches = _batch_semantic_blocks(compact_blocks, max_input_tokens)
    n_batches = len(batches)
    total_scenes = len(compact_blocks) or 1
    target_total_duration = task_config.get("target_duration_sec", 180)

    if n_batches > 1:
        print(f"[script_writer] {total_scenes} scene -> chia thành {n_batches} batch narration "
              f"(mỗi batch ~{max_input_tokens} token input) để không vượt context window.")

    resume_done: dict[str, Any] = {}
    if checkpoint_mgr is not None:
        for item_id in checkpoint_mgr.list_micro_done("narration_batch"):
            resume_done[item_id] = checkpoint_mgr.load_micro("narration_batch", item_id)
        if resume_done:
            print(f"[script_writer] Tìm thấy {len(resume_done)}/{n_batches} batch narration "
                  f"đã có checkpoint — sẽ bỏ qua, chỉ chạy phần còn lại.")

    def _generate_for_blocks(
        blocks: list[dict[str, Any]], duration: float, is_first: bool, is_last: bool,
        prior_story_so_far: str, label: str, depth: int = 0,
    ) -> list[dict[str, Any]]:
        """Sinh narration cho 1 nhóm block, TỰ CHIA ĐÔI và thử lại nếu backend
        local báo hết VRAM/RAM (LocalModelOOMError) — cho phép pipeline chạy
        xong trên phần cứng yếu (thay vì crash) bằng cách xử lý từng phần nhỏ
        hơn, đổi lấy nhiều lệnh gọi model hơn. Chỉ backend local mới OOM theo
        kiểu này (cerebras là API từ xa, không tốn VRAM máy mình)."""
        system_prompt = _narration_system_prompt(narration_language, is_continuation=not is_first)
        user_prompt = _narration_user_prompt(
            task_config, hook, director_brief, blocks,
            duration, prior_story_so_far, is_first=is_first, is_last=is_last,
        )
        try:
            return _chat_json(
                cfg, system_prompt, user_prompt, stage=f"narration_batch_{label}",
                max_tokens_override=batch_output_tokens,
            )
        except LocalModelOOMError as e:
            if len(blocks) <= 1 or depth >= 6:
                raise ScriptWriterJSONError(
                    f"Stage 'script': hết VRAM/RAM khi sinh narration cho batch '{label}' dù đã "
                    f"chia nhỏ tới {len(blocks)} scene/lần gọi (depth={depth}). Máy này có thể không "
                    f"đủ tài nguyên để chạy model local — thử đổi 'api.script_backend = \"cerebras\"' "
                    f"trong config.toml, hoặc dùng model nhẹ hơn qua "
                    f"'processing.script_local_model_name'."
                ) from e
            mid = len(blocks) // 2
            print(f"[script_writer] CẢNH BÁO: hết VRAM/RAM ở batch '{label}' ({len(blocks)} scene) — "
                  f"chia đôi và thử lại (sẽ tốn thêm lệnh gọi model nhưng tránh crash pipeline).")
            first_half, second_half = blocks[:mid], blocks[mid:]
            share = mid / len(blocks)
            first_sentences = _generate_for_blocks(
                first_half, round(duration * share, 1), is_first, False,
                prior_story_so_far, f"{label}a", depth + 1,
            )
            bridge_story = " ".join(
                t for t in [s.get("sentence", "") for s in first_sentences[-3:]] if t
            ) or prior_story_so_far
            second_sentences = _generate_for_blocks(
                second_half, round(duration * (1 - share), 1), False, is_last,
                bridge_story, f"{label}b", depth + 1,
            )
            return first_sentences + second_sentences

    all_sentences: list[dict[str, Any]] = []
    story_so_far = ""  # vài câu narration cuối, để batch sau nối mạch chuyện

    for batch_idx, batch_blocks in enumerate(batches):
        item_id = f"{batch_idx:04d}"
        batch_share = len(batch_blocks) / total_scenes
        batch_target_duration = round(target_total_duration * batch_share, 1)
        is_first = batch_idx == 0
        is_last = batch_idx == n_batches - 1

        if item_id in resume_done:
            print(f"[script_writer] Batch narration {batch_idx + 1}/{n_batches}: đã có checkpoint, bỏ qua.")
            batch_sentences = resume_done[item_id]
        else:
            print(f"[script_writer] Sinh narration batch {batch_idx + 1}/{n_batches} "
                  f"({len(batch_blocks)} scene, ~{batch_target_duration}s)...")
            batch_sentences = _generate_for_blocks(
                batch_blocks, batch_target_duration, is_first, is_last,
                story_so_far, str(batch_idx + 1),
            )
            if checkpoint_mgr is not None:
                checkpoint_mgr.save_micro("narration_batch", item_id, batch_sentences)

        all_sentences.extend(batch_sentences)
        last_texts = [s.get("sentence", "") for s in batch_sentences[-3:]]
        story_so_far = " ".join(t for t in last_texts if t)

    if checkpoint_mgr is not None and n_batches > 0:
        checkpoint_mgr.force_sync_micro("narration_batch", f"{n_batches - 1:04d}")

    # Đánh lại sentence_id tuần tự toàn cục — mỗi batch tự đánh số riêng lẻ
    # (vd cả 2 batch đều có thể trả về "sent_001") nên phải renumber sau khi nối.
    for i, s in enumerate(all_sentences):
        s["sentence_id"] = f"sent_{i + 1:03d}"

    # Lọc bỏ mọi scene_id không tồn tại thật trong semantic_blocks (an toàn chống LLM bịa).
    valid_scene_ids = {b["scene_id"] for b in semantic_blocks}
    cleaned = []
    for s in all_sentences:
        scene_ids = [sid for sid in s.get("scene_ids", []) if sid in valid_scene_ids]
        if not scene_ids:
            # Không còn scene_id hợp lệ nào -> đánh dấu review, bỏ qua khi build storyboard.
            continue
        s["scene_ids"] = scene_ids
        cleaned.append(s)

    return cleaned


_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_PUNCT_PATTERN = re.compile(
    r'[，。！？、；：\u201c\u201d\u2018\u2019\u2014\u2026?!.,-]'
)


def _count_narration_chars(sentence: str) -> int:
    """
    Đếm số ký tự 'đọc được' (bỏ dấu câu/khoảng trắng) — dùng cho văn bản chữ Hán
    (1 chữ Hán ≈ 1 âm tiết ≈ 1 đơn vị đọc). Không dùng công thức này cho tiếng Việt/
    tiếng Anh vì 1 chữ cái Latin không tương ứng 1 âm tiết (xem `_is_cjk_dominant`
    và `calc_output_duration`).
    """
    clean = _PUNCT_PATTERN.sub('', sentence)
    clean = re.sub(r'\s', '', clean)
    return len(clean)


def _is_cjk_dominant(sentence: str, threshold: float = 0.3) -> bool:
    """True nếu câu chủ yếu là chữ Hán/CJK (áp dụng công thức chars/4 gốc)."""
    letters = re.sub(r'\s', '', sentence)
    if not letters:
        return False
    cjk_count = len(_CJK_PATTERN.findall(letters))
    return (cjk_count / len(letters)) > threshold


def calc_output_duration(
    sentence: str,
    chars_per_sec: float,
    buffer_after: float,
    min_dur: float,
    words_per_sec: float = 2.5,
) -> float:
    """
    Ước tính output_duration = thời lượng đọc + buffer, tối thiểu min_dur giây.

    - Văn bản CJK (tiếng Trung...): thời lượng đọc = số_ký_tự / chars_per_sec
      (1 chữ Hán ≈ 1 âm tiết, đúng theo thiết kế gốc của skill này).
    - Văn bản chữ Latin (tiếng Việt, tiếng Anh...): số ký tự Latin KHÔNG tỉ lệ
      với số âm tiết đọc ra (vd "không" có 5 ký tự nhưng chỉ 1 âm tiết), nên
      dùng số từ (word count) / words_per_sec thay vì đếm ký tự.

    Đây vẫn là ước tính dùng để xây dựng timeline TRƯỚC KHI tổng hợp TTS thật.
    Sau khi TTS chạy (tts.py), thời lượng thực tế được đo lại và ghi vào
    `tts_report.json` để người dùng đối chiếu nếu lệch nhiều.
    """
    if _is_cjk_dominant(sentence):
        char_count = _count_narration_chars(sentence)
        speech_dur = char_count / chars_per_sec
    else:
        clean = _PUNCT_PATTERN.sub('', sentence)
        word_count = len(clean.split())
        speech_dur = word_count / words_per_sec
    return max(speech_dur + buffer_after, min_dur)


def build_storyboard(
    cfg,
    task_config: dict[str, Any],
    narration_sentences: list[dict[str, Any]],
    semantic_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Xây storyboard.json từ narration_sentences + semantic_blocks.
    Timestamp nguồn lấy trực tiếp từ block thật (min/max của các scene_ids được chọn),
    timestamp output tính tuần tự bằng calc_output_duration — không phụ thuộc LLM.
    """
    blocks_by_id = {b["scene_id"]: b for b in semantic_blocks}
    chars_per_sec = cfg.get("processing.chars_per_sec", 4.0)
    words_per_sec = cfg.get("processing.words_per_sec", 2.5)
    buffer_after = cfg.get("processing.buffer_after_speech", 0.1)
    min_dur = cfg.get("processing.min_clip_duration", 1.0)

    timeline = []
    cursor = 0.0
    for i, sent in enumerate(narration_sentences):
        scene_ids = sent["scene_ids"]
        blocks = [blocks_by_id[sid] for sid in scene_ids if sid in blocks_by_id]
        if not blocks:
            continue

        src_start = min(b["start"] for b in blocks)
        src_end = max(b["end"] for b in blocks)

        out_dur = calc_output_duration(sent["sentence"], chars_per_sec, buffer_after, min_dur, words_per_sec)
        out_start = cursor
        out_end = cursor + out_dur
        cursor = out_end

        visual_summary = " / ".join(b["visual_summary"] for b in blocks if b["visual_summary"])

        clip = {
            "clip_id": f"clip_{i + 1:03d}",
            "sentence_id": sent.get("sentence_id", f"sent_{i + 1:03d}"),
            "sentence": sent["sentence"],
            "source": {"scene_ids": scene_ids, "start": round(src_start, 2), "end": round(src_end, 2)},
            "output": {"start": round(out_start, 3), "end": round(out_end, 3)},
            "visual_summary": visual_summary,
            "match_reason": sent.get("match_reason", ""),
            "match_score": sent.get("match_score", 0.0),
            "edit": {"crop": "9:16_center", "speed": 1.0, "original_audio": "duck", "transition": "cut"},
            "review_flags": [],
        }
        timeline.append(clip)

    storyboard = {
        "task": {
            "input_video": str(cfg.resolve_path("paths.input_video")),
            "target_duration": task_config.get("target_duration_sec", 180),
            "narration_pov": task_config.get("narration_pov", "third_person"),
            "content_type": task_config.get("content_type", "movie"),
            "genre": task_config.get("genre", "drama"),
        },
        "timeline": timeline,
    }
    return storyboard


def validate_storyboard_against_sources(
    storyboard: dict[str, Any],
    asr_timeline: list[dict[str, Any]],
    vision_analysis: list[dict[str, Any]],
    max_vision_gap_sec: float = 15.0,
) -> list[dict[str, Any]]:
    """
    Đối chiếu mỗi clip với asr_timeline + vision_analysis (Step 7 trong skill.md).
    Trả về danh sách cảnh báo (không sửa tự động — để pipeline log/flag rõ ràng).
    """
    warnings = []
    vision_by_scene = {v["scene_id"]: v for v in vision_analysis}

    for clip in storyboard["timeline"]:
        src = clip["source"]
        mid = (src["start"] + src["end"]) / 2

        overlapping_asr = [
            seg for seg in asr_timeline
            if seg["start"] < src["end"] and src["start"] < seg["end"]
        ]
        if not overlapping_asr:
            # Không có thoại trong khoảng này — chỉ là cảnh báo nhẹ, không phải lỗi.
            pass

        nearest_vision = None
        best_gap = float("inf")
        for sid in src.get("scene_ids", []):
            v = vision_by_scene.get(sid)
            if v is None:
                continue
            vmid = (v.get("start", mid) + v.get("end", mid)) / 2
            gap = abs(vmid - mid)
            if gap < best_gap:
                best_gap = gap
                nearest_vision = v

        if nearest_vision is None:
            warnings.append({
                "clip_id": clip["clip_id"],
                "issue": "no_matching_vision_scene",
            })
        elif best_gap > max_vision_gap_sec:
            warnings.append({
                "clip_id": clip["clip_id"],
                "issue": "vision_gap_too_large",
                "gap_sec": round(best_gap, 2),
            })
            clip["review_flags"].append(f"vision_gap_{round(best_gap, 1)}s")

    return warnings


def run_script_writer(
    cfg,
    task_config: dict[str, Any],
    semantic_blocks: list[dict[str, Any]],
    asr_timeline: list[dict[str, Any]],
    vision_analysis: list[dict[str, Any]],
    hook: str | None = None,
    director_brief: str = "",
    checkpoint_mgr=None,
) -> dict[str, Any]:
    """Entry point cho stage 'script'. Ghi storyboard.json vào pipeline/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    backend = cfg.get("api.script_backend", "local")
    print(f"[script_writer] Generating narration (backend='{backend}')...")
    try:
        narration_sentences = generate_narration(
            cfg, semantic_blocks, task_config, hook, director_brief, checkpoint_mgr=checkpoint_mgr,
        )
    finally:
        # Giải phóng model local (nếu có) ngay sau khi dùng xong, để nhường
        # VRAM/RAM cho các stage sau (tts, render) — an toàn khi backend là
        # 'cerebras' (hàm tự bỏ qua nếu chưa từng load model local).
        unload_local_script_model()

    print(f"[script_writer] {len(narration_sentences)} câu narration được sinh. Building storyboard...")
    storyboard = build_storyboard(cfg, task_config, narration_sentences, semantic_blocks)

    warnings = validate_storyboard_against_sources(storyboard, asr_timeline, vision_analysis)
    if warnings:
        print(f"[script_writer] Cảnh báo validate: {len(warnings)} clip cần xem lại.")
        for w in warnings:
            print(f"  - {w}")

    with open(pipeline_dir / "storyboard.json", "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("script", storyboard)

    return storyboard
