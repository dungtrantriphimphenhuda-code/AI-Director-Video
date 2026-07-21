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
      - "local" (MẶC ĐỊNH): model nhẹ chạy ngay trên máy/Colab (Gemma 4 E4B QAT,
        GGUF qua llama.cpp), không cần API key, không giới hạn context nhỏ,
        không có rủi ro "nuốt hết token vào suy luận nội bộ" như model reasoning.
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
# Backend "local" — model nhẹ chạy ngay trên máy (mặc định Gemma 4 E4B QAT,
# GGUF qua llama.cpp — trước đây là Qwen3-4B-Instruct-2507 qua transformers)
# =============================================================================
#
# Lý do đổi mặc định sang local: model reasoning qua Cerebras (zai-glm-4.7) có
# thể "nuốt" hết ngân sách token vào suy luận nội bộ trước khi kịp viết JSON
# (xem BUGFIX trong _chat_cerebras), đặc biệt dễ xảy ra khi narration được
# chia thành nhiều batch nhỏ cho phim dài. Chạy local tránh hẳn vấn đề đó
# (không cần API key, không rate limit, không giới hạn context 8K khắt khe).
#
# Lý do đổi từ Qwen3-4B-Instruct-2507 (transformers) sang Gemma 4 E4B QAT
# (GGUF, llama.cpp): bản Qwen khi chạy CPU (vd runner GitHub Actions không có
# GPU) tự fallback dtype 'bfloat16' -> 'float32' — riêng trọng số model 4B ở
# float32 đã tốn ~16GB RAM, khiến runner CI bị đói RAM giữa lúc load model và
# lỗi "The hosted runner lost communication with the server" (xem
# NOTES-AI-DIRECTOR-VIDEO.md). GGUF QAT của Gemma 4 E4B nhẹ hơn hẳn (~5GB RAM,
# chất lượng gần bằng bfloat16 gốc nhờ Quantization-Aware Training) và chạy
# qua llama.cpp — vốn được tối ưu cho CPU tốt hơn nhiều so với việc "giả lập"
# 8bit bằng bitsandbytes trên transformers. Muốn quay lại Cerebras: đổi
# `api.script_backend = "cerebras"` trong config.toml.

_local_model_singleton: "LocalScriptModel | None" = None


class LocalModelOOMError(RuntimeError):
    """Hết RAM/context khi sinh text bằng model local, kể cả sau khi đã thử
    giảm max_tokens + dọn cache. Caller (vd generate_narration) có thể bắt lỗi
    này để CHIA NHỎ batch input rồi thử lại, thay vì để cả pipeline crash."""


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error" in text
        or "failed to allocate" in text
        or ("exceed" in text and ("context" in text or "n_ctx" in text))
        or "n_ctx" in text
        or isinstance(exc, MemoryError)
    )


class LocalScriptModel:
    """Bọc model GGUF (llama.cpp) + cấu hình, load một lần và tái sử dụng cho
    toàn bộ hooks + mọi batch narration của 1 project (tránh load lại model
    nặng cho mỗi lệnh gọi).

    ĐÃ ĐỔI từ Qwen3-4B-Instruct-2507 (transformers + bitsandbytes) sang
    Gemma 4 E4B bản QAT (Quantization-Aware Training), repack GGUF bởi
    Unsloth, chạy qua llama-cpp-python thay vì transformers — nhẹ hơn nhiều
    trên CPU (không có bước "giả lập" 8bit tốn RAM của bitsandbytes, vốn là
    nguyên nhân runner GitHub Actions từng bị OOM/"lost communication with
    the server" khi load Qwen3-4B ở float32 — xem NOTES-AI-DIRECTOR-VIDEO.md).
    Repo Hugging Face public, không gate, không cần hf_token.

    Khác biệt quan trọng so với bản transformers cũ: llama.cpp CẤP PHÁT SẴN
    RAM cho KV cache ngay lúc load model theo đúng n_ctx được truyền vào (thay
    vì context là thuộc tính "vốn có" của model, có thể hỏi thêm lúc chạy) —
    nên n_ctx phải được ước lượng AN TOÀN dựa trên RAM trống lúc load, chứ
    không thể "xin thêm" giữa chừng như trước.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.repo_id = cfg.get("processing.script_local_gguf_repo", "unsloth/gemma-4-E4B-it-qat-GGUF")
        self.filename = cfg.get("processing.script_local_gguf_filename", "*UD-Q4_K_XL*.gguf")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = resolve_torch_device(cfg.get("processing.script_local_device", "auto"))
        self.gpu_layers_cfg = cfg.get("processing.script_local_gpu_layers", "auto")
        self.temperature = cfg.get("processing.script_local_temperature", 0.7)
        self.top_p = cfg.get("processing.script_local_top_p", 0.8)
        self.top_k = cfg.get("processing.script_local_top_k", 20)
        self.configured_max_context_tokens = cfg.get("processing.script_local_max_context_tokens", 16000)
        self.model = None
        self.max_model_context: int = 4096  # ghi đè thật sau khi load(), đây chỉ là giá trị an toàn mặc định

    def load(self) -> None:
        import os
        # Tải nhanh hơn qua Hugging Face Hub (package hf_transfer đã thêm vào
        # requirements.txt) — không ảnh hưởng gì nếu package thiếu, chỉ tắt
        # tính năng tăng tốc tải.
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        from llama_cpp import Llama

        from platform_utils import get_free_ram_gb
        free_ram_gb = get_free_ram_gb()

        # n_ctx (context window) PHẢI cấp TRƯỚC khi load vì llama.cpp cấp phát
        # RAM cho KV cache ngay lúc tạo model theo đúng con số này — để quá
        # cao dễ OOM ngay LÚC LOAD (khác lỗi lúc generate). Tự giảm n_ctx nếu
        # RAM trống lúc load thấp, giống tinh thần "tự thích ứng phần cứng"
        # của bản Qwen/transformers cũ nhưng áp dụng SỚM hơn (trước khi load
        # thay vì trong lúc chat()).
        n_ctx = self.configured_max_context_tokens
        if free_ram_gb is not None:
            if free_ram_gb < 6.0:
                n_ctx = min(n_ctx, 4096)
            elif free_ram_gb < 10.0:
                n_ctx = min(n_ctx, 8192)
            if n_ctx < self.configured_max_context_tokens:
                print(f"[script_writer] RAM trống lúc load chỉ ~{free_ram_gb:.1f}GB — giảm n_ctx "
                      f"từ {self.configured_max_context_tokens} xuống {n_ctx} để tránh OOM lúc tải model.")

        n_gpu_layers = self.gpu_layers_cfg
        if n_gpu_layers == "auto":
            n_gpu_layers = -1 if self.device == "cuda" else 0

        n_threads = os.cpu_count() or 4
        ram_note = f", RAM trống ~{free_ram_gb:.1f}GB" if free_ram_gb is not None else ""
        print(f"[script_writer] Loading local model (GGUF) {self.repo_id}:{self.filename} "
              f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, n_threads={n_threads}{ram_note})... "
              f"(lần đầu sẽ tải model; log tải % của huggingface_hub sẽ hiện ngay bên dưới)")

        load_kwargs = dict(
            repo_id=self.repo_id,
            filename=self.filename,
            cache_dir=self.cache_dir,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        try:
            self.model = Llama.from_pretrained(**load_kwargs)
        except Exception as e:
            if not _is_oom_error(e) or n_ctx <= 2048:
                raise
            # RAM lúc load có thể đã bị stage trước (vision, ASR) giữ chỗ
            # nhiều hơn ước tính (get_free_ram_gb chỉ đo tại 1 thời điểm) —
            # thử lại 1 lần với n_ctx tối thiểu trước khi bó tay hẳn, giống
            # tinh thần "giảm rồi thử lại" của bản Qwen/transformers cũ.
            fallback_ctx = 2048
            print(f"[script_writer] CẢNH BÁO: lỗi khi tải model với n_ctx={n_ctx} ({e}) — "
                  f"thử lại với n_ctx={fallback_ctx}.")
            load_kwargs["n_ctx"] = fallback_ctx
            self.model = Llama.from_pretrained(**load_kwargs)
            n_ctx = fallback_ctx

        self.max_model_context = n_ctx

    def recommended_max_context_tokens(self, configured_default: int) -> int:
        """Ngân sách token INPUT an toàn cho mỗi batch narration — bị giới
        hạn CỨNG bởi n_ctx thật sự đã cấp cho llama.cpp lúc load() (không thể
        "xin thêm" tuỳ RAM còn dư như bản transformers cũ, vì KV cache đã
        được cấp phát cố định từ đầu)."""
        return min(configured_default, self.max_model_context)

    def unload(self) -> None:
        self.model = None
        gc.collect()

    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        """Sinh text qua chat template gắn sẵn trong GGUF (llama-cpp-python tự
        đọc chat_template từ metadata của Gemma 4 — không cần tự soạn prompt
        template tay như bản transformers cũ).

        Tự phục hồi khi hết RAM/context giữa lúc generate: dọn cache + thử
        lại với max_tokens nhỏ hơn trước khi bó tay và raise LocalModelOOMError
        (để caller — generate_narration — chia nhỏ batch input rồi thử lại)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Ước lượng input token bằng tokenizer thật của model (qua llama.cpp)
        # để tính ngân sách output còn lại — QUAN TRỌNG vì llama.cpp raise lỗi
        # cứng nếu input+output vượt n_ctx, KHÔNG tự cắt bớt như transformers.
        try:
            prompt_text = system_prompt + "\n" + user_prompt
            input_len = len(self.model.tokenize(prompt_text.encode("utf-8")))
        except Exception:
            input_len = len(system_prompt + user_prompt) // 3  # ước lượng thô nếu tokenize lỗi

        safety_margin = 64
        context_budget = max(256, self.max_model_context - input_len - safety_margin)
        attempt_tokens = min(max_new_tokens, context_budget) if max_new_tokens else context_budget
        if max_new_tokens and max_new_tokens > context_budget:
            print(f"[script_writer] Lưu ý: input dài ~{input_len} token, model chỉ còn "
                  f"~{context_budget} token cho output (n_ctx={self.max_model_context}) — "
                  f"giảm max_new_tokens xuống {attempt_tokens}.")

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                print_progress_bar(
                    0, 1, prefix="[script_writer] local",
                    suffix=f"đang sinh (max {attempt_tokens} token, input ~{input_len} token)...",
                )
                result = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=attempt_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
                print_progress_bar(1, 1, prefix="[script_writer] local", suffix="xong")
                return result["choices"][0]["message"]["content"].strip()
            except Exception as e:  # bắt rộng vì lỗi hết RAM/context của llama.cpp không có type riêng cố định
                if not _is_oom_error(e):
                    raise
                last_err = e
                gc.collect()
                if attempt_tokens > 256:
                    attempt_tokens = max(256, attempt_tokens // 2)
                    print(f"[script_writer] CẢNH BÁO: hết RAM/context lúc generate — thử lại với "
                          f"max_tokens giảm còn {attempt_tokens} (lần {attempt + 1}/3).")
                    continue
                break
        raise LocalModelOOMError(
            f"Hết RAM/context khi sinh text (input ~{input_len} token, n_ctx={self.max_model_context}) "
            f"dù đã giảm max_tokens xuống {attempt_tokens} và thử lại 3 lần liên tiếp: {last_err}"
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
    như mô tả trong skill.md Step 3. Trả về list [{"style": ..., "text": ...,
    "potential_score": ...}], đã SẮP XẾP GIẢM DẦN theo "potential_score"
    (hook có tiềm năng viral cao nhất luôn ở vị trí #1).

    Việc sắp xếp này quan trọng vì run.choose_hook() dùng hook #1 làm lựa
    chọn mặc định — cả khi người dùng nhấn Enter lẫn khi người dùng không
    nhập gì trong thời gian chờ (xem processing.hook_selection_timeout_sec) —
    nên #1 luôn phải là hook tiềm năng nhất, không chỉ là hook đầu tiên LLM
    liệt kê ra.
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
        '"language_used": "...", "potential_score": 1-10}]. '
        '"potential_score" is YOUR honest estimate of how likely this specific hook is '
        "to stop someone from scrolling in the first 1-2 seconds (10 = extremely likely). "
        "Vary the scores realistically, don't give every hook the same score. "
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
    hooks = _chat_json(cfg, system_prompt, user_prompt, stage="hooks")
    if isinstance(hooks, list):
        # Hook thiếu "potential_score" (model bỏ sót field) -> xếp cuối thay
        # vì coi là điểm cao nhất, tránh 1 hook không chấm điểm bị đẩy lên #1.
        hooks.sort(key=lambda h: h.get("potential_score", 0) if isinstance(h, dict) else 0, reverse=True)
    return hooks


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
    parts.append(
        f"Rough timing guide for THIS segment: ~{batch_target_duration} seconds — a loose fallback "
        "sized by scene count, NOT a quota. Let the actual dramatic weight of these scenes decide "
        "the real narration length; do not pad or cut sentences just to hit this number."
    )
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

    duration_mode = cfg.get("processing.target_duration_mode", "auto")
    if duration_mode == "fixed":
        cleaned = _enforce_target_duration(cfg, cleaned, target_total_duration)
    else:
        # "auto" (mặc định): KHÔNG ép tổng thời lượng về 1 con số cứng — mỗi
        # phim có độ dài nội dung tự nhiên khác nhau (cảnh cao trào cần
        # nhiều lời bình hơn cảnh chuyển tiếp), ép cắt theo 1 target chung
        # cho mọi video dễ làm mất chỗ đang cần và giữ chỗ không cần (xem
        # NOTES-AI-DIRECTOR-VIDEO.md). target_duration_sec ở đây chỉ còn là
        # GỢI Ý mềm để chia batch_target_duration cho từng batch khi soạn
        # (xem vòng lặp batch phía trên) — không có bước cắt cứng nào sau
        # khi gộp. Chỉ IN ra ước tính để người dùng biết, không tự sửa.
        _log_duration_estimate(cfg, cleaned, target_total_duration)

    return cleaned


def _log_duration_estimate(
    cfg,
    sentences: list[dict[str, Any]],
    target_total_duration: float,
) -> None:
    """In ra tổng thời lượng ước tính (chế độ 'auto', không cắt) để người dùng
    biết trước khi tốn thời gian TTS + render — KHÔNG thay đổi `sentences`."""
    if not sentences:
        return
    chars_per_sec = cfg.get("processing.chars_per_sec", 4.0)
    words_per_sec = cfg.get("processing.words_per_sec", 2.5)
    safety_margin = cfg.get("processing.speech_safety_margin", 1.0)
    buffer_after = cfg.get("processing.buffer_after_speech", 0.1)
    min_dur = cfg.get("processing.min_clip_duration", 1.0)
    total = sum(
        calc_output_duration(s.get("sentence", ""), chars_per_sec, buffer_after, min_dur, words_per_sec, safety_margin)
        for s in sentences
    )
    ratio = total / target_total_duration if target_total_duration else 0.0
    print(
        f"[script_writer] processing.target_duration_mode='auto' — không cắt cứng. "
        f"Tổng narration ước tính: {total:.1f}s ({len(sentences)} câu) so với gợi ý mềm "
        f"{target_total_duration:.0f}s (x{ratio:.2f}). Nếu muốn ép về đúng "
        f"{target_total_duration:.0f}s, đặt processing.target_duration_mode = \"fixed\"."
    )


def _enforce_target_duration(
    cfg,
    sentences: list[dict[str, Any]],
    target_total_duration: float,
) -> list[dict[str, Any]]:
    """
    [CHỈ CHẠY khi processing.target_duration_mode = "fixed", opt-in — mặc định
    pipeline dùng chế độ "auto" và KHÔNG gọi hàm này, xem _log_duration_estimate]

    Cắt bớt narration nếu TỔNG thời lượng ước tính (cộng dồn qua mọi batch)
    vượt quá target_duration_sec quá xa.

    Trước đây mỗi batch chỉ được giao một `batch_target_duration` TỈ LỆ theo
    số scene của batch đó (xem generate_narration), nhưng không có bước nào
    kiểm tra lại TỔNG sau khi gộp — nếu model viết dài hơn target ở mỗi batch
    (rất hay xảy ra), độ lệch cộng dồn qua hàng chục batch khiến video phình to
    (video 3 phút mục tiêu ra thành 25+ phút thực tế trong trường hợp thực đo).

    Cách cắt: dùng đúng công thức ước tính thời lượng mà build_storyboard()
    sẽ dùng thật (calc_output_duration, với chars_per_sec/words_per_sec/
    buffer_after_speech/min_clip_duration đọc từ config — PHẢI khớp giá trị
    thật của giọng TTS đang dùng, xem processing.words_per_sec). Nếu tổng vượt
    quá target * tolerance, loại bỏ dần câu có match_score THẤP NHẤT (câu khớp
    cảnh yếu nhất) cho tới khi vừa ngân sách — luôn giữ câu đầu (hook/mở đầu)
    và câu cuối (kết truyện) để narration không bị cụt đầu/đuôi.
    """
    if not sentences:
        return sentences

    tolerance = cfg.get("processing.target_duration_tolerance", 1.15)
    budget = target_total_duration * tolerance

    chars_per_sec = cfg.get("processing.chars_per_sec", 4.0)
    words_per_sec = cfg.get("processing.words_per_sec", 2.5)
    safety_margin = cfg.get("processing.speech_safety_margin", 1.0)
    buffer_after = cfg.get("processing.buffer_after_speech", 0.1)
    min_dur = cfg.get("processing.min_clip_duration", 1.0)

    def _dur(s: dict[str, Any]) -> float:
        return calc_output_duration(
            s.get("sentence", ""), chars_per_sec, buffer_after, min_dur, words_per_sec, safety_margin
        )

    kept = list(sentences)
    total = sum(_dur(s) for s in kept)

    if total <= budget or len(kept) <= 2:
        return kept

    original_total = total
    original_count = len(kept)

    # Chỉ số 1..len-2 là "có thể bỏ" — luôn giữ câu đầu và câu cuối.
    # Sắp theo match_score tăng dần: bỏ câu khớp cảnh yếu nhất trước.
    droppable_idx = sorted(
        range(1, len(kept) - 1),
        key=lambda i: kept[i].get("match_score", 0.0),
    )
    to_drop: set[int] = set()
    for i in droppable_idx:
        if total <= budget:
            break
        total -= _dur(kept[i])
        to_drop.add(i)

    if to_drop:
        kept = [s for i, s in enumerate(kept) if i not in to_drop]
        # Đánh lại sentence_id tuần tự sau khi cắt.
        for i, s in enumerate(kept):
            s["sentence_id"] = f"sent_{i + 1:03d}"
        print(
            f"[script_writer] Tổng narration ước tính {original_total:.1f}s vượt ngân sách "
            f"~{budget:.1f}s (target {target_total_duration:.0f}s x{tolerance}) — đã cắt "
            f"{len(to_drop)}/{original_count} câu (khớp cảnh yếu nhất) còn lại {total:.1f}s."
        )
    else:
        print(
            f"[script_writer] CẢNH BÁO: tổng narration ước tính {original_total:.1f}s vượt "
            f"ngân sách ~{budget:.1f}s nhưng không còn câu nào có thể cắt bớt (chỉ có "
            f"{original_count} câu, đã giữ câu đầu/cuối)."
        )

    return kept


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
    safety_margin: float = 1.0,
) -> float:
    """
    Ước tính output_duration = thời lượng đọc + buffer, tối thiểu min_dur giây.

    - Văn bản CJK (tiếng Trung...): thời lượng đọc = số_ký_tự / chars_per_sec
      (1 chữ Hán ≈ 1 âm tiết, đúng theo thiết kế gốc của skill này).
    - Văn bản chữ Latin (tiếng Việt, tiếng Anh...): số ký tự Latin KHÔNG tỉ lệ
      với số âm tiết đọc ra (vd "không" có 5 ký tự nhưng chỉ 1 âm tiết), nên
      dùng số từ (word count) / words_per_sec thay vì đếm ký tự.
    - safety_margin (0 < x <= 1, mặc định 1.0 = tắt): hệ số dự phòng nhân vào
      chars_per_sec/words_per_sec TRƯỚC khi chia, tức là "giả vờ" giọng đọc
      chậm hơn thực tế đo được một chút. TTS thực tế dao động ±10-20% tuỳ câu
      (ngữ điệu, dấu câu, độ dài), nên nếu ước tính đúng bằng giá trị đo trung
      vị, khoảng nửa số câu sẽ đọc CHẬM hơn ước tính -> lại tràn ra ngoài slot
      -> lặp lại đúng lỗi khoảng lặng ban đầu ở quy mô nhỏ hơn. Dùng ví dụ
      0.85 (mặc định processing.speech_safety_margin) nghĩa là luôn chừa dư
      ~15% thời gian, che được phần lớn dao động thực tế mà không cần đo lại
      từng câu (ý tưởng lấy từ speech_safety_margin trong bộ skill
      video-recap-skills — xem NOTES-AI-DIRECTOR-VIDEO.md).

    Đây vẫn là ước tính dùng để xây dựng timeline TRƯỚC KHI tổng hợp TTS thật.
    Sau khi TTS chạy (tts.py), thời lượng thực tế được đo lại và ghi vào
    `tts_report.json` để người dùng đối chiếu nếu lệch nhiều.
    """
    safety_margin = min(max(safety_margin, 0.1), 1.0)
    if _is_cjk_dominant(sentence):
        char_count = _count_narration_chars(sentence)
        speech_dur = char_count / (chars_per_sec * safety_margin)
    else:
        clean = _PUNCT_PATTERN.sub('', sentence)
        word_count = len(clean.split())
        speech_dur = word_count / (words_per_sec * safety_margin)
    return max(speech_dur + buffer_after, min_dur)


def compute_pacing_report(cfg, storyboard: dict[str, Any]) -> dict[str, Any]:
    """
    Chẩn đoán tốc độ từng clip — KHÔNG cắt/sửa gì, chỉ báo cáo (giống triết lý
    "coverage is diagnostic, not a creative quota" trong bộ skill
    video-recap-skills — xem NOTES-AI-DIRECTOR-VIDEO.md). Lý do cần tách riêng
    khỏi việc ép tổng thời lượng: 1 video có thể có TỔNG thời lượng hợp lý
    nhưng vẫn có VÀI clip riêng lẻ bị "gấp" (source quá ngắn so với lời bình,
    render.py phải tua nhanh clip gốc) hoặc "lê thê" (source quá dài so với
    lời bình, clip gốc bị chiếu chậm/kéo dài) — ép tổng không phát hiện được
    lỗi cục bộ kiểu này.

    speed = src_span / out_dur, ĐÚNG công thức render.render_clip dùng để
    quyết định tốc độ tua nhanh/chậm clip gốc thật:
      - speed > max_speed_ratio -> clip gốc bị tua nhanh quá mức (rushed/gấp)
      - speed < min_speed_ratio -> clip gốc bị kéo chậm quá mức (dragging/lê thê)
    """
    max_speed_ratio = cfg.get("processing.max_speed_ratio", 4.0)
    min_speed_ratio = cfg.get("processing.min_speed_ratio", 0.5)

    flags = []
    speeds = []
    for clip in storyboard.get("timeline", []):
        src = clip["source"]
        out = clip["output"]
        src_span = max(src["end"] - src["start"], 0.01)
        out_dur = max(out["end"] - out["start"], 0.01)
        speed = src_span / out_dur
        speeds.append(speed)
        if speed > max_speed_ratio:
            flags.append({
                "clip_id": clip["clip_id"],
                "issue": "rushed",
                "speed": round(speed, 2),
                "detail": (
                    f"Cảnh gốc dài {src_span:.1f}s nhưng lời bình chỉ {out_dur:.1f}s -> "
                    f"clip gốc sẽ bị tua nhanh {speed:.1f}x (> max_speed_ratio={max_speed_ratio})."
                ),
            })
        elif speed < min_speed_ratio:
            flags.append({
                "clip_id": clip["clip_id"],
                "issue": "dragging",
                "speed": round(speed, 2),
                "detail": (
                    f"Cảnh gốc chỉ dài {src_span:.1f}s nhưng lời bình cần {out_dur:.1f}s -> "
                    f"clip gốc sẽ bị kéo chậm còn {speed:.2f}x (< min_speed_ratio={min_speed_ratio}), "
                    "dễ trông lê thê/đơ hình."
                ),
            })

    n = len(speeds) or 1
    report = {
        "clip_count": len(speeds),
        "avg_speed": round(sum(speeds) / n, 2) if speeds else 0.0,
        "rushed_count": sum(1 for f in flags if f["issue"] == "rushed"),
        "dragging_count": sum(1 for f in flags if f["issue"] == "dragging"),
        "max_speed_ratio": max_speed_ratio,
        "min_speed_ratio": min_speed_ratio,
        "flags": flags,
    }
    return report


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
    safety_margin = cfg.get("processing.speech_safety_margin", 1.0)
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

        out_dur = calc_output_duration(
            sent["sentence"], chars_per_sec, buffer_after, min_dur, words_per_sec, safety_margin
        )
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

    pacing_report = compute_pacing_report(cfg, storyboard)
    with open(pipeline_dir / "pacing_report.json", "w", encoding="utf-8") as f:
        json.dump(pacing_report, f, ensure_ascii=False, indent=2)
    if pacing_report["rushed_count"] or pacing_report["dragging_count"]:
        print(
            f"[script_writer] pacing_report.json: {pacing_report['rushed_count']} clip có thể bị GẤP, "
            f"{pacing_report['dragging_count']} clip có thể bị LÊ THÊ trên tổng {pacing_report['clip_count']} "
            "clip (chỉ chẩn đoán, không tự sửa — xem chi tiết trong file để chỉnh lại câu văn/scene nếu cần)."
        )

    with open(pipeline_dir / "storyboard.json", "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("script", storyboard)

    return storyboard
