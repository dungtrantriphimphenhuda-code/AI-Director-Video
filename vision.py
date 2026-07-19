"""
vision.py — phân tích thị giác cho từng scene.

Hỗ trợ 4 backend, chọn qua config.toml [processing] vision_backend:
  - "local"    : Qwen3-VL-4B-Instruct tải từ Hugging Face, chạy qua `transformers`
                 (cần GPU tốt, chậm với video nhiều scene vì generate() tuần tự
                 từng scene một; trên CPU thuần như GitHub Actions runner thì
                 KHÔNG khả thi với video nhiều scene — quá chậm).
  - "cerebras" : Gemma 4 31B multimodal qua Cerebras API (OpenAI-compatible),
                 dùng chung cerebras_api_key đã có trong [api]. Nhanh hơn nhiều
                 cho video dài vì chạy trên wafer-scale chip, không tốn thời
                 gian tải/giữ model 4B trong VRAM Colab.
  - "mistral"  : Model multimodal của Mistral (mặc định Mistral Large — API id
                 "mistral-large-latest", model lớn nhất/vision tốt nhất hiện có
                 của Mistral: 675B tổng/41B active MoE, vision encoder tích hợp
                 sẵn) qua API chính thức Mistral (OpenAI-compatible), dùng
                 mistral_api_key riêng trong [api]. Không dùng chung engine với
                 script_writer.py (vẫn là Cerebras/GLM) — chỉ thay backend đọc
                 ảnh cho stage vision. Phù hợp cho máy không GPU/RAM thấp.
  - "moondream": Moondream2 (~1.9B, dense), tải từ Hugging Face, chạy CPU-only
                 qua `transformers` — được thiết kế riêng cho CPU/edge nên
                 nhanh hơn nhiều lần so với "local" (Qwen3-VL-4B) trên cùng
                 phần cứng không GPU (đúng trường hợp GitHub Actions runner).
                 Đánh đổi: API gốc của Moondream2 chỉ nhận 1 ảnh/lần hỏi (không
                 có multi-image chat template như Qwen), nên backend này CHỈ
                 dùng 1 keyframe đại diện/scene thay vì gộp cả
                 vision_frames_per_scene ảnh — đổi lấy tốc độ, chất lượng đọc
                 cảnh (đặc biệt các trường suy luận như emotion/visual_intensity)
                 sẽ kém tinh tế hơn Qwen3-VL-4B hoặc các backend API lớn.

Output JSON giữ nguyên schema `vision_analysis.json` mô tả trong
ref-asr-vision-pipeline.md để không phá vỡ các stage sau (semantic graph,
script writer, storyboard) vốn đã được thiết kế để tiêu thụ schema đó.
"""

from __future__ import annotations

import base64
import gc
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from platform_utils import resolve_torch_device
from progress_utils import print_progress_bar

VISION_SYSTEM_PROMPT = (
    "You are a visual analyst for a video commentary pipeline. "
    "Look at the provided frames from one video scene and describe concrete, "
    "visible facts first, then a short interpretation. "
    "Respond ONLY with a single JSON object with these exact keys: "
    "visual_summary (string), characters (array of strings), location (string), "
    "actions (array of strings), emotion (string), shot_type (string), "
    "visual_intensity (number 0-1), tags (array of strings). "
    "No markdown, no extra text, only the JSON object."
)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Cố gắng parse JSON từ output model; nếu lỗi, trả về kết quả rỗng an toàn."""
    text = text.strip()
    # Model đôi khi bọc JSON trong ```json ... ``` dù đã được yêu cầu không làm vậy.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: tìm { ... } đầu tiên trong chuỗi
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return {
        "visual_summary": data.get("visual_summary", ""),
        "characters": data.get("characters", []),
        "location": data.get("location", ""),
        "actions": data.get("actions", []),
        "emotion": data.get("emotion", ""),
        "shot_type": data.get("shot_type", ""),
        "visual_intensity": float(data.get("visual_intensity", 0.0) or 0.0),
        "tags": data.get("tags", []),
    }


def _empty_result(scene_id: str, reason: str) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "visual_summary": "",
        "characters": [],
        "location": "",
        "actions": [],
        "emotion": "",
        "shot_type": "",
        "visual_intensity": 0.0,
        "tags": [],
        "review_flag": reason,
    }


# =============================================================================
# Backend "local" — Qwen3-VL-4B-Instruct qua transformers
# =============================================================================

class LocalVisionAnalyzer:
    """Bọc model + processor Qwen3-VL-4B-Instruct, load một lần và tái sử dụng."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.vision_model_name", "Qwen/Qwen3-VL-4B-Instruct")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = resolve_torch_device(cfg.get("processing.vision_device", "auto"))
        self.dtype_name = cfg.get("processing.vision_dtype", "float16")
        self.max_new_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.attn_implementation = cfg.get("processing.vision_attn_implementation", "sdpa")
        self.batch_size = max(1, cfg.get("processing.vision_batch_size", 4))
        self.model = None
        self.processor = None
        self._torch = None

    def load(self) -> None:
        """Load model + processor vào GPU/CPU. Gọi 1 lần trước khi phân tích cả loạt scene."""
        import torch  # lazy import: chỉ cần khi thực sự dùng backend "local"
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            self.dtype_name, torch.float16
        )
        self.dtype = dtype

        print(f"[vision] Loading {self.model_name} on {self.device} ({self.dtype_name}, "
              f"attn={self.attn_implementation}, batch_size={self.batch_size})... "
              f"(lần đầu sẽ tải model; log tải % / tốc độ của huggingface_hub sẽ hiện ngay bên dưới)")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
        )
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
            )
        except (ImportError, ValueError) as e:
            # flash_attention_2 có thể không cài được / không hỗ trợ trên GPU này (vd. Colab T4).
            # Không để cả pipeline chết vì việc này -> rơi về "sdpa" (built-in PyTorch, luôn có sẵn).
            print(f"[vision] attn_implementation='{self.attn_implementation}' không dùng được ({e}), "
                  f"fallback về 'sdpa'.")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
        if self.device in ("cpu", "mps"):
            self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        """Giải phóng model khỏi VRAM/RAM sau khi phân tích xong toàn bộ scene."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        """Phân tích 1 scene (dùng khi batch_size=1 hoặc làm fallback)."""
        results = self.analyze_batch([(scene_id, frame_paths)])
        return results[0]

    def _build_messages(self, frame_paths: list[str]) -> list[dict]:
        return [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": p} for p in frame_paths
            ] + [{
                "type": "text",
                "text": "Analyze this scene and return the JSON object described in the system prompt.",
            }]},
        ]

    def analyze_batch(self, scenes: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        """
        Phân tích nhiều scene trong 1 lần forward/generate() thay vì tuần tự từng scene.
        Input/output cho MỖI scene giữ nguyên (cùng ảnh, cùng prompt, cùng max_new_tokens)
        -> không đổi chất lượng, chỉ đổi cách GPU xử lý (song song theo batch dimension
        thay vì lần lượt), nên tận dụng GPU tốt hơn nhiều so với gọi generate() 1-scene-1-lần.
        """
        if self.model is None:
            raise RuntimeError("LocalVisionAnalyzer chưa được load(). Gọi .load() trước.")

        valid: list[tuple[str, list[str]]] = []
        empties: list[dict[str, Any]] = []
        for scene_id, frame_paths in scenes:
            existing = [p for p in frame_paths if Path(p).exists()]
            if existing:
                valid.append((scene_id, existing))
            else:
                empties.append(_empty_result(scene_id, reason="no_frames"))

        if not valid:
            return empties

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        messages_batch = [self._build_messages(paths) for _, paths in valid]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        if process_vision_info is not None:
            # Cách chuẩn của họ Qwen-VL để gom ảnh nhiều mẫu vào 1 batch đúng thứ tự.
            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = self.processor(
                text=texts, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        else:
            # Fallback nếu thiếu qwen-vl-utils: tự mở ảnh bằng PIL (vẫn đúng, chỉ kém tối ưu hơn 1 chút).
            images_flat = [Image.open(p).convert("RGB") for _, paths in valid for p in paths]
            inputs = self.processor(
                text=texts, images=images_flat, padding=True, return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results = []
        for (scene_id, _), text in zip(valid, output_texts):
            parsed = _parse_json_response(text)
            parsed["scene_id"] = scene_id
            results.append(parsed)

        # Giữ nguyên thứ tự scene ban đầu (kể cả scene rỗng bị tách ra ở trên).
        by_id = {r["scene_id"]: r for r in results + empties}
        return [by_id[scene_id] for scene_id, _ in scenes]


# =============================================================================
# Backend "cerebras" — Gemma 4 31B multimodal qua Cerebras API
# =============================================================================

class CerebrasVisionAnalyzer:
    """Gọi Gemma 4 31B (multimodal, public preview) trên Cerebras Cloud để phân tích scene.

    Tự giới hạn số ảnh/request và tốc độ gọi (RPM) theo tier hiện tại (config.toml),
    để tránh dồn 429 khi chạy nhiều scene liên tiếp. Xem rate-limit chính thức tại
    https://inference-docs.cerebras.ai/support/rate-limits (free trial: 5 RPM/30K TPM/
    1M TPD, tối đa 2 ảnh/request; Developer: 300 RPM/500K TPM, tối đa 5 ảnh/request).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = cfg.get("api.cerebras_api_key", "")
        self.base_url = cfg.get("api.cerebras_base_url", "https://api.cerebras.ai/v1")
        self.model = cfg.get("processing.cerebras_vision_model", "gemma-4-31b")
        self.max_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.max_images = cfg.get("processing.cerebras_vision_max_images", 2)
        self.rpm = max(1, cfg.get("processing.cerebras_vision_rpm", 5))
        self._min_interval = 60.0 / self.rpm
        self._last_call_ts = 0.0
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        if not self.api_key or self.api_key.startswith("PASTE_"):
            raise ValueError(
                "Chưa cấu hình api.cerebras_api_key trong config.toml "
                "(cần cho vision_backend = \"cerebras\")."
            )
        print(f"[vision] Dùng Cerebras API, model '{self.model}' "
              f"(backend=cerebras, {self.rpm} RPM, tối đa {self.max_images} ảnh/request).")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def unload(self) -> None:
        self.client = None

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
        mime = "jpeg" if ext == "jpg" else ext
        return f"data:image/{mime};base64,{b64}"

    def _throttle(self) -> None:
        """Chờ đủ để không vượt RPM đã cấu hình (token-bucket đơn giản, không cần lib ngoài)."""
        elapsed = time.monotonic() - self._last_call_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        existing = [p for p in frame_paths if Path(p).exists()][: self.max_images]
        if not existing:
            return _empty_result(scene_id, reason="no_frames")

        content = [
            {"type": "image_url", "image_url": {"url": self._encode_image(p)}}
            for p in existing
        ]
        content.append({
            "type": "text",
            "text": "Analyze this scene and return the JSON object described in the system prompt.",
        })

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                )
                output_text = response.choices[0].message.content or ""
                break
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if is_rate_limit and attempt < max_retries:
                    backoff = 60.0 * attempt  # 60s, 120s... tôn trọng cửa sổ TPM/RPM
                    print(f"[vision] {scene_id}: bị rate limit (lần {attempt}), "
                          f"chờ {backoff:.0f}s rồi thử lại...")
                    time.sleep(backoff)
                    continue
                print(f"[vision] Lỗi gọi Cerebras cho {scene_id}: {e}")
                return _empty_result(scene_id, reason="api_error")

        parsed = _parse_json_response(output_text)
        parsed["scene_id"] = scene_id
        return parsed


# =============================================================================
# Backend "mistral" — Mistral Medium 3.5 (multimodal) qua API chính thức Mistral
# =============================================================================
#
# Dùng riêng cho stage vision (đọc ảnh scene) — KHÔNG liên quan đến engine
# Cerebras/GLM (zai-glm-4.7) mà script_writer.py đang dùng để viết kịch bản,
# 2 việc này độc lập hoàn toàn với nhau trong config.toml.
#
# API Mistral tương thích OpenAI (base_url https://api.mistral.ai/v1, dùng
# chung thư viện `openai` đã có trong requirements.txt), nhưng field ảnh khác
# Cerebras/OpenAI một chút: Mistral nhận "image_url" là 1 CHUỖI (URL hoặc
# data-URI base64) trực tiếp, không bọc thêm {"url": ...} như OpenAI.
class MistralVisionAnalyzer:
    """Gọi model multimodal của Mistral (mặc định Mistral Medium 3.5 — model
    lớn nhất, chất lượng đọc ảnh tốt nhất trong dòng Mistral hiện tại) để
    phân tích scene. Free tier của Mistral rộng rãi hơn Cerebras nhiều, nhưng
    vẫn giữ throttle + retry-on-429 cho an toàn (RPM/TPM thực tế có thể đổi
    theo tài khoản, không cứng ngưỡng cụ thể)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = cfg.get("api.mistral_api_key", "")
        self.base_url = cfg.get("api.mistral_base_url", "https://api.mistral.ai/v1")
        self.model = cfg.get("processing.mistral_vision_model", "mistral-large-latest")
        self.max_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.max_images = cfg.get("processing.mistral_vision_max_images", 8)
        self.rpm = max(1, cfg.get("processing.mistral_vision_rpm", 15))
        self._min_interval = 60.0 / self.rpm
        self._last_call_ts = 0.0
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        if not self.api_key or self.api_key.startswith("PASTE_"):
            raise ValueError(
                "Chưa cấu hình api.mistral_api_key trong config.toml "
                "(cần cho vision_backend = \"mistral\")."
            )
        print(f"[vision] Dùng Mistral API, model '{self.model}' "
              f"(backend=mistral, {self.rpm} RPM, tối đa {self.max_images} ảnh/request).")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def unload(self) -> None:
        self.client = None

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
        mime = "jpeg" if ext == "jpg" else ext
        return f"data:image/{mime};base64,{b64}"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        existing = [p for p in frame_paths if Path(p).exists()][: self.max_images]
        if not existing:
            return _empty_result(scene_id, reason="no_frames")

        # Mistral: "image_url" là chuỗi data-URI trực tiếp (khác OpenAI/Cerebras
        # vốn bọc trong {"url": ...}) — xem docs.mistral.ai/studio-api/conversations/vision
        content = [
            {"type": "image_url", "image_url": self._encode_image(p)}
            for p in existing
        ]
        content.append({
            "type": "text",
            "text": "Analyze this scene and return the JSON object described in the system prompt.",
        })

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                )
                output_text = response.choices[0].message.content or ""
                break
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if is_rate_limit and attempt < max_retries:
                    backoff = 60.0 * attempt
                    print(f"[vision] {scene_id}: bị rate limit (lần {attempt}), "
                          f"chờ {backoff:.0f}s rồi thử lại...")
                    time.sleep(backoff)
                    continue
                print(f"[vision] Lỗi gọi Mistral cho {scene_id}: {e}")
                return _empty_result(scene_id, reason="api_error")

        parsed = _parse_json_response(output_text)
        parsed["scene_id"] = scene_id
        return parsed


# =============================================================================
# =============================================================================
# Backend "qwen2vl" — Qwen2-VL-2B-Instruct (~2B), nhẹ + ổn định + mọi phần cứng
# =============================================================================
#
# Thay thế moondream (bị lỗi trust_remote_code với transformers>=5.0).
# Qwen2-VL-2B chỉ ~2B tham số (nhẹ hơn Qwen3-VL-4B 2x, ngang moondream2),
# chạy ổn định trên transformers chính thức (không cần trust_remote_code),
# hỗ trợ multi-image batch như Qwen3-VL, và chạy được trên CPU/CUDA/MPS.
# Đây là lựa chọn mặc định cho máy không GPU hoặc GPU yếu.

class Qwen2VLVisionAnalyzer:
    """Bọc Qwen2-VL-2B-Instruct, load một lần và tái sử dụng."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.qwen2vl_model_name", "Qwen/Qwen2-VL-2B-Instruct")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = resolve_torch_device(cfg.get("processing.vision_device", "auto"))
        self.max_new_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.batch_size = max(1, cfg.get("processing.vision_batch_size", 4))
        self.model = None
        self.processor = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self._torch = torch

        print(f"[vision] Loading {self.model_name} on {self.device} "
              f"(backend=qwen2vl, batch_size={self.batch_size})... "
              f"(~2B params, nhẹ hơn Qwen3-VL-4B, không cần trust_remote_code)")

        kwargs = {"cache_dir": self.cache_dir}

        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        elif self.device == "mps":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, **kwargs)

        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def _build_messages(self, frame_paths: list[str]) -> list[dict]:
        return [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": p} for p in frame_paths
            ] + [{
                "type": "text",
                "text": "Analyze this scene and return the JSON object described in the system prompt.",
            }]},
        ]

    def analyze_batch(self, scenes: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Qwen2VLVisionAnalyzer chưa được load(). Gọi .load() trước.")

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        results: list[dict[str, Any]] = []
        for scene_id, frame_paths in scenes:
            existing = [p for p in frame_paths if Path(p).exists()]
            if not existing:
                results.append(_empty_result(scene_id, reason="no_frames"))
                continue

            messages = self._build_messages(existing)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if process_vision_info is not None:
                image_inputs, _ = process_vision_info(messages)
            else:
                image_inputs = [{"type": "image", "image": p} for p in existing]

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.model.device)

            with self._torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
            )[0]

            parsed = _parse_json_response(output_text)
            parsed["scene_id"] = scene_id
            results.append(parsed)

        return results

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        results = self.analyze_batch([(scene_id, frame_paths)])
        return results[0]


# Backend "moondream" — Moondream2 (~1.9B), tối ưu cho CPU/edge
# =============================================================================
#
# Khác 2 backend API ở trên (không tốn CPU vì compute nằm ở server), backend
# này VẪN chạy local như "local" (Qwen3-VL-4B) — nhưng Moondream2 nhỏ hơn
# ~2x và được huấn luyện/tối ưu riêng để chạy tốt trên CPU/Raspberry Pi, nên
# nhanh hơn nhiều lần so với Qwen3-VL-4B trên cùng máy không GPU (đúng cảnh
# GitHub Actions runner). Không dùng multi-image batch như LocalVisionAnalyzer
# vì API gốc của Moondream2 (encode_image + answer_question) chỉ nhận 1 ảnh/
# câu hỏi — nên implement analyze_scene() đơn lẻ (không có analyze_batch),
# và chỉ đọc 1 keyframe đại diện/scene (frame ở giữa danh sách keyframes).

def _patch_moondream_class(model_name: str, cache_dir: str | None = None, revision: str | None = None) -> None:
    """
    Monkey-patch HfMoondream (loaded via trust_remote_code) để tương thích
    transformers 5.x. transformers 5 yêu cầu:
      1. `all_tied_weights_keys` property trong _finalize_model_loading
      2. `_tied_weights_keys` phải tồn tại sau __init__
    Code remote của moondream2 (rev 2025-06-21) chưa có cả hai.

    Cách tiếp cận: dùng get_class_from_dynamic_module (API chính thức của
    transformers) để lấy class TRƯỚC khi from_pretrained tạo instance,
    rồi patch trực tiếp trên class. Đây là cách chắc chắn nhất vì class
    được patch trước khi bất kỳ instance nào được tạo.
    """
    import sys
    import importlib

    cls = None
    cfg_kwargs = {"trust_remote_code": True}
    if cache_dir:
        cfg_kwargs["cache_dir"] = cache_dir
    if revision:
        cfg_kwargs["revision"] = revision

    # --- Cách 1: Dùng get_class_from_dynamic_module (API chính thức HF) ---
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, **cfg_kwargs)
        class_ref = None
        if hasattr(config, "auto_map") and config.auto_map:
            class_ref = config.auto_map.get("AutoModelForCausalLM")

        if class_ref:
            cls = get_class_from_dynamic_module(class_ref, model_name, **cfg_kwargs)
    except Exception:
        pass

    # --- Cách 2: Tìm trong sys.modules (đã load từ lần chạy trước) ---
    if cls is None:
        for mod_name, mod in list(sys.modules.items()):
            if hasattr(mod, "HfMoondream"):
                cls = mod.HfMoondream
                break

    # --- Cách 3: Thử import trực tiếp từ transformers_modules ---
    if cls is None:
        parts = model_name.replace("/", ".").split(".")
        for suffix in ["hf_moondream", "moondream"]:
            try:
                mod_path = f"transformers_modules.{'.'.join(parts)}.{suffix}"
                mod = importlib.import_module(mod_path)
                if hasattr(mod, "HfMoondream"):
                    cls = mod.HfMoondream
                    break
            except Exception:
                pass

    if cls is None:
        print("[vision] Warning: Không tìm thấy class HfMoondream để patch. "
              "Nếu transformers>=5.0, load model có thể crash với lỗi all_tied_weights_keys.")
        return

    # --- Patch 1: all_tied_weights_keys property ---
    if not hasattr(cls, "all_tied_weights_keys"):
        @property
        def all_tied_weights_keys(self):
            return getattr(self, "_tied_weights_keys", {})
        cls.all_tied_weights_keys = all_tied_weights_keys

    # --- Patch 2: Đảm bảo _tied_weights_keys tồn tại sau __init__ ---
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "_tied_weights_keys"):
            self._tied_weights_keys = {}

    cls.__init__ = patched_init
    print(f"[vision] Patched {cls.__name__} for transformers 5.x compatibility.")



class MoondreamVisionAnalyzer:
    """Bọc model Moondream2, load một lần và tái sử dụng cho toàn bộ scene."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.moondream_model_name", "vikhyatk/moondream2")
        self.revision = cfg.get("processing.moondream_revision", "2025-06-21")
        self.cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))
        self.device = resolve_torch_device(cfg.get("processing.vision_device", "auto"))
        self.max_new_tokens = cfg.get("processing.vision_max_new_tokens", 512)
        self.model = None
        self.tokenizer = None
        self._torch = None

    def load(self) -> None:
        import torch  # lazy import: chỉ cần khi thực sự dùng backend "moondream"
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        self._torch = torch

        print(f"[vision] Loading {self.model_name} (rev={self.revision or 'main'}) "
              f"on {self.device} (backend=moondream, 1 keyframe/scene)... "
              f"(lần đầu sẽ tải model; nhẹ hơn Qwen3-VL-4B nhiều nên tải nhanh hơn)")

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }
        if self.revision:
            kwargs["revision"] = self.revision

        # --- Patch transformers 5.x compatibility ---
        # BƯỚC 1: Load config để HF tải dynamic module (hf_moondream.py) vào sys.modules
        # BƯỚC 2: Patch class TRƯỚC KHI tạo instance
        # BƯỚC 3: Load model với class đã được patch
        try:
            _ = AutoConfig.from_pretrained(self.model_name, **kwargs)
        except Exception as e:
            print(f"[vision] Warning: Could not preload config for patching: {e}")

        _patch_moondream_class(self.model_name, self.cache_dir, self.revision)

        # --- Tự động chọn dtype & device_map theo phần cứng ---
        if self.device == "cuda":
            # CUDA: dùng float16 (hoặc bfloat16 nếu GPU hỗ trợ) + device_map để tự động phân bổ
            if torch.cuda.is_bf16_supported():
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        elif self.device == "mps":
            # Apple Silicon: bfloat16 ổn định hơn float16 trên MPS
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            # CPU: float32 để tránh lỗi precision một số op không hỗ trợ int8/fp16 trên CPU
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

        # Nếu không dùng device_map (cpu/mps), đảm bảo model nằm đúng device
        if "device_map" not in kwargs:
            self.model.to(self.device)

        self.model.eval()

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _pick_frame(frame_paths: list[str]) -> str | None:
        """Chọn 1 frame đại diện cho scene: frame ở giữa danh sách (thường
        tránh được transition/fade dính ở frame đầu/cuối scene) thay vì
        luôn lấy frame đầu tiên."""
        existing = [p for p in frame_paths if Path(p).exists()]
        if not existing:
            return None
        return existing[len(existing) // 2]

    def analyze_scene(self, scene_id: str, frame_paths: list[str]) -> dict[str, Any]:
        frame_path = self._pick_frame(frame_paths)
        if frame_path is None:
            return _empty_result(scene_id, reason="no_frames")

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"[vision] Lỗi mở ảnh {frame_path} cho {scene_id}: {e}")
            return _empty_result(scene_id, reason="image_open_error")

        # Moondream2 không có slot "system prompt" riêng như chat model
        # thông thường -> gộp system prompt vào luôn câu hỏi.
        prompt = (
            VISION_SYSTEM_PROMPT
            + " This is a single representative frame from the scene "
              "(not all frames) — analyze it and return the JSON object "
              "described above."
        )

        try:
            enc_image = self.model.encode_image(image)
            output_text = self.model.answer_question(
                enc_image, prompt, self.tokenizer, max_new_tokens=self.max_new_tokens,
            )
        except TypeError:
            # Vài revision cũ của moondream2 không nhận max_new_tokens qua
            # answer_question() -> gọi lại không kèm tham số này.
            try:
                output_text = self.model.answer_question(enc_image, prompt, self.tokenizer)
            except Exception as e:
                print(f"[vision] Lỗi Moondream2 cho {scene_id}: {e}")
                return _empty_result(scene_id, reason="model_error")
        except Exception as e:
            print(f"[vision] Lỗi Moondream2 cho {scene_id}: {e}")
            return _empty_result(scene_id, reason="model_error")

        parsed = _parse_json_response(output_text)
        parsed["scene_id"] = scene_id
        return parsed


def _build_analyzer(cfg):
    backend = cfg.get("processing.vision_backend", "local")
    if backend == "cerebras":
        return CerebrasVisionAnalyzer(cfg)
    if backend == "mistral":
        return MistralVisionAnalyzer(cfg)
    if backend == "qwen2vl":
        return Qwen2VLVisionAnalyzer(cfg)
    if backend == "moondream":
        return MoondreamVisionAnalyzer(cfg)
    return LocalVisionAnalyzer(cfg)


def run_vision_analysis(cfg, preprocess_result: dict[str, Any], checkpoint_mgr=None) -> list[dict[str, Any]]:
    """Entry point cho stage 'vision'. Ghi vision_analysis.json vào pipeline/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    scenes = preprocess_result["scenes"]
    keyframes = preprocess_result["keyframes"]

    total_scenes = len(scenes)

    # --- Resume: tải các scene đã có micro-checkpoint từ lần chạy trước ---
    # Mỗi scene được checkpoint RIÊNG LẺ (key = scene_id), nên ta có thể biết
    # chính xác scene nào đã xong mà không cần chạy lại từ đầu.
    results_by_scene: dict[str, dict[str, Any]] = {}
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("vision")
        for scene in scenes:
            sid = scene["scene_id"]
            if sid in done_ids:
                try:
                    results_by_scene[sid] = checkpoint_mgr.load_micro("vision", sid)
                except (OSError, json.JSONDecodeError, KeyError):
                    pass  # file lỗi/hỏng -> coi như chưa xong, phân tích lại

        if results_by_scene:
            print(f"[vision] Tìm thấy checkpoint: {len(results_by_scene)}/{total_scenes} "
                  f"scene đã phân tích trước đó, sẽ bỏ qua và chỉ chạy tiếp phần còn lại.")

    remaining_scenes = [s for s in scenes if s["scene_id"] not in results_by_scene]

    analyzer = _build_analyzer(cfg)
    batch_size = getattr(analyzer, "batch_size", 1)

    if remaining_scenes:
        analyzer.load()
        try:
            if hasattr(analyzer, "analyze_batch"):
                done = total_scenes - len(remaining_scenes)
                for i in range(0, len(remaining_scenes), batch_size):
                    chunk = remaining_scenes[i:i + batch_size]
                    batch_input = [(s["scene_id"], keyframes.get(s["scene_id"], [])) for s in chunk]
                    batch_results = analyzer.analyze_batch(batch_input)
                    for scene, analysis in zip(chunk, batch_results):
                        analysis["start"] = scene["start"]
                        analysis["end"] = scene["end"]
                        results_by_scene[scene["scene_id"]] = analysis
                    done += len(chunk)
                    print_progress_bar(
                        done, total_scenes,
                        prefix="[vision] analyzing",
                        suffix=f"batch {i // batch_size + 1} ({len(chunk)} scene)",
                    )
                    # Micro-checkpoint TỪNG SCENE ngay sau khi batch xong (không
                    # phải copy cả list kết quả) -> mỗi file chỉ chứa đúng 1
                    # scene, resume được và không tốn ổ đĩa/API tăng dần theo
                    # cấp số. LUÔN lưu ngay sau mỗi batch (không throttle theo
                    # micro_interval ở đây) vì việc ghi JSON cục bộ rất rẻ, còn
                    # nếu throttle theo done % micro_interval thì khi
                    # vision_batch_size không chia hết cho micro_interval, cả
                    # 1 batch đã phân tích xong có thể bị bỏ hẳn không checkpoint
                    # -> mất nhiều tiến độ hơn dự kiến nếu crash giữa chừng.
                    # Việc throttle tần suất SYNC LÊN CLOUD (tốn API hơn nhiều)
                    # đã được xử lý riêng trong CheckpointManager.save_micro().
                    if checkpoint_mgr is not None:
                        for scene, analysis in zip(chunk, batch_results):
                            checkpoint_mgr.save_micro("vision", scene["scene_id"], analysis)
            else:
                total_remaining = len(remaining_scenes)
                for scene_idx, scene in enumerate(remaining_scenes, start=1):
                    scene_id = scene["scene_id"]
                    frame_paths = keyframes.get(scene_id, [])
                    analysis = analyzer.analyze_scene(scene_id, frame_paths)
                    analysis["start"] = scene["start"]
                    analysis["end"] = scene["end"]
                    results_by_scene[scene_id] = analysis
                    done_total = total_scenes - total_remaining + scene_idx
                    print_progress_bar(
                        done_total, total_scenes,
                        prefix="[vision] analyzing",
                        suffix=f"{scene_id} ({len(frame_paths)} frames)",
                    )
                    # LUÔN lưu checkpoint ngay sau mỗi scene (không throttle theo
                    # micro_interval) — giống nhánh batch ở trên: mỗi scene ở đây
                    # đã tốn tiền/quota gọi API backend (Cerebras/Mistral), nên nếu
                    # throttle và pipeline bị ngắt giữa chừng, các scene đã gọi API
                    # xong nhưng chưa tới mốc lưu sẽ mất trắng và phải tốn API gọi
                    # lại. Việc throttle tần suất SYNC LÊN CLOUD đã được xử lý
                    # riêng trong CheckpointManager.save_micro().
                    if checkpoint_mgr is not None:
                        checkpoint_mgr.save_micro("vision", scene_id, analysis)
        finally:
            analyzer.unload()
            # Lưới an toàn cuối stage: đảm bảo MỌI micro-checkpoint vision đã
            # tạo ra trong lần chạy này thực sự nằm trên cloud, không chỉ
            # nằm trên đĩa tạm của Colab (xem docstring flush_pending_syncs).
            if checkpoint_mgr is not None:
                checkpoint_mgr.flush_pending_syncs()
    else:
        print("[vision] Tất cả scene đã có checkpoint, bỏ qua bước phân tích.")

    # Ghép kết quả theo đúng thứ tự scene gốc.
    results: list[dict[str, Any]] = [results_by_scene[s["scene_id"]] for s in scenes]

    with open(pipeline_dir / "vision_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[vision] Xong: {len(results)} scene đã phân tích.")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("vision", results)

    return results
