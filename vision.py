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
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from platform_utils import resolve_torch_device
from platform_utils import PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE as _EARLY_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE
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


_MODELSCOPE_HIJACK_SUBMODULES = [
    "transformers.dynamic_module_utils",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.auto_factory",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.image_processing_auto",
    "transformers.models.auto.feature_extraction_auto",
    "transformers.models.auto.processing_auto",
]

# Bản "sạch" (chưa bị modelscope monkeypatch) của get_class_from_dynamic_module.
#
# ƯU TIÊN dùng bản đã chụp SẴN từ `platform_utils` (chụp ngay lúc
# `from platform_utils import ...` chạy ở GẦN DÒNG ĐẦU TIÊN của run.py —
# TRƯỚC CẢ khi `ensure_python_packages()` gọi `__import__("funasr")`, chính
# là nơi funasr kéo theo modelscope và modelscope monkeypatch transformers
# ngay lúc import). Đây là bản THẬT SỰ đáng tin cậy.
#
# Bản chụp NGAY TẠI ĐÂY (lúc `import vision`) chỉ dùng làm dự phòng: `import
# vision` chỉ xảy ra rất muộn trong run_pipeline_on_project() — SAU KHI
# ensure_python_packages() đã chạy xong từ lâu — nên nếu tới đây transformers
# đã bị vá rồi, bản chụp này chụp nhầm luôn bản đã hỏng (đúng lý do bản vá
# trước không có tác dụng trên Colab, dù ASR stage có bị bỏ qua vì checkpoint
# hay không — vì __import__("funasr") vẫn luôn chạy ở bước kiểm tra dependency).
try:
    import transformers.dynamic_module_utils as _dmu_pristine
    _LATE_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE = _dmu_pristine.get_class_from_dynamic_module
except Exception:
    _LATE_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE = None

_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE = (
    _EARLY_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE or _LATE_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE
)


def _rebind_get_class_from_dynamic_module(original_fn) -> None:
    """Rebind `get_class_from_dynamic_module` về đúng `original_fn` ở mọi nơi
    modelscope có thể đã ghi đè (kể cả `transformers.dynamic_module_utils`
    gốc, không chỉ các submodule auto.*)."""
    import importlib
    for mod_name in _MODELSCOPE_HIJACK_SUBMODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if getattr(mod, "get_class_from_dynamic_module", None) is not original_fn:
            mod.get_class_from_dynamic_module = original_fn


def _patch_modelscope_dynamic_module_hijack() -> None:
    """Sửa lỗi: funasr (ASR stage, chạy TRƯỚC vision stage) import modelscope,
    và modelscope tự động monkeypatch `get_class_from_dynamic_module` bên trong
    nhiều submodule `transformers.models.auto.*` (và đôi khi cả bản gốc trong
    `transformers.dynamic_module_utils`) để mọi model tải qua
    `trust_remote_code=True` (kể cả model KHÔNG liên quan gì tới modelscope,
    như Moondream2/Qwen3-VL ở đây) bị "hijack" sang tải qua ModelScope Hub
    thay vì Hugging Face Hub.

    Với modelscope 1.38.1 + transformers 4.57.6, bản patch đó có bug: nó gọi
    `args[0] = snapshot_download(...)` trong khi `args` lúc này là 1 tuple
    (immutable) -> crash ngay khi bắt đầu load model:
        TypeError: 'tuple' object does not support item assignment
    (modelscope/utils/hf_util/patcher.py, hàm get_class_from_dynamic_module)

    Cách sửa: rebind lại `get_class_from_dynamic_module` (ở TỪNG submodule
    auto.* lẫn ở chính `transformers.dynamic_module_utils`) về bản ĐÃ CHỤP
    SẴN từ lúc `import vision` (biến `_PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE`
    ở trên) — không đọc lại "bản gốc" tại thời điểm này nữa, vì tại đây có
    thể đã quá trễ (modelscope có thể đã patch xong). Kèm 1 lưới an toàn phụ
    vá thẳng vào modelscope nếu module đó đã được import. Không cần biết chi
    tiết modelscope patch thế nào / không cần gỡ cài đặt modelscope (vẫn cần
    cho FunASR ở ASR stage).
    """
    original_fn = _PRISTINE_GET_CLASS_FROM_DYNAMIC_MODULE
    if original_fn is None:
        try:
            import transformers.dynamic_module_utils as _dmu
            original_fn = _dmu.get_class_from_dynamic_module
        except Exception:
            return  # transformers chưa cài / đổi cấu trúc module -> bỏ qua, không crash pipeline

    _rebind_get_class_from_dynamic_module(original_fn)

    # Lưới an toàn phụ: nếu modelscope đã được import, vá LUÔN thẳng vào
    # module gốc giữ hàm lỗi (modelscope.utils.hf_util.patcher) — phòng
    # trường hợp có nơi nào đó gọi thẳng vào tham chiếu của module này thay
    # vì qua tên đã rebind ở transformers.*.
    try:
        import modelscope.utils.hf_util.patcher as _ms_patcher
        if getattr(_ms_patcher, "get_class_from_dynamic_module", None) is not original_fn:
            _ms_patcher.get_class_from_dynamic_module = original_fn
    except Exception:
        pass


_INTENSITY_WORD_MAP = {
    "none": 0.0,
    "very low": 0.1,
    "low": 0.2,
    "mild": 0.2,
    "moderate": 0.5,
    "medium": 0.5,
    "average": 0.5,
    "high": 0.8,
    "very high": 0.9,
    "intense": 0.9,
    "extreme": 1.0,
    "maximum": 1.0,
}


def _safe_visual_intensity(raw: Any) -> float:
    """Chuyển giá trị visual_intensity trả về từ model sang float một cách an toàn.

    Model vision (đặc biệt các backend nhẹ như moondream) đôi khi không tuân
    theo schema và trả về một câu mô tả (vd: "Moderate, capturing the hand's
    interaction with the note.") thay vì một con số. Trước đây code ép
    float() trực tiếp lên giá trị này khiến cả pipeline crash giữa chừng.
    Hàm này cố gắng khôi phục một con số hợp lý, và nếu không thể thì trả
    về 0.0 thay vì raise lỗi.
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0.0
        # Thử parse trực tiếp trước (trường hợp là "0.7" hoặc "0,7")
        try:
            return max(0.0, min(1.0, float(text.replace(",", "."))))
        except ValueError:
            pass
        # Thử tìm số thực đầu tiên trong chuỗi (vd: "khoảng 0.6 do...")
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                val = float(match.group())
                # Nếu model trả thang điểm 0-10 thay vì 0-1, chuẩn hoá lại
                if val > 1.0:
                    val = val / 10.0 if val <= 10.0 else 1.0
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        # Thử map các từ mô tả cường độ (không phân biệt hoa/thường)
        lowered = text.lower()
        for word, val in _INTENSITY_WORD_MAP.items():
            if word in lowered:
                return val
        # Không nhận diện được — fallback an toàn, không crash pipeline
        return 0.0
    return 0.0


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
        "visual_intensity": _safe_visual_intensity(data.get("visual_intensity", 0.0)),
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
        _patch_modelscope_dynamic_module_hijack()
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
# QUAN TRỌNG — transformers v5.x đổi cách PreTrainedModel khởi tạo nội bộ:
# nó cần self.all_tied_weights_keys được set trong post_init(). Code custom
# (trust_remote_code=True) của Moondream2 (class HfMoondream) chưa cập nhật
# theo API mới này, nên AutoModelForCausalLM.from_pretrained(...) sẽ vỡ với:
#   AttributeError: 'HfMoondream' object has no attribute
#   'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?
# Cách sửa CHÍNH là ghim transformers<5.0.0 (xem requirements.txt) — NHƯNG
# ensure_python_packages() trong run.py chỉ cài package còn THIẾU, không tự
# hạ version 1 package ĐÃ importable sẵn (vd Colab đã cache transformers>=5.0
# từ trước) -> pin trong requirements.txt sẽ KHÔNG có tác dụng trừ khi chạy
# tay "pip install -r requirements.txt --upgrade". Patch dưới đây là lưới an
# toàn PHỤ, hoạt động bất kể version transformers đang cài là gì: chặn đúng
# 1 attribute bị thiếu, không đổi hành vi nào khác của model/torch.
def _patch_transformers_v5_tied_weights_compat() -> None:
    import torch
    if getattr(torch.nn.Module, "_moondream_v5_patch_applied", False):
        return
    _orig_getattr = torch.nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _orig_getattr(self, name)

    torch.nn.Module.__getattr__ = _patched_getattr
    torch.nn.Module._moondream_v5_patch_applied = True


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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        _patch_modelscope_dynamic_module_hijack()
        _patch_transformers_v5_tied_weights_compat()

        print(f"[vision] Loading {self.model_name} (rev={self.revision or 'main'}) "
              f"on {self.device} (backend=moondream, 1 keyframe/scene)... "
              f"(lần đầu sẽ tải model; nhẹ hơn Qwen3-VL-4B nhiều nên tải nhanh hơn)")

        kwargs: dict[str, Any] = {"trust_remote_code": True, "cache_dir": self.cache_dir}
        if self.revision:
            kwargs["revision"] = self.revision

        def _load_with_modelscope_retry(loader, *args, **kw):
            """Gọi `loader(*args, **kw)`; nếu dính đúng lỗi hijack của
            modelscope (TypeError trên tuple), vá lại 1 lần nữa (bao gồm cả
            lưới an toàn vá thẳng modelscope) rồi thử lại ĐÚNG 1 lần — tránh
            crash toàn bộ pipeline vì 1 lỗi tương thích thư viện bên ngoài,
            không phải lỗi logic của chính vision.py."""
            try:
                return loader(*args, **kw)
            except TypeError as e:
                msg = str(e)
                if "tuple" not in msg or "item assignment" not in msg:
                    raise
                print("[vision] Phát hiện ModelScope vẫn hijack from_pretrained "
                      "(TypeError tuple) dù đã vá — vá sâu thêm lần nữa rồi thử lại...")
                _patch_modelscope_dynamic_module_hijack()
                return loader(*args, **kw)

        self.model = _load_with_modelscope_retry(AutoModelForCausalLM.from_pretrained, self.model_name, **kwargs)
        self.tokenizer = _load_with_modelscope_retry(AutoTokenizer.from_pretrained, self.model_name, **kwargs)
        if self.device in ("cpu", "mps"):
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

        try:
            parsed = _parse_json_response(output_text)
        except Exception as e:
            # An toàn: một scene lỗi không được phép làm sập cả pipeline khi
            # đã chạy hàng chục/hàng trăm scene trước đó tốn nhiều thời gian.
            print(f"[vision] Lỗi parse JSON cho {scene_id}: {e}")
            return _empty_result(scene_id, reason="parse_error")
        parsed["scene_id"] = scene_id
        return parsed


# =============================================================================
# Backend "gemma4_video" — Gemma 4 12B (QAT CHÍNH THỨC từ Google), chạy qua
# Ollama (OpenAI-compatible API) thay vì transformers.
#
# THAY ĐỔI LỚN so với bản safetensors cũ (google/gemma-4-12B-it-qat-q4_0-
# unquantized, ~24GB): bản cũ load bằng
# AutoModelForMultimodalLM.from_pretrained(...) khiến accelerate bật disk
# offload rồi mmap cả file 24GB -> vượt RAM runner GitHub Actions (~16GB),
# chết với "unable to mmap ... Cannot allocate memory (12)" ngay ở stage
# vision. Cách chạy MỚI là qua Ollama `gemma4:12b-it-qat` (GGUF Q4_0 ~7GB,
# cùng nhánh QAT chính thức, llama.cpp tối ưu CPU) — đúng cơ chế đã được
# kiểm chứng chạy THÀNH CÔNG trên GitHub Actions ở dự án fork OpenManus
# (handandfeet-agent): nhẹ, không cần GPU, không mmap file khổng lồ.
#
# Cơ chế THINKING theo mức độ (kế thừa nguyên xi từ handandfeet-agent):
#   processing.gemma4_reasoning_effort = "high" | "medium" | "low" | "none"
#   - "high"  (MẶC ĐỊNH): suy nghĩ sâu nhất, chất lượng cao nhất, chậm nhất.
#   - "none"  : tắt hẳn thinking (nhanh nhưng kém chất lượng).
# Giá trị được gửi qua body field `reasoning_effort` của
# POST {base}/chat/completions (Ollama OpenAI-compatible) — y hệt cách
# handandfeet-agent truyền qua extra_body của openai python client.
#
# Vì Ollama nhận ẢNH (base64) chứ không nhận video như transformers, mỗi
# clip .mp4 được trích `gemma4_frames_per_clip` frame cách đều theo thời
# gian, gửi kèm phụ đề cắt theo đúng cửa sổ [start, end] — giữ được tinh
# thần "xem clip + đối chiếu thoại" của backend này mà không cần GPU.
# =============================================================================

GEMMA4_VIDEO_SYSTEM_PROMPT = (
    "You are a visual analyst for a video commentary pipeline. You are given "
    "a short video clip (with audio) from one scene, together with the "
    "subtitle lines spoken during this exact clip. Watch the motion and "
    "listen to/read the dialogue, cross-reference what you see with what is "
    "said, and describe concrete, visible/audible facts first, then a short "
    "interpretation. Do not invent content that is not shown or said in "
    "this clip. Respond ONLY with a single JSON object with these exact "
    "keys: visual_summary (string), characters (array of strings), "
    "location (string), actions (array of strings), emotion (string), "
    "shot_type (string), visual_intensity (number 0-1), tags (array of "
    "strings), draft_narration (string — a short draft commentary line for "
    "this clip, based only on what you actually saw/heard here; this is a "
    "rough draft, not the final script). No markdown, no extra text, only "
    "the JSON object."
)


def _slice_subtitles_for_window(asr_timeline: list[dict[str, Any]], start: float, end: float) -> str:
    """Lọc các dòng phụ đề (từ asr.py, mốc thời gian TUYỆT ĐỐI so với đầu
    phim, đơn vị giây — xem asr.py: mỗi entry có "start"/"end"/"text") có
    khoảng thời gian giao với [start, end], trả về text đã ghép, kèm mốc
    thời gian TƯƠNG ĐỐI so với đầu clip (không phải mốc tuyệt đối của cả
    phim) để Gemma dễ đối chiếu với timestamp nó tự thấy trong video.

    Clip không có thoại nào -> trả chuỗi rỗng, KHÔNG coi là lỗi (nhiều
    scene hành động/không lời vẫn hợp lệ).
    """
    lines = []
    for seg in asr_timeline:
        seg_start, seg_end = seg.get("start", 0.0), seg.get("end", 0.0)
        if seg_end < start or seg_start > end:
            continue  # không giao với cửa sổ [start, end]
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        rel_start = max(0.0, seg_start - start)
        lines.append(f"[{rel_start:05.1f}s] {text}")
    return "\n".join(lines)


def _strip_thinking_block(raw_text: str) -> str:
    """Bóc phần thinking (<|channel>thought...<channel|>, <think>...</think>,
    <start_of_thinking>...</end_of_thinking>, ... tuỳ phiên bản chat template
    của model) ra khỏi output thô của Gemma 4 trước khi đưa vào
    _parse_json_response — nếu không, phần thinking (không phải JSON) sẽ
    luôn khiến json.loads() lỗi.

    Dùng như lưới an toàn CHÍNH cho backend Ollama: content trả về từ
    chat/completions có thể kèm cả phần thinking (Gemma 4 QAT hay bọc JSON
    trả lời sau chuỗi suy nghĩ) — bóc sạch rồi mới parse."""
    text = raw_text or ""
    # Dạng <|channel>thought ... <channel|>
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    # Dạng <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Dạng Gemma 4 sử dụng trên nhiều template: <start_of_thinking>...</end_of_thinking>
    text = re.sub(r"<start_of_thinking>.*?</end_of_thinking>", "", text, flags=re.DOTALL)
    # Dạng <|thinking_start|>...</|thinking_end|>
    text = re.sub(r"<\|thinking_start\|>.*?<\|thinking_end\|>", "", text, flags=re.DOTALL)
    return text.strip()


class Gemma4VideoAnalyzer:
    """Bọc Gemma 4 12B QAT chạy QUA OLLAMA (OpenAI-compatible API), không
    giữ model trong tiến trình Python — Ollama (llama.cpp) tự quản lý RAM.

    Khác các backend còn lại: nhận `scene_clips` (clip .mp4, từ
    preprocess.extract_scene_clips) thay vì `frame_paths` (ảnh tĩnh) — báo
    hiệu qua thuộc tính `input_kind = "clips"` để run_vision_analysis() biết
    đường lấy input đúng (xem input_kind ở _build_analyzer/run_vision_analysis,
    không hardcode theo tên class để dễ mở rộng backend khác sau này).
    """

    input_kind = "clips"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("processing.gemma4_model_name", "gemma4:12b-it-qat")
        self.base_url = str(cfg.get("processing.gemma4_ollama_base_url", "http://127.0.0.1:11434/v1")).rstrip("/")
        # Cơ chế thinking theo mức độ (kế thừa từ handandfeet-agent):
        # "high" (mặc định) | "medium" | "low" | "none".
        self.reasoning_effort = str(cfg.get("processing.gemma4_reasoning_effort", "high")).strip().lower()
        self.max_new_tokens = cfg.get("processing.gemma4_max_new_tokens", 8192)
        self.generation_timeout_sec = cfg.get("processing.gemma4_generation_timeout_sec", 1800)
        self.max_clip_seconds = cfg.get("processing.gemma4_max_clip_seconds", 60.0)
        # Số frame cách đều trích từ mỗi clip để gửi cho Ollama (không gửi
        # video .mp4 nguyên bản — Ollama chỉ nhận ảnh base64).
        self.frames_per_clip = max(1, cfg.get("processing.gemma4_frames_per_clip", 6))
        self.frame_max_side = cfg.get("processing.keyframe_max_side", 768)
        self.temperature = float(cfg.get("processing.gemma4_temperature", 0.0))
        self.ollama_api_key = cfg.get("processing.gemma4_ollama_api_key", "ollama")
        self._session = None
        self._asr_timeline: list[dict[str, Any]] = []

    def set_asr_timeline(self, asr_timeline: list[dict[str, Any]]) -> None:
        """run_vision_analysis() gọi hàm này trước khi phân tích, để
        analyze_scene có phụ đề đối chiếu cho từng clip (xem
        _slice_subtitles_for_window). Nếu không gọi, coi như phim không có
        phụ đề (mọi clip nhận chuỗi phụ đề rỗng, KHÔNG lỗi)."""
        self._asr_timeline = asr_timeline or []

    def _check_server(self) -> None:
        """Kiểm tra Ollama server còn sống + model đã được pull — báo lỗi
        rõ ràng ngay từ đầu thay vì chết ngầm ở clip đầu tiên."""
        import requests
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=15)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Không kết nối được Ollama tại {self.base_url} ({e}). "
                f"Trên GitHub Actions workflow đã tự cài + pull model; nếu chạy "
                f"máy thật/Colab: cài Ollama rồi chạy 'ollama pull {self.model_name}'."
            )
        try:
            names = {m.get("id", "") for m in resp.json().get("data", [])}
        except Exception:
            names = set()
        if not any(n == self.model_name or n.startswith(self.model_name + ":") for n in names):
            raise RuntimeError(
                f"Model '{self.model_name}' chưa có trong Ollama. Chạy lệnh: "
                f"ollama pull {self.model_name}"
            )

    def load(self) -> None:
        """Không tải model vào tiến trình Python — chỉ kiểm tra Ollama
        server sẵn sàng + model đã pull (chi phí gần bằng 0, model nằm trong
        RAM của tiến trình ollama serve)."""
        import requests  # lazy import: chỉ cần khi thực sự dùng backend "gemma4_video"
        if self._session is None:
            self._session = requests.Session()

        print(f"[vision] Gemma 4 12B qua Ollama (model={self.model_name}, "
              f"base_url={self.base_url}, reasoning_effort={self.reasoning_effort}, "
              f"max_clip={self.max_clip_seconds}s, frames/clip={self.frames_per_clip})...")

        self._check_server()
        print("[vision] Ollama sẵn sàng, model đã pull sẵn — sẵn sàng phân tích.")

    def unload(self) -> None:
        """Đẩy model ra khỏi RAM của Ollama để các stage SAU (script chạy
        llama.cpp GGUF ~7GB) không bị OOM khi cả 2 model cùng nằm trong RAM
        runner ~16GB. Dùng `ollama stop` (nhẹ, không chạy generation); nếu
        không có binary ollama thì fallback qua HTTP keep_alive=0."""
        import subprocess
        if self._session is not None:
            try:
                subprocess.run(
                    ["ollama", "stop", self.model_name],
                    timeout=30, capture_output=True, check=False,
                )
            except Exception:
                try:
                    server_root = self.base_url.replace("/v1", "").rstrip("/") or "http://127.0.0.1:11434"
                    self._session.post(
                        f"{server_root}/api/chat",
                        json={
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": "unload"}],
                            "keep_alive": 0,
                            "stream": False,
                        },
                        timeout=30,
                    )
                except Exception:
                    pass
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _extract_clip_frames(self, clip_path: str) -> list[str]:
        """Trích `frames_per_clip` frame cách đều theo thời gian từ clip
        .mp4 bằng OpenCV, resize về `frame_max_side` rồi trả về list chuỗi
        base64 (dạng data URL JPEG) để đưa vào message content cho Ollama.

        Không để lỗi 1 clip làm chết cả pipeline: nếu không mở được video
        hoặc không đọc được frame nào, trả về list rỗng (caller sẽ coi clip
        đó là lỗi "clip_read_error")."""
        import base64 as _b64
        import io as _io

        import cv2  # lazy import: opencv là dependency sẵn có (preprocess dùng chung)
        from PIL import Image

        frames: list[str] = []
        cap = None
        try:
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                return frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                return frames
            n = min(self.frames_per_clip, frame_count)
            if n <= 0:
                return frames
            indices = []
            for i in range(n):
                idx = int(round(i * (frame_count - 1) / max(1, n - 1)))
                indices.append(idx)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((self.frame_max_side, self.frame_max_side), Image.LANCZOS)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                frames.append("data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception as e:
            print(f"[vision] Lỗi trích frame từ {clip_path}: {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return frames

    def _run_ollama_with_watchdog(self, payload: dict) -> dict[str, Any] | None:
        """Gọi POST {base}/chat/completions trong thread nền, chờ tối đa
        `generation_timeout_sec` giây. Ollama không có cách huỷ request giữa
        chừng một cách "sạch" -> nếu quá hạn, BỎ MẶC thread đó (daemon=True,
        giống pattern `_input_with_timeout` trong run.py) và trả về None để
        caller coi clip này là lỗi (fallback _empty_result) thay vì treo vô
        hạn, kéo sập cả job CI."""
        import queue
        import threading

        result_queue: queue.Queue = queue.Queue()

        def _call() -> None:
            try:
                resp = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(30, self.generation_timeout_sec + 60),
                )
                resp.raise_for_status()
                result_queue.put(("ok", resp.json()))
            except Exception as e:
                result_queue.put(("error", e))

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        try:
            status, payload_out = result_queue.get(timeout=self.generation_timeout_sec)
        except queue.Empty:
            return None
        if status == "error":
            raise payload_out
        return payload_out

    def _analyze_clip(self, clip_path: str, start: float, end: float) -> dict[str, Any]:
        """Phân tích 1 sub-clip đơn lẻ (<= max_clip_seconds), trả về dict
        theo schema vision (kèm draft_narration), KHÔNG gắn scene_id (việc
        gắn/gộp nhiều sub-clip về 1 record scene do analyze_scene() lo).

        Gửi `frames_per_clip` frame (base64) + phụ đề cắt theo cửa sổ
        [start, end] cho Gemma 4 qua Ollama, kèm `reasoning_effort` để điều
        khiển độ sâu thinking. Content thô của Ollama có thể chứa phần
        thinking (<think>... / <|channel>thought... hoặc dạng <start_of_thinking>)
        — được bóc bằng _strip_thinking_block trước khi parse JSON."""
        subtitles_text = _slice_subtitles_for_window(self._asr_timeline, start, end)
        subtitles_block = (
            f"[Phụ đề đoạn này]:\n{subtitles_text}" if subtitles_text
            else "[Phụ đề đoạn này]: (không có thoại trong đoạn này)"
        )

        frames_b64 = self._extract_clip_frames(clip_path)
        if not frames_b64:
            return _empty_result("", reason="clip_read_error")

        messages = [
            {"role": "system", "content": GEMMA4_VIDEO_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64}} for b64 in frames_b64
            ] + [
                {"type": "text", "text": subtitles_block + "\n\nAnalyze this clip and return the JSON object described in the system prompt."},
            ]},
        ]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
            "reasoning_effort": self.reasoning_effort,
        }

        from progress_utils import Heartbeat
        with Heartbeat(f"vision:gemma4_video:{Path(clip_path).stem}", interval=15.0):
            response = self._run_ollama_with_watchdog(payload)

        if response is None:
            print(f"[vision] Timeout ({self.generation_timeout_sec}s) khi generate cho clip {clip_path}, bỏ qua.")
            return _empty_result("", reason="generation_timeout")

        try:
            content_text = (response["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            print(f"[vision] Response Ollama thiếu content cho {clip_path}: {e}")
            return _empty_result("", reason="empty_response")

        if not content_text:
            print(f"[vision] Content rỗng cho {clip_path} (thinking có thể ăn hết "
                  f"max_tokens={self.max_new_tokens}). Cân nhắc tăng gemma4_max_new_tokens.")
            return _empty_result("", reason="empty_response")

        content_text = _strip_thinking_block(content_text)

        try:
            parsed = _parse_json_response(content_text)
            parsed["draft_narration"] = ""
            try:
                data_for_narration = json.loads(content_text) if content_text.strip().startswith("{") else {}
            except json.JSONDecodeError:
                start_i, end_i = content_text.find("{"), content_text.rfind("}")
                data_for_narration = {}
                if start_i != -1 and end_i > start_i:
                    try:
                        data_for_narration = json.loads(content_text[start_i:end_i + 1])
                    except json.JSONDecodeError:
                        pass
            parsed["draft_narration"] = str(data_for_narration.get("draft_narration", "") or "")
        except Exception as e:
            print(f"[vision] Lỗi parse JSON (gemma4_video) cho {clip_path}: {e}")
            return _empty_result("", reason="parse_error")

        return parsed

    @staticmethod
    def _merge_clip_results(scene_id: str, clip_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Gộp kết quả nhiều sub-clip (do scene > max_clip_seconds bị chia)
        về ĐÚNG 1 record vision cho cả scene — bắt buộc, vì
        semantic_graph.py/script_writer.py mong đợi 1-scene-1-record.

        Cách gộp:
          - visual_summary / draft_narration: nối theo thứ tự thời gian
            (mỗi sub-clip đã tự đủ ngữ cảnh cho đoạn của nó).
          - actions: nối danh sách theo thứ tự thời gian (giữ thứ tự xảy ra).
          - characters / tags: hợp nhất, bỏ trùng lặp, giữ thứ tự xuất hiện.
          - visual_intensity: lấy MAX (không lấy trung bình) — 1 khoảnh khắc
            đáng chú ý trong scene đủ để coi cả scene "có cường độ cao", lấy
            trung bình sẽ pha loãng khoảnh khắc đó nếu phần còn lại của scene
            tĩnh lặng.
          - emotion / shot_type: lấy từ sub-clip có visual_intensity cao nhất
            (đại diện cho khoảnh khắc đáng chú ý nhất của scene).
          - location: lấy giá trị không rỗng đầu tiên (thường không đổi
            trong 1 scene).
        """
        if not clip_results:
            return _empty_result(scene_id, reason="no_clips")
        if len(clip_results) == 1:
            r = dict(clip_results[0])
            r["scene_id"] = scene_id
            return r

        def _dedup_extend(dst: list, src: list) -> None:
            for item in src:
                if item not in dst:
                    dst.append(item)

        summaries = [r.get("visual_summary", "") for r in clip_results if r.get("visual_summary")]
        narrations = [r.get("draft_narration", "") for r in clip_results if r.get("draft_narration")]
        actions: list[str] = []
        characters: list[str] = []
        tags: list[str] = []
        for r in clip_results:
            actions.extend(r.get("actions", []) or [])
            _dedup_extend(characters, r.get("characters", []) or [])
            _dedup_extend(tags, r.get("tags", []) or [])

        best = max(clip_results, key=lambda r: r.get("visual_intensity", 0.0))
        location = next((r.get("location", "") for r in clip_results if r.get("location")), "")

        return {
            "scene_id": scene_id,
            "visual_summary": " ".join(summaries),
            "characters": characters,
            "location": location,
            "actions": actions,
            "emotion": best.get("emotion", ""),
            "shot_type": best.get("shot_type", ""),
            "visual_intensity": max((r.get("visual_intensity", 0.0) for r in clip_results), default=0.0),
            "tags": tags,
            "draft_narration": " ".join(narrations),
        }

    def analyze_scene(self, scene_id: str, scene_clips: list[dict[str, Any]]) -> dict[str, Any]:
        if not scene_clips:
            return _empty_result(scene_id, reason="no_clips")

        clip_results = []
        for clip in scene_clips:
            clip_path = clip.get("clip_path")
            if not clip_path or not Path(clip_path).exists():
                clip_results.append(_empty_result("", reason="clip_missing"))
                continue
            try:
                result = self._analyze_clip(clip_path, clip.get("start", 0.0), clip.get("end", 0.0))
            except Exception as e:
                # Lỗi từng clip (model lỗi, OOM, v.v.) -> fallback rỗng cho
                # ĐÚNG clip đó, không để văng exception làm chết cả pipeline
                # (đúng triết lý xử lý lỗi hiện có trong file — xem
                # MoondreamVisionAnalyzer.analyze_scene).
                print(f"[vision] Lỗi Gemma4VideoAnalyzer cho clip {clip_path}: {e}")
                result = _empty_result("", reason="model_error")
            clip_results.append(result)

        return self._merge_clip_results(scene_id, clip_results)


def _build_analyzer(cfg):
    backend = cfg.get("processing.vision_backend", "local")
    if backend == "cerebras":
        return CerebrasVisionAnalyzer(cfg)
    if backend == "mistral":
        return MistralVisionAnalyzer(cfg)
    if backend == "moondream":
        return MoondreamVisionAnalyzer(cfg)
    if backend == "gemma4_video":
        return Gemma4VideoAnalyzer(cfg)
    return LocalVisionAnalyzer(cfg)


def run_vision_analysis(
    cfg,
    preprocess_result: dict[str, Any],
    checkpoint_mgr=None,
    asr_timeline: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Entry point cho stage 'vision'. Ghi vision_analysis.json vào pipeline/.

    asr_timeline: CHỈ cần cho backend "gemma4_video" (đối chiếu phụ đề với
    từng clip — xem _slice_subtitles_for_window). Tham số optional, mặc định
    None để không phá các lời gọi cũ (run.py) trước khi có thay đổi này —
    nếu None và backend là gemma4_video, mọi clip nhận phụ đề rỗng (không lỗi,
    chỉ mất khả năng đối chiếu thoại)."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    scenes = preprocess_result["scenes"]
    keyframes = preprocess_result["keyframes"]
    scene_clips = preprocess_result.get("scene_clips", {})

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
    # "frames" (mặc định, backend cũ dùng keyframes ảnh tĩnh) | "clips"
    # (gemma4_video dùng scene_clips .mp4) — đọc qua thuộc tính thay vì
    # if/else cứng theo tên class, để dễ mở rộng backend khác sau này mà
    # không phải sửa lại hàm này.
    input_kind = getattr(analyzer, "input_kind", "frames")
    if input_kind == "clips" and hasattr(analyzer, "set_asr_timeline"):
        analyzer.set_asr_timeline(asr_timeline or [])

    if remaining_scenes:
        analyzer.load()
        try:
            if hasattr(analyzer, "analyze_batch"):
                # Backend batch (Local/Cerebras/Mistral) luôn dùng keyframes
                # (ảnh) — gemma4_video không có analyze_batch nên không rơi
                # vào nhánh này.
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
                    if input_kind == "clips":
                        scene_input = scene_clips.get(scene_id, [])
                        input_count = len(scene_input)
                        input_label = "clips"
                    else:
                        scene_input = keyframes.get(scene_id, [])
                        input_count = len(scene_input)
                        input_label = "frames"
                    analysis = analyzer.analyze_scene(scene_id, scene_input)
                    analysis["start"] = scene["start"]
                    analysis["end"] = scene["end"]
                    results_by_scene[scene_id] = analysis
                    done_total = total_scenes - total_remaining + scene_idx
                    print_progress_bar(
                        done_total, total_scenes,
                        prefix="[vision] analyzing",
                        suffix=f"{scene_id} ({input_count} {input_label})",
                    )
                    # LUÔN lưu checkpoint ngay sau mỗi scene (không throttle theo
                    # micro_interval) — giống nhánh batch ở trên: mỗi scene ở đây
                    # đã tốn tiền/quota gọi API backend (Cerebras/Mistral) hoặc thời
                    # gian generate() (gemma4_video), nên nếu throttle và pipeline
                    # bị ngắt giữa chừng, các scene đã xong nhưng chưa tới mốc lưu
                    # sẽ mất trắng và phải tốn công chạy lại. Việc throttle tần suất
                    # SYNC LÊN CLOUD đã được xử lý riêng trong
                    # CheckpointManager.save_micro().
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
