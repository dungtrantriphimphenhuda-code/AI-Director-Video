"""
vision.py — phân tích thị giác cho từng scene.

Hỗ trợ 3 backend, chọn qua config.toml [processing] vision_backend:
  - "local"    : Qwen3-VL-4B-Instruct tải từ Hugging Face, chạy qua `transformers`
                 (cần GPU tốt, chậm với video nhiều scene vì generate() tuần tự
                 từng scene một).
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


def _build_analyzer(cfg):
    backend = cfg.get("processing.vision_backend", "local")
    if backend == "cerebras":
        return CerebrasVisionAnalyzer(cfg)
    if backend == "mistral":
        return MistralVisionAnalyzer(cfg)
    return LocalVisionAnalyzer(cfg)


def run_vision_analysis(cfg, preprocess_result: dict[str, Any], checkpoint_mgr=None) -> list[dict[str, Any]]:
    """Entry point cho stage 'vision'. Ghi vision_analysis.json vào pipeline/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    scenes = preprocess_result["scenes"]
    keyframes = preprocess_result["keyframes"]

    total_scenes = len(scenes)
    micro_interval = cfg.get("processing.micro_checkpoint_interval", 1)

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
                    if checkpoint_mgr is not None and scene_idx % micro_interval == 0:
                        checkpoint_mgr.save_micro("vision", scene_id, analysis)
        finally:
            analyzer.unload()
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
