"""
asr.py — sinh Dialogue Timeline (asr_timeline.json).

Hỗ trợ 2 backend ASR, chọn qua `processing.asr_backend` trong config.toml:
  - "funasr"  (MẶC ĐỊNH): FunASR (Paraformer/SenseVoice).
  - "whisper": faster-whisper.

Dù chọn "funasr", nếu backend này lỗi ở BẤT KỲ bước nào (thiếu package
`funasr`, lỗi tải model, lỗi lúc transcribe...) thì run_asr() tự động
fallback sang faster-whisper — không cần sửa code, chỉ cần đổi
`asr_backend` trong config.toml nếu muốn tắt hẳn FunASR.

faster-whisper: tự động chọn device/compute_type theo config + fallback về
CPU nếu CUDA/cublas không khả dụng (đúng theo hành vi mô tả trong
references gốc, nhưng giờ đọc toàn bộ tham số từ config.toml thay vì
hard-code path Windows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from platform_utils import resolve_torch_device
from progress_utils import print_progress_bar


# =============================================================================
# faster-whisper (phương án dự phòng / asr_backend="whisper")
# =============================================================================

def _resolve_whisper_device(preferred: str) -> str:
    # faster-whisper (CTranslate2) chỉ hỗ trợ "cuda" hoặc "cpu", không có "mps".
    device = resolve_torch_device(preferred)
    return "cpu" if device == "mps" else device


def _resolve_compute_type(preferred: str, device: str) -> str:
    if preferred == "auto":
        return "float16" if device == "cuda" else "int8"
    return preferred


def load_whisper_model(cfg):
    """
    Load faster-whisper model. Model được tự động tải về `paths.model_cache_dir`
    (faster-whisper dùng huggingface_hub cache dưới nền, ta chỉ cần trỏ
    HF cache dir bằng biến môi trường tại thời điểm import — xem run.py).
    """
    from faster_whisper import WhisperModel

    model_size = cfg.get("processing.asr_model_size", "small")
    device = _resolve_whisper_device(cfg.get("processing.asr_device", "auto"))
    compute_type = _resolve_compute_type(cfg.get("processing.asr_compute_type", "auto"), device)
    cache_dir = str(cfg.resolve_path("paths.model_cache_dir"))

    print(f"[asr:whisper] Loading faster-whisper '{model_size}' on {device} ({compute_type})... "
          f"(lần đầu sẽ tải model; log tải % / tốc độ của huggingface_hub sẽ hiện ngay bên dưới)")
    try:
        # Không bọc Heartbeat ở đây: huggingface_hub đã tự in thanh tiến độ tải
        # (%, MB/s, ETA) qua tqdm. Bọc thêm sẽ khiến 2 log cùng ghi \r đè lên
        # nhau và người dùng chỉ thấy dòng "vẫn đang chạy..." thay vì log thật.
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
    except RuntimeError as e:
        if device == "cuda" and "cublas" in str(e).lower():
            print("[asr:whisper] cublas không khả dụng, fallback sang CPU (int8).")
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=cache_dir,
            )
        else:
            raise
    return model


def transcribe_whisper(model, audio_path: Path, language: str | None = None) -> list[dict[str, Any]]:
    """Chạy ASR + VAD trên file audio bằng faster-whisper, trả về danh sách
    segment chuẩn hoá (start/end/text/confidence)."""
    lang = language or None
    segments, info = model.transcribe(str(audio_path), language=lang, vad_filter=True)
    total_dur = getattr(info, "duration", 0.0) or 0.0

    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "confidence": round(seg.avg_logprob, 4),
        })
        if total_dur > 0:
            # faster-whisper trả segment tuần tự theo thời gian -> seg.end / total_dur ~ % đã xử lý.
            done_pct = min(seg.end / total_dur, 1.0)
            print_progress_bar(
                int(done_pct * 1000), 1000,
                prefix="[asr:whisper] transcribing",
                suffix=f"{seg.end:.0f}s/{total_dur:.0f}s ({len(results)} đoạn)",
            )
    print_progress_bar(1000, 1000, prefix="[asr:whisper] transcribing", suffix=f"xong ({len(results)} đoạn)")
    return results


def _run_whisper(cfg, audio_path: Path, language: str | None) -> list[dict[str, Any]]:
    model = load_whisper_model(cfg)
    print("[asr:whisper] Transcribing...")
    timeline = transcribe_whisper(model, audio_path, language=language)
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return timeline


# =============================================================================
# FunASR (backend chính / asr_backend="funasr")
# =============================================================================

def _resolve_funasr_device(preferred: str) -> str:
    # FunASR (giống faster-whisper) chỉ hỗ trợ tốt "cuda"/"cpu"; MPS chưa
    # được hỗ trợ chính thức -> coi như "cpu".
    device = resolve_torch_device(preferred)
    return "cpu" if device == "mps" else device


def load_funasr_model(cfg):
    """
    Load FunASR AutoModel (ASR + VAD + Punctuation) theo cấu hình
    `processing.asr_funasr_*`. Model được cache theo cơ chế mặc định của
    FunASR/modelscope (thư mục ~/.cache, không dùng chung model_cache_dir
    của faster-whisper vì 2 thư viện quản lý cache khác nhau).
    """
    from funasr import AutoModel

    asr_model = cfg.get("processing.asr_funasr_model", "paraformer-zh")
    vad_model = cfg.get("processing.asr_funasr_vad_model", "fsmn-vad") or None
    punc_model = cfg.get("processing.asr_funasr_punc_model", "ct-punc") or None
    device = _resolve_funasr_device(cfg.get("processing.asr_funasr_device", "auto"))

    print(f"[asr:funasr] Loading FunASR '{asr_model}' "
          f"(vad='{vad_model}', punc='{punc_model}') on {device}... "
          f"(lần đầu sẽ tải model, có thể mất vài phút)")
    model = AutoModel(
        model=asr_model,
        vad_model=vad_model,
        punc_model=punc_model,
        device=device,
        disable_update=True,
    )
    return model


def transcribe_funasr(model, audio_path: Path, language: str | None = None) -> list[dict[str, Any]]:
    """Chạy ASR bằng FunASR, chuẩn hoá kết quả về cùng schema với
    transcribe_whisper() (start/end/text/confidence) để phần còn lại của
    pipeline (semantic_graph, script_writer...) dùng chung 1 định dạng,
    không cần biết ASR nào đã chạy.

    FunASR trả timestamp ở mili-giây (VAD + punc bật -> có "sentence_info"
    theo từng câu); nếu vì lý do gì đó (model/param khác) không có
    "sentence_info", fallback về 1 segment duy nhất chứa toàn bộ text.
    """
    kwargs: dict[str, Any] = {"input": str(audio_path), "sentence_timestamp": True}
    if language and language != "auto":
        kwargs["language"] = language

    print("[asr:funasr] Transcribing...")
    raw = model.generate(**kwargs)

    results: list[dict[str, Any]] = []
    if raw:
        first = raw[0]
        sentence_info = first.get("sentence_info")
        if sentence_info:
            for sent in sentence_info:
                start_ms = sent.get("start", 0)
                end_ms = sent.get("end", start_ms)
                text = str(sent.get("text", "")).strip()
                if not text:
                    continue
                results.append({
                    "start": round(start_ms / 1000.0, 2),
                    "end": round(end_ms / 1000.0, 2),
                    "text": text,
                    # FunASR không trả confidence/log-prob theo câu như
                    # whisper -> để None thay vì bịa 1 con số không có ý nghĩa.
                    "confidence": None,
                })
        else:
            text = str(first.get("text", "")).strip()
            if text:
                results.append({"start": 0.0, "end": 0.0, "text": text, "confidence": None})

    print_progress_bar(1000, 1000, prefix="[asr:funasr] transcribing", suffix=f"xong ({len(results)} đoạn)")
    return results


def _run_funasr(cfg, audio_path: Path, language: str | None) -> list[dict[str, Any]]:
    funasr_language = cfg.get("processing.asr_funasr_language", "auto") or "auto"
    model = load_funasr_model(cfg)
    timeline = transcribe_funasr(model, audio_path, language=funasr_language)
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return timeline


# =============================================================================
# Entry point chung
# =============================================================================

def run_asr(cfg, preprocess_result: dict[str, Any], checkpoint_mgr=None) -> list[dict[str, Any]]:
    """Entry point cho stage 'asr'. Ghi asr_timeline.json vào pipeline/.

    Chọn backend qua `processing.asr_backend` ("funasr" mặc định, hoặc
    "whisper"). Nếu backend "funasr" lỗi ở bất kỳ bước nào (import, load
    model, transcribe), tự động fallback sang faster-whisper thay vì làm
    crash cả pipeline — đúng yêu cầu "FunASR chính, whisper dự phòng".
    """
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(preprocess_result["audio_path"])
    language = cfg.get("processing.asr_language", "") or None
    backend = (cfg.get("processing.asr_backend", "funasr") or "funasr").strip().lower()

    timeline: list[dict[str, Any]] | None = None
    used_backend = backend

    if backend == "funasr":
        try:
            timeline = _run_funasr(cfg, audio_path, language)
            if not timeline:
                # generate() chạy được nhưng không trả về câu nào (vd audio
                # câm/lỗi định dạng) -> coi như thất bại, thử whisper thay vì
                # ghi ra 1 asr_timeline.json rỗng một cách âm thầm.
                print("[asr] FunASR không trả về đoạn thoại nào, fallback sang faster-whisper.")
                raise RuntimeError("FunASR trả về kết quả rỗng")
        except Exception as e:
            print(f"[asr] CẢNH BÁO: FunASR lỗi ({type(e).__name__}: {e}) "
                  f"-> fallback sang faster-whisper. Nếu muốn tắt hẳn FunASR, "
                  f"đổi processing.asr_backend = \"whisper\" trong config.toml.")
            timeline = None
            used_backend = "whisper (fallback từ funasr)"

    if timeline is None:
        timeline = _run_whisper(cfg, audio_path, language)
        if backend == "whisper":
            used_backend = "whisper"

    with open(pipeline_dir / "asr_timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"[asr] Xong ({used_backend}): {len(timeline)} đoạn thoại.")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("asr", timeline)

    return timeline
