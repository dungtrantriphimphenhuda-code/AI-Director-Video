"""
tts.py — sinh giọng đọc (voiceover) cho từng câu trong storyboard.

Hỗ trợ 4 engine, chọn qua `tts.engine` trong config.toml:
  - "edge-tts" (mặc định): dùng edge-tts, không cần API key, không cần GPU.
    LƯU Ý: đây là API không chính thức của Microsoft, thỉnh thoảng bị chặn/
    rớt kết nối khi chạy từ IP datacenter (vd GitHub Actions) — lỗi
    "edge_tts.exceptions.NoAudioReceived".
  - "viterbox": model TTS tiếng Việt chạy LOCAL (fine-tune từ Chatterbox,
    xem viterbox_env.py / viterbox_worker.py), không phụ thuộc dịch vụ ngoài
    nên không bị lỗi mạng như trên. Cần tải model (~3-4GB) lần đầu, chạy
    nhanh trên GPU và chậm hơn nhiều trên CPU. Chạy trong 1 venv RIÊNG
    (.viterbox_venv) vì package `viterbox` ghim version numpy/transformers
    xung đột với requirements.txt chính của project. Chất lượng giọng cao
    nhất trong 4 engine (voice cloning), nhưng nặng/rắc rối nhất để vận hành.
  - "piper": model TTS LOCAL cực nhẹ (VITS, ~15-60MB tuỳ giọng, chạy qua
    ONNX Runtime CPU) — KHÔNG cần GPU, KHÔNG cần venv riêng, tốc độ gần như
    real-time ngay cả trên CPU yếu. Chất lượng giọng ở mức "nghe hiểu tốt",
    không tự nhiên/biểu cảm bằng Viterbox hay Gemini, nhưng đơn giản và ổn
    định nhất trong 4 engine — phù hợp khi ưu tiên nhẹ/nhanh/không rắc rối
    hơn là chất lượng giọng đỉnh cao. Không hỗ trợ voice cloning.
  - "gemini": gọi Gemini TTS API của Google (cần `api.gemini_api_key` trong
    config.toml) — chất lượng giọng rất tự nhiên, hỗ trợ tiếng Việt native,
    KHÔNG cần GPU/venv riêng vì chỉ là API call. Chi phí cực rẻ (~0.01-0.05
    USD cho cả 1 video ~10-15 phút, tính theo token audio output). Nhược
    điểm: cần internet + API key, model đang ở trạng thái Preview (Google có
    thể đổi giới hạn/giá bất cứ lúc nào), và có thể lệch tiến độ TTS nếu
    request bị timeout/lỗi tạm thời (đã có retry tự động). Audio được gắn
    watermark SynthID (không nghe thấy được, chỉ máy nhận diện).

Mỗi câu narration được tổng hợp thành 1 file audio riêng, sau đó ghép lại
thành 1 file audio tổng (voiceover.mp3) với timestamp đặt đúng theo
`timeline[].output.start` trong storyboard.json — để đồng bộ với video render
ở bước sau.

edge-tts và gemini dùng asyncio + semaphore (chạy song song nhiều câu, giới
hạn số luồng đồng thời) vì đây là I/O mạng, hưởng lợi rõ từ việc chạy song
song. viterbox chạy đồng bộ qua subprocess (model AI cục bộ, không hưởng lợi
gì từ song song). piper chạy đồng bộ tuần tự (model ONNX cực nhẹ, mỗi câu
chỉ mất một phần nhỏ của giây trên CPU nên không cần song song hoá).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import edge_tts

from progress_utils import print_progress_bar, run_ffmpeg_with_progress


async def _synthesize_one_with_retry(
    text: str, voice: str, rate: str, volume: str, pitch: str, out_path: Path,
    max_retries: int = 3,
) -> None:
    """Sinh 1 file audio, tự retry nếu edge-tts báo lỗi thoáng qua (mất mạng
    tạm thời, timeout...). edge-tts nhìn chung rất ổn định, lỗi thường tự
    qua khi thử lại ngay, không cần backoff dài như API có rate-limit thật."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume, pitch=pitch)
            await communicate.save(str(out_path))
            return
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(1.0 * attempt)
    raise RuntimeError(f"edge-tts thất bại sau {max_retries} lần thử: {last_err}") from last_err


async def _synthesize_all(
    sentences: list[dict[str, Any]],
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    out_dir: Path,
    checkpoint_mgr=None,
    max_concurrency: int = 40,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(sentences)
    # Danh sách kết quả theo ĐÚNG thứ tự sentences gốc (quan trọng: mix_voiceover_track
    # ghép audio dựa vào thứ tự này khớp với timeline) dù các câu hoàn thành
    # không theo thứ tự khi chạy song song.
    paths: list[Path | None] = [None] * total

    # --- Resume: bỏ qua câu đã tổng hợp xong ở lần chạy trước ---
    done_ids: set[str] = set()
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("tts")

    # edge-tts rất ổn định + không có rate-limit cứng như các API trả phí ở
    # vision.py, nên chạy SONG SONG nhiều câu cùng lúc thay vì tuần tự từng
    # câu như trước -- giới hạn bằng semaphore để tránh mở quá nhiều kết nối
    # cùng lúc (an toàn thực tế: ~40 luồng đồng thời).
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    progress = {"done": 0, "skipped": 0}

    async def _run_one(idx: int, sent: dict[str, Any]) -> None:
        clip_id = sent["clip_id"]
        out_path = out_dir / f"{clip_id}.mp3"

        if clip_id in done_ids and out_path.exists():
            paths[idx] = out_path
            progress["done"] += 1
            progress["skipped"] += 1
            print_progress_bar(progress["done"], total, prefix="[tts] synthesizing",
                                suffix=f"{clip_id} (cached)")
            return

        async with semaphore:
            await _synthesize_one_with_retry(sent["sentence"], voice, rate, volume, pitch, out_path)

        paths[idx] = out_path
        progress["done"] += 1
        print_progress_bar(progress["done"], total, prefix="[tts] synthesizing", suffix=clip_id)

        # Micro-checkpoint NGAY sau khi câu này xong. checkpoint_mgr.save_micro()
        # gọi boto3 đồng bộ (chặn) để upload cloud -> đẩy sang thread riêng
        # bằng asyncio.to_thread() để KHÔNG chặn event loop, giữ đúng nhiều
        # luồng edge-tts chạy song song thật sự thay vì bị nghẽn cổ chai ở
        # bước ghi checkpoint (CheckpointManager đã được thêm lock để an
        # toàn khi nhiều luồng gọi save_micro() đồng thời -- xem checkpoint.py).
        if checkpoint_mgr is not None:
            await asyncio.to_thread(checkpoint_mgr.save_micro, "tts", clip_id, {
                "clip_id": clip_id,
                "audio_path": str(out_path),
            })

    await asyncio.gather(*(_run_one(i, sent) for i, sent in enumerate(sentences)))

    if checkpoint_mgr is not None and sentences:
        # Lưới an toàn cuối stage: đảm bảo MỌI câu đã tổng hợp trong lần
        # chạy này thực sự lên cloud.
        checkpoint_mgr.flush_pending_syncs()

    if progress["skipped"]:
        print(f"[tts] Bỏ qua {progress['skipped']}/{total} câu đã tổng hợp sẵn (resume từ checkpoint).")

    return paths  # type: ignore[return-value]


def _cuda_available_hint() -> bool:
    """Kiểm tra nhanh có GPU NVIDIA khả dụng không. Import torch CỤC BỘ (không
    ở module scope) vì torch có thể chưa được cài trong tiến trình chính nếu
    người dùng chỉ dùng backend API cho vision/script -- best-effort, không
    quan trọng nếu không xác định được (mặc định coi như không có GPU)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _synthesize_all_viterbox(
    sentences: list[dict[str, Any]],
    cfg,
    out_dir: Path,
    checkpoint_mgr=None,
) -> list[Path]:
    """
    Sinh audio bằng backend 'viterbox': model TTS tiếng Việt chạy LOCAL (fine-
    tune từ Chatterbox), KHÔNG phụ thuộc dịch vụ đám mây của Microsoft như
    edge-tts -- không bị ảnh hưởng bởi việc edge-tts hay bị chặn/rớt kết nối
    (lỗi "NoAudioReceived") khi chạy từ IP datacenter như GitHub Actions.

    Chạy trong 1 venv RIÊNG (.viterbox_venv, xem viterbox_env.py) qua
    subprocess vì package `viterbox` ghim version numpy/transformers xung đột
    với requirements.txt chính -- KHÔNG import/chạy trực tiếp trong tiến
    trình Python hiện tại.
    """
    import viterbox_env  # import cục bộ: chỉ cần khi thực sự dùng engine này

    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent
    venv_python = viterbox_env.ensure_viterbox_env(project_root)

    device = cfg.get("tts.viterbox_device") or ("cuda" if _cuda_available_hint() else "cpu")
    if device == "cpu":
        print("[tts] CẢNH BÁO: chạy Viterbox trên CPU (không có GPU) — chậm hơn NHIỀU "
              "so với GPU, có thể vài chục giây/câu. Với video dài + chạy trên GitHub "
              "Actions runner miễn phí (không có GPU), stage này có thể mất rất lâu; "
              "cân nhắc chạy trên máy/Colab có GPU nếu cần nhanh hơn.")

    audio_prompt = cfg.get("tts.viterbox_audio_prompt")
    if audio_prompt:
        audio_prompt = str(cfg.resolve_path("tts.viterbox_audio_prompt"))

    done_ids: set[str] = set()
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("tts")

    total = len(sentences)
    paths: list[Path | None] = [None] * total
    pending: list[dict[str, Any]] = []
    for i, sent in enumerate(sentences):
        clip_id = sent["clip_id"]
        out_path = out_dir / f"{clip_id}.wav"
        paths[i] = out_path
        if clip_id in done_ids and out_path.exists():
            continue
        pending.append({"clip_id": clip_id, "text": sent["sentence"], "out_path": str(out_path), "_idx": i})

    skipped = total - len(pending)
    if skipped:
        print(f"[tts] Bỏ qua {skipped}/{total} câu đã tổng hợp sẵn (resume từ checkpoint).")

    if not pending:
        return paths  # type: ignore[return-value]

    job = {
        "device": device,
        "language": cfg.get("tts.viterbox_language", "vi"),
        "audio_prompt": audio_prompt,
        "exaggeration": cfg.get("tts.viterbox_exaggeration", 0.5),
        "cfg_weight": cfg.get("tts.viterbox_cfg_weight", 0.5),
        "temperature": cfg.get("tts.viterbox_temperature", 0.8),
        "items": [{"clip_id": it["clip_id"], "text": it["text"], "out_path": it["out_path"]} for it in pending],
    }
    job_path = out_dir / "_viterbox_job.json"
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)

    worker_script = project_root / "viterbox_worker.py"
    print(f"[tts] Synthesizing {len(pending)} câu bằng Viterbox (device={device})...")

    idx_by_clip = {it["clip_id"]: it["_idx"] for it in pending}
    done_count = skipped
    proc = subprocess.Popen(
        [str(venv_python), str(worker_script), str(job_path)],
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line == "VITERBOX_MODEL_LOADED":
            print("[tts] Viterbox model đã load xong, bắt đầu tổng hợp...")
        elif line.startswith("VITERBOX_OK "):
            clip_id = line.split(" ", 1)[1]
            done_count += 1
            print_progress_bar(done_count, total, prefix="[tts] synthesizing", suffix=clip_id)
            if checkpoint_mgr is not None:
                idx = idx_by_clip[clip_id]
                checkpoint_mgr.save_micro("tts", clip_id, {
                    "clip_id": clip_id,
                    "audio_path": str(paths[idx]),
                })
        elif line.startswith("VITERBOX_ERR "):
            print(f"[tts] LỖI từ viterbox-worker: {line}")
        elif line != "VITERBOX_DONE":
            print(f"[viterbox-worker] {line}")

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"viterbox_worker.py thoát với mã lỗi {returncode} — xem log phía trên để biết câu "
            f"nào lỗi. Checkpoint đã lưu các câu tổng hợp thành công trước đó, chạy lại pipeline "
            f"sẽ resume tiếp từ đúng chỗ dừng."
        )

    if checkpoint_mgr is not None:
        checkpoint_mgr.flush_pending_syncs()

    return paths  # type: ignore[return-value]


def _ensure_piper_voice(cfg, voice_id: str) -> tuple[Path, Path]:
    """
    Đảm bảo 2 file (.onnx model + .onnx.json config) của giọng Piper `voice_id`
    đã có sẵn cục bộ, tự tải về (từ HuggingFace, qua CLI `piper.download_voices`
    có sẵn của package) nếu chưa có. Idempotent — lần chạy sau chỉ cần vài ms
    để kiểm tra file đã tồn tại, không tải lại.
    """
    data_dir = cfg.get("tts.piper_data_dir")
    if data_dir:
        data_dir = cfg.resolve_path("tts.piper_data_dir")
    else:
        data_dir = cfg.resolve_path("paths.model_cache_dir") / "piper_voices"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path = data_dir / f"{voice_id}.onnx"
    config_path = data_dir / f"{voice_id}.onnx.json"
    if model_path.exists() and config_path.exists():
        return model_path, config_path

    print(f"[tts] Tải giọng Piper '{voice_id}' về {data_dir} (chỉ cần 1 lần, "
          f"file rất nhẹ — vài chục MB)...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", "--data-dir", str(data_dir), voice_id],
        check=True,
    )
    if not (model_path.exists() and config_path.exists()):
        raise RuntimeError(
            f"[tts] Tải giọng Piper '{voice_id}' xong nhưng không thấy file mong đợi tại "
            f"{model_path} / {config_path} — kiểm tra lại tên giọng (tts.piper_voice) có đúng "
            f"định dạng '<mã_ngôn_ngữ>-<tên_giọng>-<chất_lượng>' không (vd 'vi_VN-vais1000-medium')."
        )
    return model_path, config_path


def _synthesize_all_piper(
    sentences: list[dict[str, Any]],
    cfg,
    out_dir: Path,
    checkpoint_mgr=None,
) -> list[Path]:
    """
    Sinh audio bằng backend 'piper': model TTS LOCAL cực nhẹ (VITS qua ONNX
    Runtime CPU) — không cần GPU, không cần venv riêng, cài đặt/vận hành đơn
    giản nhất trong các engine local. Đổi lại, giọng đọc không tự nhiên/biểu
    cảm bằng Viterbox (Chatterbox fine-tune) hay Gemini TTS, nhưng đủ rõ ràng,
    dễ nghe, tốc độ gần real-time ngay cả trên máy yếu.
    """
    from piper import PiperVoice  # import cục bộ: chỉ cần khi thực sự dùng engine này

    voice_id = cfg.get("tts.piper_voice", "vi_VN-vais1000-medium")
    model_path, config_path = _ensure_piper_voice(cfg, voice_id)

    print(f"[tts] Đang load giọng Piper '{voice_id}'...")
    voice = PiperVoice.load(str(model_path), config_path=str(config_path))
    # 2 tên hàm khác nhau tuỳ version package piper-tts đang cài
    # (bản mới: synthesize_wav, bản cũ hơn: synthesize) — dò để tương thích cả hai.
    synth_fn = getattr(voice, "synthesize_wav", None) or getattr(voice, "synthesize", None)
    if synth_fn is None:
        raise RuntimeError(
            "[tts] Package 'piper-tts' đang cài không có method synthesize_wav/synthesize "
            "trên PiperVoice — có thể version quá cũ/mới không tương thích, thử "
            "'pip install -U piper-tts'."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("tts")

    total = len(sentences)
    paths: list[Path] = []
    done_count = 0
    skipped = 0
    print(f"[tts] Synthesizing {total} câu bằng Piper (local, CPU)...")
    for sent in sentences:
        clip_id = sent["clip_id"]
        out_path = out_dir / f"{clip_id}.wav"

        if clip_id in done_ids and out_path.exists():
            paths.append(out_path)
            done_count += 1
            skipped += 1
            print_progress_bar(done_count, total, prefix="[tts] synthesizing",
                                suffix=f"{clip_id} (cached)")
            continue

        with wave.open(str(out_path), "wb") as wav_file:
            synth_fn(sent["sentence"], wav_file)

        paths.append(out_path)
        done_count += 1
        print_progress_bar(done_count, total, prefix="[tts] synthesizing", suffix=clip_id)

        if checkpoint_mgr is not None:
            checkpoint_mgr.save_micro("tts", clip_id, {
                "clip_id": clip_id,
                "audio_path": str(out_path),
            })

    if skipped:
        print(f"[tts] Bỏ qua {skipped}/{total} câu đã tổng hợp sẵn (resume từ checkpoint).")
    if checkpoint_mgr is not None:
        checkpoint_mgr.flush_pending_syncs()

    return paths


def _gemini_call_sync(client, model: str, voice_name: str, input_text: str) -> bytes:
    """Gọi Gemini TTS API (đồng bộ — chạy trong thread riêng qua asyncio.to_thread
    ở nơi gọi, để không chặn event loop khi chạy song song nhiều câu)."""
    import base64

    interaction = client.interactions.create(
        model=model,
        input=input_text,
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": voice_name}]},
    )
    return base64.b64decode(interaction.output_audio.data)


async def _gemini_synthesize_one_with_retry(
    client, model: str, voice_name: str, style_prompt: str, text: str, out_path: Path,
    max_retries: int = 3,
) -> None:
    """Sinh 1 file audio qua Gemini TTS, tự retry nếu gặp lỗi thoáng qua. Theo
    tài liệu chính thức, model TTS của Gemini thỉnh thoảng trả về text token
    thay vì audio token (~1 tỉ lệ nhỏ ngẫu nhiên request), gây lỗi 500 — nên
    cần retry logic ở phía client thay vì coi đây là lỗi cố định."""
    input_text = f"{style_prompt}\n\n{text}" if style_prompt else text
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            pcm_bytes = await asyncio.to_thread(_gemini_call_sync, client, model, voice_name, input_text)
            # Gemini TTS trả PCM 16-bit mono 24kHz (theo tài liệu chính thức) -> đóng gói WAV.
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_bytes)
            return
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(2.0 * attempt)
    raise RuntimeError(f"Gemini TTS thất bại sau {max_retries} lần thử: {last_err}") from last_err


async def _synthesize_all_gemini(
    sentences: list[dict[str, Any]],
    cfg,
    out_dir: Path,
    checkpoint_mgr=None,
) -> list[Path]:
    """
    Sinh audio bằng backend 'gemini': gọi Gemini TTS API của Google. Mỗi câu
    được gửi thành 1 request riêng (khớp với khuyến nghị chính thức: chất
    lượng audio có thể giảm dần với output dài hơn vài phút trong CÙNG 1
    session -> chia nhỏ theo câu là cách an toàn nhất, và cũng khớp sẵn với
    kiến trúc checkpoint/resume theo câu của pipeline này).
    """
    from google import genai  # import cục bộ: chỉ cần khi thực sự dùng engine này

    api_key = cfg.get("api.gemini_api_key", "")
    if not api_key or api_key.startswith("PASTE_"):
        raise ValueError(
            "Chưa cấu hình api.gemini_api_key trong config.toml. Lấy API key miễn phí tại "
            "https://aistudio.google.com/apikey rồi điền vào config.toml (mục [api] "
            "gemini_api_key)."
        )
    client = genai.Client(api_key=api_key)
    model = cfg.get("tts.gemini_model", "gemini-2.5-flash-preview-tts")
    voice_name = cfg.get("tts.gemini_voice", "Kore")
    style_prompt = cfg.get("tts.gemini_style_prompt", "")
    max_concurrency = cfg.get("tts.gemini_max_concurrency", 5)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(sentences)
    paths: list[Path | None] = [None] * total

    done_ids: set[str] = set()
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("tts")

    # Concurrency thấp hơn hẳn edge-tts (40) vì đây là API TRẢ PHÍ có rate-limit
    # thực sự (RPM giới hạn ở free tier/preview) -- mặc định 5 là mức an toàn,
    # tăng lên nếu tài khoản có quota cao hơn.
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    progress = {"done": 0, "skipped": 0}

    async def _run_one(idx: int, sent: dict[str, Any]) -> None:
        clip_id = sent["clip_id"]
        out_path = out_dir / f"{clip_id}.wav"

        if clip_id in done_ids and out_path.exists():
            paths[idx] = out_path
            progress["done"] += 1
            progress["skipped"] += 1
            print_progress_bar(progress["done"], total, prefix="[tts] synthesizing",
                                suffix=f"{clip_id} (cached)")
            return

        async with semaphore:
            await _gemini_synthesize_one_with_retry(
                client, model, voice_name, style_prompt, sent["sentence"], out_path
            )

        paths[idx] = out_path
        progress["done"] += 1
        print_progress_bar(progress["done"], total, prefix="[tts] synthesizing", suffix=clip_id)

        if checkpoint_mgr is not None:
            await asyncio.to_thread(checkpoint_mgr.save_micro, "tts", clip_id, {
                "clip_id": clip_id,
                "audio_path": str(out_path),
            })

    print(f"[tts] Synthesizing {total} câu bằng Gemini TTS (model={model}, voice={voice_name}, "
          f"song song tối đa {max_concurrency} luồng)...")
    await asyncio.gather(*(_run_one(i, sent) for i, sent in enumerate(sentences)))

    if checkpoint_mgr is not None and sentences:
        checkpoint_mgr.flush_pending_syncs()

    if progress["skipped"]:
        print(f"[tts] Bỏ qua {progress['skipped']}/{total} câu đã tổng hợp sẵn (resume từ checkpoint).")

    return paths  # type: ignore[return-value]


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def mix_voiceover_track(
    clip_audio_paths: list[Path],
    storyboard: dict[str, Any],
    out_path: Path,
) -> Path:
    """
    Ghép các file audio từng câu thành 1 track duy nhất, đặt mỗi file bắt đầu
    đúng tại `output.start` của clip tương ứng trong storyboard (im lặng ở khoảng trống).
    Dùng ffmpeg `amix`/`adelay` filter.
    """
    timeline = storyboard["timeline"]
    if not timeline:
        raise ValueError(
            "mix_voiceover_track: storyboard rỗng (timeline không có câu narration nào). "
            "Có thể do toàn bộ câu bị lọc bỏ ở stage trước (không còn scene_id hợp lệ nào) — "
            "kiểm tra lại output của stage 'script'/'storyboard' trước khi chạy TTS."
        )
    total_dur = timeline[-1]["output"]["end"]

    inputs = []
    filter_parts = []
    for i, (clip, audio_path) in enumerate(zip(timeline, clip_audio_paths)):
        delay_ms = int(clip["output"]["start"] * 1000)
        inputs += ["-i", str(audio_path)]
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

    amix_inputs = "".join(f"[a{i}]" for i in range(len(timeline)))
    filter_complex = ";".join(filter_parts) + f";{amix_inputs}amix=inputs={len(timeline)}:normalize=0[aout]"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", str(total_dur + 1.0),
        str(out_path),
    ]
    run_ffmpeg_with_progress(cmd, label="tts:mix_voiceover", total_duration=total_dur)
    return out_path


def run_tts(cfg, storyboard: dict[str, Any], checkpoint_mgr=None) -> dict[str, Any]:
    """
    Entry point cho stage 'tts'. Sinh audio cho từng câu + track tổng hợp
    (voiceover.mp3) đồng bộ timestamp với storyboard.
    """
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    tts_dir = pipeline_dir / "tts_clips"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    engine = cfg.get("tts.engine", "edge-tts")
    if engine not in ("edge-tts", "viterbox", "piper", "gemini"):
        print(f"[tts] CẢNH BÁO: tts.engine = '{engine}' trong config.toml không hợp lệ "
              f"(chỉ hỗ trợ 'edge-tts', 'viterbox', 'piper' hoặc 'gemini') — dùng 'edge-tts' mặc định.")
        engine = "edge-tts"

    sentences = storyboard["timeline"]

    if engine == "viterbox":
        clip_audio_paths = _synthesize_all_viterbox(sentences, cfg, tts_dir, checkpoint_mgr=checkpoint_mgr)
    elif engine == "piper":
        clip_audio_paths = _synthesize_all_piper(sentences, cfg, tts_dir, checkpoint_mgr=checkpoint_mgr)
    elif engine == "gemini":
        clip_audio_paths = asyncio.run(
            _synthesize_all_gemini(sentences, cfg, tts_dir, checkpoint_mgr=checkpoint_mgr)
        )
    else:
        voice = cfg.get("tts.voice", "vi-VN-HoangMinhNeural")
        rate = cfg.get("tts.rate", "+0%")
        volume = cfg.get("tts.volume", "+0%")
        pitch = cfg.get("tts.pitch", "+0Hz")
        max_concurrency = cfg.get("tts.max_concurrency", 40)

        print(f"[tts] Synthesizing {len(sentences)} câu bằng edge-tts (voice={voice}, "
              f"song song tối đa {max_concurrency} luồng)...")
        clip_audio_paths = asyncio.run(
            _synthesize_all(sentences, voice, rate, volume, pitch, tts_dir,
                             checkpoint_mgr=checkpoint_mgr, max_concurrency=max_concurrency)
        )

    # Đo lại thời lượng thực tế TTS để phát hiện lệch lớn so với output_duration ước tính.
    actual_durations = []
    for clip, audio_path in zip(sentences, clip_audio_paths):
        actual = _ffprobe_duration(audio_path)
        expected = clip["output"]["end"] - clip["output"]["start"]
        actual_durations.append({
            "clip_id": clip["clip_id"],
            "expected_sec": round(expected, 2),
            "actual_tts_sec": round(actual, 2),
        })

    voiceover_path = pipeline_dir / "voiceover.mp3"
    print("[tts] Ghép track voiceover tổng hợp...")
    mix_voiceover_track(clip_audio_paths, storyboard, voiceover_path)

    result = {
        "voiceover_path": str(voiceover_path),
        "clip_audio_paths": [str(p) for p in clip_audio_paths],
        "duration_check": actual_durations,
    }

    with open(pipeline_dir / "tts_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[tts] Xong: {voiceover_path}")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("tts", result)

    return result
