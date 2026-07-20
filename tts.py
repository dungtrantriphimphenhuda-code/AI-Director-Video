"""
tts.py — sinh giọng đọc (voiceover) cho từng câu trong storyboard.

Hỗ trợ 2 engine, chọn qua `tts.engine` trong config.toml:
  - "edge-tts" (mặc định): dùng edge-tts, không cần API key, không cần GPU.
    LƯU Ý: đây là API không chính thức của Microsoft, thỉnh thoảng bị chặn/
    rớt kết nối khi chạy từ IP datacenter (vd GitHub Actions) — lỗi
    "edge_tts.exceptions.NoAudioReceived".
  - "viterbox": model TTS tiếng Việt chạy LOCAL (fine-tune từ Chatterbox,
    xem viterbox_env.py / viterbox_worker.py), không phụ thuộc dịch vụ ngoài
    nên không bị lỗi mạng như trên. Cần tải model (~3-4GB) lần đầu, chạy
    nhanh trên GPU và chậm hơn nhiều trên CPU. Chạy trong 1 venv RIÊNG
    (.viterbox_venv) vì package `viterbox` ghim version numpy/transformers
    xung đột với requirements.txt chính của project.

Mỗi câu narration được tổng hợp thành 1 file audio riêng, sau đó ghép lại
thành 1 file audio tổng (voiceover.mp3) với timestamp đặt đúng theo
`timeline[].output.start` trong storyboard.json — để đồng bộ với video render
ở bước sau.

edge-tts dùng asyncio (bọc lại bằng asyncio.run() để phần còn lại của
pipeline, vốn đồng bộ, gọi bình thường). viterbox chạy đồng bộ qua subprocess
(model AI, không hưởng lợi gì từ việc chạy song song nhiều luồng như network
I/O của edge-tts).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
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
    if engine not in ("edge-tts", "viterbox"):
        print(f"[tts] CẢNH BÁO: tts.engine = '{engine}' trong config.toml không hợp lệ "
              f"(chỉ hỗ trợ 'edge-tts' hoặc 'viterbox') — dùng 'edge-tts' mặc định.")
        engine = "edge-tts"

    sentences = storyboard["timeline"]

    if engine == "viterbox":
        clip_audio_paths = _synthesize_all_viterbox(sentences, cfg, tts_dir, checkpoint_mgr=checkpoint_mgr)
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
