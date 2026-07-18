"""
tts.py — sinh giọng đọc (voiceover) cho từng câu trong storyboard bằng edge-tts.

edge-tts không cần API key. Mỗi câu narration được tổng hợp thành 1 file .mp3
riêng, sau đó ghép lại thành 1 file audio tổng (voiceover.mp3) với timestamp
đặt đúng theo `timeline[].output.start` trong storyboard.json — để đồng bộ
với video render ở bước sau.

Vì đây là module I/O bất đồng bộ (edge-tts dùng asyncio), ta bọc lại bằng
asyncio.run() để module còn lại (đồng bộ) có thể gọi bình thường.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import edge_tts

from progress_utils import print_progress_bar, run_ffmpeg_with_progress


async def _synthesize_one(text: str, voice: str, rate: str, volume: str, pitch: str, out_path: Path) -> None:
    """Sinh 1 file audio cho 1 câu narration bằng edge-tts."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(str(out_path))


async def _synthesize_all(
    sentences: list[dict[str, Any]],
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    out_dir: Path,
    checkpoint_mgr=None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = len(sentences)

    # --- Resume: bỏ qua câu đã tổng hợp xong ở lần chạy trước ---
    # Đây là bước dễ bị ngắt nhất (gọi mạng tuần tự cho từng câu), nên phải
    # checkpoint NGAY sau mỗi câu, không phải đợi xong hết vòng lặp mới ghi.
    done_ids: set[str] = set()
    if checkpoint_mgr is not None:
        done_ids = checkpoint_mgr.list_micro_done("tts")

    n_skipped = 0
    for i, sent in enumerate(sentences, start=1):
        clip_id = sent["clip_id"]
        out_path = out_dir / f"{clip_id}.mp3"

        if clip_id in done_ids and out_path.exists():
            # Đã tổng hợp xong ở lần chạy trước và file audio vẫn còn -> bỏ qua.
            paths.append(out_path)
            n_skipped += 1
            print_progress_bar(i, total, prefix="[tts] synthesizing", suffix=f"{clip_id} (cached)")
            continue

        # edge-tts qua mạng — chạy tuần tự để tránh bị rate-limit/timeout hàng loạt.
        await _synthesize_one(sent["sentence"], voice, rate, volume, pitch, out_path)
        paths.append(out_path)
        print_progress_bar(i, total, prefix="[tts] synthesizing", suffix=clip_id)

        # Micro-checkpoint NGAY sau khi câu này xong — đây chính là chỗ dễ bị
        # ngắt mạng/treo nhất trong cả pipeline, nên không thể đợi tổng hợp
        # xong hết mới ghi checkpoint (nếu vậy bị ngắt giữa chừng sẽ mất sạch,
        # y như không có checkpoint).
        if checkpoint_mgr is not None:
            checkpoint_mgr.save_micro("tts", clip_id, {
                "clip_id": clip_id,
                "audio_path": str(out_path),
            })

    if checkpoint_mgr is not None and sentences:
        # Lưới an toàn cuối stage: đảm bảo MỌI câu đã tổng hợp trong lần
        # chạy này thực sự lên cloud (không chỉ câu cuối như trước).
        checkpoint_mgr.flush_pending_syncs()

    if n_skipped:
        print(f"[tts] Bỏ qua {n_skipped}/{total} câu đã tổng hợp sẵn (resume từ checkpoint).")

    return paths


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
    if engine != "edge-tts":
        print(f"[tts] CẢNH BÁO: tts.engine = '{engine}' trong config.toml, nhưng hiện tại "
              f"chỉ có engine 'edge-tts' được cài đặt thật sự — vẫn dùng edge-tts, "
              f"giá trị '{engine}' không có tác dụng.")

    voice = cfg.get("tts.voice", "vi-VN-HoangMinhNeural")
    rate = cfg.get("tts.rate", "+0%")
    volume = cfg.get("tts.volume", "+0%")
    pitch = cfg.get("tts.pitch", "+0Hz")

    sentences = storyboard["timeline"]
    print(f"[tts] Synthesizing {len(sentences)} câu bằng edge-tts (voice={voice})...")
    clip_audio_paths = asyncio.run(
        _synthesize_all(sentences, voice, rate, volume, pitch, tts_dir, checkpoint_mgr=checkpoint_mgr)
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
