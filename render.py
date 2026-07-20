"""
render.py — dựng video cuối cùng từ storyboard.json + voiceover.

Thuật toán theo đúng ref-asr-vision-pipeline.md:
  1. Dịch source.start lùi về trước theo thời lượng đọc TTS (source timestamp shift),
     để khi TTS đọc xong và tiếng gốc fade-in, nội dung nghe được đúng với lời bình.
  2. Với mỗi clip: cắt theo source range, chỉnh tốc độ (setpts/atempo) để khớp
     đúng output_duration đã tính từ trước, hạ âm lượng gốc và mix với TTS.
  3. Ghép toàn bộ clip theo thứ tự (không có khoảng trống) thành final_preview.mp4.
  4. Sinh narration_subtitle.srt từ output timeline.
  5. Validate: |video_duration - srt_end| < 0.5s. Nếu không đạt, dùng ffprobe đo
     lại thời lượng thực tế từng clip và rebuild SRT bằng cộng dồn thời lượng thật.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import srt
import datetime

from script_writer import calc_output_duration
from progress_utils import print_progress_bar, run_ffmpeg_with_progress


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def shift_source_timestamps(
    storyboard: dict[str, Any],
    chars_per_sec: float,
    words_per_sec: float = 2.5,
    safety_margin: float = 1.0,
) -> dict[str, Any]:
    """
    源时间戳前移: new_source_start = max(0, source_start - thời_lượng_đọc_ước_tính).
    source.end giữ nguyên khoảng cách gốc với source.start.

    Dùng chung công thức ước tính thời lượng đọc với script_writer.calc_output_duration
    (CJK: số ký tự / chars_per_sec; Latin: số từ / words_per_sec) trừ đi buffer,
    để nhất quán giữa lúc build storyboard và lúc render.
    """
    for clip in storyboard["timeline"]:
        # buffer_after=0 ở đây vì ta chỉ cần thời lượng ĐỌC, không cộng buffer.
        tts_dur = calc_output_duration(clip["sentence"], chars_per_sec, 0.0, 0.0, words_per_sec, safety_margin)
        src = clip["source"]
        src_span = src["end"] - src["start"]
        new_start = max(0.0, src["start"] - tts_dur)
        src["start"] = round(new_start, 3)
        src["end"] = round(new_start + src_span, 3)
    return storyboard


def _atempo_chain(speed: float) -> str:
    """atempo chỉ nhận [0.5, 2.0] mỗi lần — nối chuỗi nếu speed vượt khoảng đó."""
    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def render_clip(
    input_video: str,
    clip: dict[str, Any],
    tts_audio_path: Path | None,
    out_path: Path,
    max_speed_ratio: float = 4.0,
    min_speed_ratio: float = 0.5,
) -> None:
    """Render 1 clip: cắt theo source, chỉnh tốc độ khớp output_duration, mix TTS + tiếng gốc (duck).

    Tốc độ tua nhanh/chậm được KẸP trong [min_speed_ratio, max_speed_ratio]
    thay vì chỉ in cảnh báo rồi vẫn dùng tốc độ cực đoan như trước — trước
    đây max_speed_ratio chỉ là 1 dòng log, ffmpeg vẫn tua/kéo clip gốc theo
    đúng tỉ lệ toán học src_dur/out_dur dù có thể ra tới hàng chục lần, khiến
    clip trông "gấp" (tua nhanh vượt max) hoặc "lê thê" (kéo chậm dưới min).
    Khi kẹp tốc độ khiến video ngắn hơn out_dur (trường hợp lẽ ra phải kéo
    chậm hơn nữa), phần thiếu được bù bằng cách đông cứng khung hình cuối
    (tpad) thay vì để clip kết thúc sớm và lệch thời lượng so với storyboard.
    """
    src = clip["source"]
    out = clip["output"]
    src_dur = max(src["end"] - src["start"], 0.1)
    out_dur = max(out["end"] - out["start"], 0.1)
    natural_speed = src_dur / out_dur
    speed = min(max(natural_speed, min_speed_ratio), max_speed_ratio)

    if natural_speed > max_speed_ratio:
        print(
            f"[render] {clip['clip_id']}: tốc độ tự nhiên {natural_speed:.2f}x vượt "
            f"max_speed_ratio={max_speed_ratio} -> kẹp còn {speed:.2f}x (clip sẽ chỉ chiếu "
            "hết phần đầu của cảnh gốc trong thời lượng lời bình cho phép)."
        )
    elif natural_speed < min_speed_ratio:
        print(
            f"[render] {clip['clip_id']}: tốc độ tự nhiên {natural_speed:.2f}x dưới "
            f"min_speed_ratio={min_speed_ratio} -> kẹp còn {speed:.2f}x (bù phần thời lượng "
            "còn thiếu bằng đông cứng khung hình cuối thay vì kéo chậm quá mức)."
        )

    # Thời lượng video thật sự sau khi cắt+chỉnh tốc độ (kẹp) — có thể khác out_dur
    # nếu speed đã bị kẹp khỏi giá trị tự nhiên.
    cut_dur = src_dur / speed
    pad_needed = max(out_dur - cut_dur, 0.0)

    video_only = out_path.with_suffix(".novoice.mp4")

    if abs(speed - 1.0) < 0.01:
        vf = f"tpad=stop_mode=clone:stop_duration={pad_needed:.3f}" if pad_needed > 0.05 else None
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(src["start"]), "-t", str(src_dur),
            "-i", input_video,
            *(["-vf", vf] if vf else []),
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-af", "volume=0.15",
            "-c:a", "aac", "-b:a", "128k",
            str(video_only),
        ]
    else:
        atempo_chain = _atempo_chain(speed)
        vpts = f"setpts={1/speed:.4f}*PTS"
        vf_chain = f"{vpts},tpad=stop_mode=clone:stop_duration={pad_needed:.3f}" if pad_needed > 0.05 else vpts
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(src["start"]), "-t", str(src_dur),
            "-i", input_video,
            "-filter_complex",
            f"[0:v]{vf_chain}[v];[0:a]{atempo_chain},volume=0.15[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(video_only),
        ]
    run_ffmpeg_with_progress(cmd, label=f"render:{clip['clip_id']}:cut", total_duration=src_dur)

    if tts_audio_path is None or not Path(tts_audio_path).exists():
        video_only.rename(out_path)
        return

    # Mix: TTS ở đầu clip full volume, tiếng gốc duck 0.15 rồi fade-in 1.5s cuối clip.
    fade_start = max(out_dur - 1.5, 0.0)
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(tts_audio_path),
        "-filter_complex",
        f"[0:a]afade=t=in:st={fade_start}:d=1.5[orig];"
        f"[1:a]apad[tts];"
        f"[tts][orig]amix=inputs=2:duration=first:normalize=0[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(out_dur),
        str(out_path),
    ]
    run_ffmpeg_with_progress(cmd_mix, label=f"render:{clip['clip_id']}:mix", total_duration=out_dur)
    video_only.unlink(missing_ok=True)


def concat_clips(clip_paths: list[Path], out_path: Path, total_duration: float | None = None) -> None:
    """Ghép các clip theo thứ tự (re-encode để đảm bảo timeline chính xác, không dùng -c copy)."""
    concat_list = out_path.parent / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    run_ffmpeg_with_progress(cmd, label="render:concat_clips", total_duration=total_duration)


def build_srt(storyboard: dict[str, Any]) -> str:
    """Sinh nội dung SRT từ timeline[].sentence + timeline[].output."""
    subs = []
    for i, clip in enumerate(storyboard["timeline"], start=1):
        start = datetime.timedelta(seconds=clip["output"]["start"])
        end = datetime.timedelta(seconds=clip["output"]["end"])
        subs.append(srt.Subtitle(index=i, start=start, end=end, content=clip["sentence"]))
    return srt.compose(subs)


def rebuild_srt_from_actual_durations(storyboard: dict[str, Any], clip_paths: list[Path]) -> str:
    """
    校验规则 修复流程: dùng ffprobe đo thời lượng thực tế từng clip đã render,
    cộng dồn thuần tuý (không thêm khoảng trống) để rebuild timestamp SRT.
    """
    subs = []
    cursor = 0.0
    for i, (clip, path) in enumerate(zip(storyboard["timeline"], clip_paths), start=1):
        actual_dur = _ffprobe_duration(path)
        start = datetime.timedelta(seconds=cursor)
        end = datetime.timedelta(seconds=cursor + actual_dur)
        subs.append(srt.Subtitle(index=i, start=start, end=end, content=clip["sentence"]))
        cursor += actual_dur
    return srt.compose(subs)


def validate_alignment(video_path: Path, srt_text: str, tolerance: float = 0.5) -> dict[str, Any]:
    """字幕-视频结束时间校验: so sánh video_dur với srt_end cuối cùng."""
    video_dur = _ffprobe_duration(video_path)
    subs = list(srt.parse(srt_text))
    srt_end = subs[-1].end.total_seconds() if subs else 0.0
    diff = abs(video_dur - srt_end)

    report = {
        "video_duration_sec": round(video_dur, 3),
        "srt_end_sec": round(srt_end, 3),
        "diff_sec": round(diff, 3),
        "subtitle_count": len(subs),
        "passed": diff < tolerance,
    }
    print("=== 校验报告 / Validation report ===")
    print(f"视频时长 / video duration: {report['video_duration_sec']}s")
    print(f"字幕末尾 / srt end:        {report['srt_end_sec']}s")
    print(f"差值 / diff:              {report['diff_sec']}s")
    print(f"字幕条数 / subtitle count: {report['subtitle_count']}")
    print(f"状态 / status:            {'✓ 通过 / PASS' if report['passed'] else '✗ 失败 / FAIL'}")
    return report


def run_render(cfg, storyboard: dict[str, Any], tts_result: dict[str, Any] | None, checkpoint_mgr=None) -> dict[str, Any]:
    """Entry point cho stage 'render'. Ghi final_preview.mp4 + narration_subtitle.srt vào deliverables/."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    deliverables_dir = output_dir / "deliverables"
    clips_dir = deliverables_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    input_video = storyboard["task"]["input_video"]
    chars_per_sec = cfg.get("processing.chars_per_sec", 4.0)
    words_per_sec = cfg.get("processing.words_per_sec", 2.5)
    safety_margin = cfg.get("processing.speech_safety_margin", 1.0)
    max_speed_ratio = cfg.get("processing.max_speed_ratio", 4.0)
    min_speed_ratio = cfg.get("processing.min_speed_ratio", 0.5)

    print("[render] Dịch source timestamp lùi theo thời lượng TTS...")
    storyboard = shift_source_timestamps(storyboard, chars_per_sec, words_per_sec, safety_margin)
    with open(pipeline_dir / "storyboard.json", "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)

    tts_audio_map: dict[str, Path] = {}
    if tts_result:
        for clip, path in zip(storyboard["timeline"], tts_result["clip_audio_paths"]):
            tts_audio_map[clip["clip_id"]] = Path(path)

    clip_paths = []
    total_clips = len(storyboard["timeline"])

    # Resume: bỏ qua các clip đã render xong ở lần chạy trước (checkpoint đã
    # ghi VÀ file .mp4 tương ứng vẫn còn trên đĩa). Trước đây stage này chỉ
    # GHI checkpoint mà không bao giờ đọc lại, nên bị ngắt giữa chừng sẽ luôn
    # render lại từ đầu toàn bộ clip dù dữ liệu cũ vẫn còn nguyên.
    already_done = checkpoint_mgr.list_micro_done("render") if checkpoint_mgr is not None else set()
    if already_done:
        print(f"[render] Tìm thấy {len(already_done)} clip đã render ở lần chạy trước, sẽ bỏ qua nếu file còn nguyên.")

    for idx, clip in enumerate(storyboard["timeline"], start=1):
        out_path = clips_dir / f"{clip['clip_id']}.mp4"

        if clip["clip_id"] in already_done and out_path.exists():
            clip_paths.append(out_path)
            print_progress_bar(idx, total_clips, prefix="[render] clips (resumed)", suffix=clip["clip_id"])
            continue

        render_clip(input_video, clip, tts_audio_map.get(clip["clip_id"]), out_path, max_speed_ratio, min_speed_ratio)
        clip_paths.append(out_path)
        print_progress_bar(idx, total_clips, prefix="[render] clips", suffix=clip["clip_id"])

        # Micro-checkpoint per clip. LUÔN lưu (không throttle theo
        # micro_interval) — clip vừa render tốn thời gian ffmpeg thật, nếu
        # throttle và crash giữa chừng, clip đã render xong nhưng chưa tới
        # mốc lưu sẽ bị RENDER LẠI không cần thiết khi resume (giống lỗi đã
        # sửa ở nhánh API của vision.py). Ghi JSON cục bộ rất rẻ; việc
        # throttle tần suất SYNC LÊN CLOUD đã được xử lý riêng trong
        # CheckpointManager.save_micro().
        if checkpoint_mgr is not None:
            checkpoint_mgr.save_micro("render", clip["clip_id"], {
                "clip_id": clip["clip_id"],
                "clip_path": str(out_path),
                "progress": f"{idx}/{total_clips}",
            })

    if checkpoint_mgr is not None:
        # Lưới an toàn cuối stage: đảm bảo mọi clip đã render trong lần
        # chạy này thực sự lên cloud trước khi ghép final_preview.mp4.
        checkpoint_mgr.flush_pending_syncs()

    final_path = deliverables_dir / "final_preview.mp4"
    print("[render] Ghép các clip thành final_preview.mp4...")
    total_duration = storyboard["timeline"][-1]["output"]["end"] if storyboard["timeline"] else None
    concat_clips(clip_paths, final_path, total_duration=total_duration)

    srt_text = build_srt(storyboard)
    report = validate_alignment(final_path, srt_text)

    if not report["passed"]:
        print("[render] Validate thất bại, rebuild SRT từ thời lượng clip thực tế...")
        srt_text = rebuild_srt_from_actual_durations(storyboard, clip_paths)
        report = validate_alignment(final_path, srt_text)

    srt_path = deliverables_dir / "narration_subtitle.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    with open(pipeline_dir / "render_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    result = {
        "final_preview_path": str(final_path),
        "srt_path": str(srt_path),
        "validation_report": report,
    }

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("render", result)

    print(f"[render] Xong: {final_path}")
    return result
