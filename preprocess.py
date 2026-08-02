"""
preprocess.py — bước tiền xử lý video:
  1. Probe thông tin video (thời lượng, fps, độ phân giải) bằng ffprobe.
  2. Tách audio ra file .wav (dùng cho ASR).
  3. Phát hiện scene/shot boundaries bằng PySceneDetect.
  4. Trích khung hình đại diện (keyframes) cho mỗi scene bằng OpenCV.

Không đọc os.getenv — mọi tham số (ngưỡng scene, thư mục output...) lấy từ config.toml
qua đối tượng Config được truyền vào.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from progress_utils import print_progress_bar, run_ffmpeg_with_progress, Heartbeat


def _resolve_input_video(cfg) -> Path:
    """
    Trả về Path video đầu vào hợp lệ. Nếu đường dẫn trong config.toml
    (paths.input_video) không tồn tại và đang chạy tương tác (có TTY), hỏi
    người dùng nhập lại đường dẫn thay vì raise FileNotFoundError ngay —
    lặp lại cho tới khi nhập đúng file tồn tại hoặc bỏ trống để huỷ.

    Nếu chạy non-interactive (không có TTY, vd: chạy nền/CI), không thể hỏi
    nên vẫn raise FileNotFoundError như cũ kèm hướng dẫn sửa config.toml.
    """
    video_path = cfg.resolve_path("paths.input_video")
    if video_path.exists():
        return video_path

    interactive = sys.stdin.isatty()
    if not interactive:
        raise FileNotFoundError(
            f"Không tìm thấy video đầu vào: {video_path}\n"
            f"Đang chạy non-interactive nên không thể hỏi đường dẫn — hãy đặt "
            f"đúng file vào đường dẫn trên, hoặc sửa paths.input_video trong "
            f"config.toml rồi chạy lại."
        )

    print(f"[preprocess] Không tìm thấy video tại: {video_path}")
    while True:
        entered = input(
            "Nhập đường dẫn video đầu vào (Enter để huỷ pipeline): "
        ).strip()
        if not entered:
            raise FileNotFoundError(
                "Đã huỷ: không có video đầu vào hợp lệ."
            )
        candidate = Path(entered).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.is_file():
            cfg.set("paths.input_video", str(candidate))
            print(f"[preprocess] Dùng video: {candidate}")
            print(
                f"[preprocess] Gợi ý: sửa paths.input_video = \"{candidate}\" "
                f"trong config.toml để không phải nhập lại ở lần chạy sau."
            )
            return candidate
        print(f"[preprocess] Không tìm thấy file: {candidate}. Thử lại.")


def probe_video(video_path: Path) -> dict[str, Any]:
    """Dùng ffprobe để lấy metadata cơ bản của video (duration, fps, resolution)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info.get("streams", [{}])[0]
    fmt = info.get("format", {})

    duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    # r_frame_rate thường ở dạng "30000/1001"
    fps = 0.0
    rfr = stream.get("r_frame_rate", "0/1")
    if "/" in rfr:
        num, den = rfr.split("/")
        den = float(den) if float(den) != 0 else 1.0
        fps = float(num) / den

    return {
        "path": str(video_path),
        "duration_sec": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
    }


def extract_audio(
    video_path: Path, out_wav_path: Path, sample_rate: int = 16000, duration_sec: float | None = None,
) -> Path:
    """Tách audio track thành .wav mono 16kHz (định dạng chuẩn cho faster-whisper)."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(out_wav_path),
    ]
    run_ffmpeg_with_progress(cmd, label="preprocess:extract_audio", total_duration=duration_sec)
    return out_wav_path


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.0,
) -> list[dict[str, float]]:
    """
    Phát hiện ranh giới scene bằng PySceneDetect (ContentDetector).
    Trả về danh sách {"scene_id", "start", "end"} theo giây.
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    fps = video.frame_rate or 25.0
    min_len_frames = max(1, int(min_scene_len_sec * fps))
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len_frames))
    try:
        import tqdm  # noqa: F401
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    if has_tqdm:
        # tqdm có sẵn -> PySceneDetect tự in thanh % tiến độ gốc, không bọc Heartbeat
        # (bọc thêm sẽ khiến 2 log cùng ghi \r đè lên nhau, làm log gốc bị rối).
        scene_manager.detect_scenes(video=video, show_progress=True)
    else:
        # Không có tqdm -> PySceneDetect không in gì cả, dùng Heartbeat để báo còn sống.
        with Heartbeat("preprocess:detect_scenes", interval=5.0):
            scene_manager.detect_scenes(video=video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        scenes.append({
            "scene_id": f"scene_{idx:04d}",
            "start": round(start_tc.get_seconds(), 3),
            "end": round(end_tc.get_seconds(), 3),
        })

    # Fallback: nếu không phát hiện được scene nào (video quá ngắn/đồng nhất),
    # coi toàn bộ video là một scene duy nhất.
    if not scenes:
        info = probe_video(video_path)
        scenes = [{"scene_id": "scene_0000", "start": 0.0, "end": info["duration_sec"]}]

    return scenes


def _resize_for_vision(frame, max_side: int = 768):
    """
    Thu nhỏ frame để cạnh dài nhất <= max_side, giữ nguyên tỉ lệ khung hình.
    Lý do: VLM (Qwen3-VL) dùng dynamic-resolution -> số vision token tỉ lệ
    thuận với số pixel ảnh đưa vào. Ảnh full-res 1080p/4K khiến 1 lần
    generate() cần cấp phát nhiều GiB VRAM dù batch_size=1 (đây là nguyên
    nhân gốc gây OOM, không phải do batch_size). Giới hạn 768px vẫn đủ để
    model đọc bố cục/cảnh/nhân vật/chữ lớn, không ảnh hưởng đáng kể chất
    lượng phân tích scene. Không upscale ảnh vốn đã nhỏ hơn max_side.
    """
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_keyframes(
    video_path: Path,
    scenes: list[dict[str, float]],
    out_dir: Path,
    frames_per_scene: int = 3,
    max_side: int = 768,
    jpeg_quality: int = 90,
) -> dict[str, list[str]]:
    """
    Với mỗi scene, trích ra `frames_per_scene` khung hình đại diện
    (đầu / giữa / cuối), resize cạnh dài nhất về `max_side` px (giữ tỉ lệ)
    rồi lưu thành ảnh .jpg trong out_dir. Trả về map scene_id -> [đường dẫn ảnh].

    Resize ngay khi trích (thay vì để full-res rồi resize lúc infer) vừa
    giảm VRAM cần cho vision model, vừa giảm dung lượng đĩa/I-O, vừa giúp
    generate() nhanh hơn đáng kể vì ít vision token hơn hẳn.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    keyframes: dict[str, list[str]] = {}
    total_scenes = len(scenes)
    try:
        for scene_idx, scene in enumerate(scenes, start=1):
            scene_id = scene["scene_id"]
            start, end = scene["start"], scene["end"]
            if frames_per_scene <= 1:
                timestamps = [(start + end) / 2]
            else:
                timestamps = [
                    start + (end - start) * i / (frames_per_scene - 1)
                    for i in range(frames_per_scene)
                ]

            paths = []
            for i, ts in enumerate(timestamps):
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = _resize_for_vision(frame, max_side=max_side)
                out_path = out_dir / f"{scene_id}_{i}.jpg"
                cv2.imwrite(str(out_path), frame, encode_params)
                paths.append(str(out_path))
            keyframes[scene_id] = paths
            print_progress_bar(scene_idx, total_scenes, prefix="[preprocess] keyframes", suffix=scene_id)
    finally:
        cap.release()

    return keyframes


def extract_scene_clips(
    video_path: Path,
    scenes: list[dict[str, float]],
    out_dir: Path,
    max_clip_seconds: float = 60.0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Với mỗi scene, cắt ra 1 hoặc nhiều clip .mp4 bằng ffmpeg (qua
    run_ffmpeg_with_progress đã có sẵn ở preprocess_utils, KHÔNG tự viết
    subprocess.run trần, để giữ nhất quán heartbeat/log với phần preprocess
    còn lại). Nếu scene dài hơn max_clip_seconds, chia thành nhiều sub-clip
    liên tiếp không chồng lấn, mỗi sub-clip <= max_clip_seconds — cần thiết
    vì Gemma 4 chỉ nhận video tối đa 60s/lần, không tự chunk.

    CHỈ dùng cho backend vision "gemma4_video" — không lãng phí thời gian
    ffmpeg nếu backend khác (đã được kiểm tra ở run_preprocess trước khi
    gọi hàm này).

    Re-encode nhẹ (libx264 + aac) thay vì `-c copy` (stream copy): stream
    copy nhanh hơn nhiều nhưng cắt lệch keyframe H.264 -> clip có thể hỏng/
    lệch vài frame đầu so với mốc `start` yêu cầu. Vì Gemma 4 cần đối chiếu
    thời gian trong clip với phụ đề (đã cắt theo đúng [start, end] tuyệt
    đối), độ chính xác mốc thời gian quan trọng hơn tốc độ cắt ở đây — ĐỪNG
    "tối ưu" lại thành `-c copy`.

    Trả về: scene_id -> [{"clip_path", "start", "end", "sub_index"}, ...]
    (1 phần tử nếu scene ngắn hơn max_clip_seconds, nhiều phần tử nếu bị chia).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    clips_by_scene: dict[str, list[dict[str, Any]]] = {}
    total_scenes = len(scenes)

    for scene_idx, scene in enumerate(scenes, start=1):
        scene_id = scene["scene_id"]
        start, end = scene["start"], scene["end"]
        duration = max(0.0, end - start)

        if duration <= max_clip_seconds:
            windows = [(start, end)]
        else:
            # Chia thành các sub-clip liên tiếp, không chồng lấn, mỗi
            # sub-clip <= max_clip_seconds. Sub-clip cuối có thể ngắn hơn.
            windows = []
            t = start
            while t < end:
                t_end = min(t + max_clip_seconds, end)
                windows.append((t, t_end))
                t = t_end

        entries = []
        for sub_index, (w_start, w_end) in enumerate(windows):
            clip_path = out_dir / f"{scene_id}_{sub_index}.mp4"
            if not clip_path.exists():
                # Resume: nếu clip đã tồn tại trên đĩa từ lần chạy trước
                # (vd crash giữa preprocess), bỏ qua không cắt lại.
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(w_start),
                    "-to", str(w_end),
                    "-i", str(video_path),
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-c:a", "aac",
                    str(clip_path),
                ]
                run_ffmpeg_with_progress(
                    cmd,
                    label=f"preprocess:extract_scene_clip:{scene_id}_{sub_index}",
                    total_duration=w_end - w_start,
                )
            entries.append({
                "clip_path": str(clip_path),
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "sub_index": sub_index,
            })

        clips_by_scene[scene_id] = entries
        print_progress_bar(scene_idx, total_scenes, prefix="[preprocess] scene_clips", suffix=scene_id)

    return clips_by_scene


def run_preprocess(cfg, checkpoint_mgr=None) -> dict[str, Any]:
    """
    Entry point cho stage 'preprocess'. Đọc mọi tham số từ cfg (Config).
    Trả về dict: {video_info, audio_path, scenes, keyframes_dir, keyframes}
    """
    video_path = _resolve_input_video(cfg)
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] (1/4) Probing video: {video_path}")
    video_info = probe_video(video_path)

    print("[preprocess] (2/4) Extracting audio...")
    audio_path = extract_audio(video_path, pipeline_dir / "audio.wav", duration_sec=video_info["duration_sec"])

    threshold = cfg.get("processing.scene_threshold", 27.0)
    min_len = cfg.get("processing.min_scene_len_sec", 1.0)
    print(f"[preprocess] (3/4) Detecting scenes (threshold={threshold})...")
    scenes = detect_scenes(video_path, threshold=threshold, min_scene_len_sec=min_len)
    print(f"[preprocess]      -> Tìm thấy {len(scenes)} scene.")

    frames_per_scene = cfg.get("processing.vision_frames_per_scene", 3)
    keyframe_max_side = cfg.get("processing.keyframe_max_side", 768)
    keyframe_jpeg_quality = cfg.get("processing.keyframe_jpeg_quality", 90)
    keyframes_dir = pipeline_dir / "keyframes"
    print(f"[preprocess] (4/4) Extracting keyframes ({frames_per_scene}/scene, {len(scenes)} scene, "
          f"max_side={keyframe_max_side}px)...")
    keyframes = extract_keyframes(
        video_path, scenes, keyframes_dir, frames_per_scene,
        max_side=keyframe_max_side, jpeg_quality=keyframe_jpeg_quality,
    )

    # scene_clips: CHỈ cắt khi vision_backend == "gemma4_video" — các backend
    # khác chỉ cần keyframes (ảnh), cắt clip .mp4 tốn thời gian ffmpeg vô ích
    # nếu không dùng tới.
    scene_clips: dict[str, list[dict[str, Any]]] = {}
    if cfg.get("processing.vision_backend", "local") == "gemma4_video":
        max_clip_seconds = cfg.get("processing.gemma4_max_clip_seconds", 60.0)
        clips_dir = pipeline_dir / "scene_clips"
        print(f"[preprocess] (4b/4) Extracting scene clips cho gemma4_video "
              f"(max {max_clip_seconds}s/clip, {len(scenes)} scene)...")
        scene_clips = extract_scene_clips(
            video_path, scenes, clips_dir, max_clip_seconds=max_clip_seconds,
        )

    # Micro-checkpoint: save after each keyframe batch
    if checkpoint_mgr is not None:
        micro_interval = max(1, cfg.get("processing.micro_checkpoint_interval", 1))
        for i in range(0, len(scenes), micro_interval):
            chunk = scenes[i:i + micro_interval]
            chunk_keyframes = {s["scene_id"]: keyframes.get(s["scene_id"], []) for s in chunk}
            checkpoint_mgr.save_micro("preprocess", f"keyframes_{i}", {
                "scenes_chunk": chunk,
                "keyframes_chunk": chunk_keyframes,
            })
        # Lưới an toàn cuối stage: đảm bảo mọi keyframe checkpoint đã tạo
        # trong lần chạy này thực sự lên cloud.
        checkpoint_mgr.flush_pending_syncs()

    with open(pipeline_dir / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    result = {
        "video_info": video_info,
        "audio_path": str(audio_path),
        "scenes": scenes,
        "keyframes_dir": str(keyframes_dir),
        "keyframes": keyframes,
        # "scene_clips": chỉ có nội dung khi vision_backend == "gemma4_video"
        # (dict rỗng {} với backend khác) — giữ nguyên "keyframes" để không
        # phá các backend cũ vốn chỉ đọc field này.
        "scene_clips": scene_clips,
    }

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("preprocess", result)

    return result
