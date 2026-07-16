#!/usr/bin/env python3
"""
setup.py — script setup 1 lần, chạy được trên Google Colab, Linux, macOS, Windows.

Chạy độc lập, KHÔNG cần tham số dòng lệnh:

    python3 setup.py      (Linux/macOS)
    python setup.py       (Windows)

Việc script này làm:
  1. In thông tin GPU / RAM / Python / hệ điều hành.
  2. Cài FFmpeg (tự phát hiện trình quản lý gói theo OS) + gói hệ thống.
  3. Cài mọi package Python từ requirements.txt (bổ sung nếu thiếu).
  4. Hỏi người dùng nhập: Cerebras API key, HF token (tuỳ chọn), giọng Edge TTS,
     đường dẫn video đầu vào, thư mục output, tên model Qwen3-VL, endpoint Cerebras.
  5. Ghi ra config.toml từ các câu trả lời đó.
  6. Tải trước faster-whisper (small) và Qwen3-VL-4B-Instruct vào cache.
  7. Hiển thị hướng dẫn chạy tiếp theo (python run.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from platform_utils import print_system_info, ensure_ffmpeg
from progress_utils import run_subprocess


def _run(cmd: list[str], check: bool = True, label: str | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    if label:
        return run_subprocess(cmd, label=label, heartbeat_interval=5.0, check=check)
    return subprocess.run(cmd, check=check)


def install_system_packages() -> None:
    print("=" * 70)
    print("CÀI ĐẶT GÓI HỆ THỐNG / SYSTEM PACKAGES")
    print("=" * 70)
    try:
        ensure_ffmpeg()
        print("FFmpeg: OK.")
    except RuntimeError as e:
        print(f"CẢNH BÁO: {e}")
    print()


def install_python_packages() -> None:
    print("=" * 70)
    print("CÀI ĐẶT PYTHON PACKAGES")
    print("=" * 70)
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print(f"CẢNH BÁO: không tìm thấy {req_file}, bỏ qua bước cài đặt.")
        return
    print("Đang cài đặt (không có output chi tiết vì dùng pip -q; sẽ có heartbeat mỗi 5s "
          "để biết chưa bị treo, có thể mất vài phút tuỳ mạng)...")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
         label="setup:pip_install_requirements")
    print("Đã cài xong Python packages.\n")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def collect_user_config() -> dict:
    print("=" * 70)
    print("NHẬP CẤU HÌNH / CONFIGURATION INPUT")
    print("=" * 70)

    cerebras_api_key = ask("Cerebras API key (bắt buộc để sinh kịch bản)")
    while not cerebras_api_key:
        print("Cerebras API key không được để trống.")
        cerebras_api_key = ask("Cerebras API key")

    cerebras_endpoint = ask("Cerebras base URL", "https://api.cerebras.ai/v1")
    hf_token = ask("Hugging Face token (bỏ trống nếu model public / đã cache)", "")
    tts_voice = ask("Giọng Edge TTS (vd: vi-VN-HoangMinhNeural, en-US-JennyNeural)", "vi-VN-HoangMinhNeural")
    input_video = ask("Đường dẫn video đầu vào", "./input/source.mp4")
    output_dir = ask("Thư mục output", "./output")
    vision_model = ask("Tên model Qwen3-VL trên Hugging Face", "Qwen/Qwen3-VL-4B-Instruct")

    print("\n-- Filebase cloud sync (tuỳ chọn, bỏ trống access key để tắt) --")
    filebase_access_key = ask("Filebase access key (bỏ trống nếu không dùng cloud sync)", "")
    filebase_secret_key = ""
    filebase_bucket = "ai-director-video"
    if filebase_access_key:
        filebase_secret_key = ask("Filebase secret key")
        filebase_bucket = ask(
            "Tên bucket Filebase (phải DUY NHẤT TOÀN CỤC, giống S3)", "ai-director-video"
        )

    return {
        "cerebras_api_key": cerebras_api_key,
        "cerebras_endpoint": cerebras_endpoint,
        "hf_token": hf_token,
        "tts_voice": tts_voice,
        "input_video": input_video,
        "output_dir": output_dir,
        "vision_model": vision_model,
        "filebase_access_key": filebase_access_key,
        "filebase_secret_key": filebase_secret_key,
        "filebase_bucket": filebase_bucket,
    }


def write_config_toml(answers: dict, out_path: Path = Path("config.toml")) -> None:
    content = f'''# Tự động sinh bởi setup.py

[api]
cerebras_api_key = "{answers['cerebras_api_key']}"
cerebras_base_url = "{answers['cerebras_endpoint']}"
cerebras_model = "gemma-4-31b"
cerebras_max_tokens = 8000
cerebras_temperature = 0.8
hf_token = "{answers['hf_token']}"

[tts]
engine = "edge-tts"
voice = "{answers['tts_voice']}"
rate = "+0%"
volume = "+0%"
pitch = "+0Hz"

[processing]
asr_model_size = "small"
asr_device = "auto"
asr_compute_type = "auto"
asr_language = ""
vision_model_name = "{answers['vision_model']}"
vision_device = "auto"
vision_dtype = "float16"
vision_max_new_tokens = 512
vision_frames_per_scene = 3
scene_threshold = 27.0
min_scene_len_sec = 1.0
narration_pov = "third_person"
content_type = "movie"
genre = "drama"
target_duration_sec = 180
chars_per_sec = 4.0
buffer_after_speech = 0.1
min_clip_duration = 1.0
max_speed_ratio = 4.0
micro_checkpoint_interval = 1
auto_sync_cloud = true

[paths]
input_video = "{answers['input_video']}"
output_dir = "{answers['output_dir']}"
checkpoint_dir = "./checkpoints"
model_cache_dir = "./model_cache"
projects_dir = "./projects"

[filebase]
access_key = "{answers['filebase_access_key']}"
secret_key = "{answers['filebase_secret_key']}"
bucket_name = "{answers['filebase_bucket']}"
endpoint_url = "https://s3.filebase.com"
enabled = {str(bool(answers['filebase_access_key'])).lower()}

[project]
auto_project_scan = true
show_project_menu_on_start = true
auto_save_interval = 0
cloud_sync_retries = 3
'''
    out_path.write_text(content, encoding="utf-8")
    print(f"Đã ghi cấu hình vào {out_path.resolve()}\n")


def predownload_models(answers: dict) -> None:
    print("=" * 70)
    print("TẢI TRƯỚC MODEL / PRE-DOWNLOADING MODELS")
    print("=" * 70)

    cache_dir = Path("./model_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if answers["hf_token"]:
        os.environ["HF_TOKEN"] = answers["hf_token"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = answers["hf_token"]

    # Không bọc Heartbeat quanh các lệnh tải bên dưới: huggingface_hub đã tự in
    # thanh tiến độ tải (%, MB/s, ETA) qua tqdm. Bọc thêm sẽ khiến 2 log cùng
    # ghi \r đè lên nhau, chỉ còn thấy "vẫn đang chạy..." thay vì log tải thật.
    print("[1/2] Tải faster-whisper 'small'... (log tải % của huggingface_hub hiện bên dưới)")
    try:
        from faster_whisper import WhisperModel
        WhisperModel("small", device="cpu", compute_type="int8", download_root=str(cache_dir))
        print("  -> OK.")
    except Exception as e:
        print(f"  -> CẢNH BÁO: tải faster-whisper thất bại ({e}). Sẽ thử lại khi chạy run.py.")

    print(f"[2/2] Tải {answers['vision_model']}... (log tải % của huggingface_hub hiện bên dưới)")
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            answers["vision_model"], cache_dir=str(cache_dir), trust_remote_code=True
        )
        print("  -> Đã tải processor. Trọng số model sẽ tải khi chạy stage vision lần đầu"
              " (file lớn, tải ngầm qua transformers cache).")
    except Exception as e:
        print(f"  -> CẢNH BÁO: tải processor thất bại ({e}). Sẽ thử lại khi chạy run.py.")
    print()


def print_next_steps(answers: dict) -> None:
    print("=" * 70)
    print("HOÀN TẤT SETUP / SETUP COMPLETE")
    print("=" * 70)
    print(f"""
Cấu hình đã được ghi vào config.toml.

Bước tiếp theo:
  1. Upload video của bạn vào: {answers['input_video']}
     (hoặc sửa lại `paths.input_video` trong config.toml)
  2. Chạy pipeline:
       python run.py   (Windows: python run.py)
  3. Kết quả sẽ nằm trong: {answers['output_dir']}/deliverables/
       - final_preview.mp4
       - narration_subtitle.srt

Nếu tiến trình bị ngắt giữa chừng (mất kết nối Colab, tắt máy...), chỉ cần chạy
lại `python run.py` — pipeline sẽ tự động resume từ checkpoint gần nhất trong
./checkpoints/.
""")


def main() -> None:
    print_system_info()
    install_system_packages()
    install_python_packages()
    answers = collect_user_config()
    write_config_toml(answers)
    predownload_models(answers)
    print_next_steps(answers)


if __name__ == "__main__":
    main()
