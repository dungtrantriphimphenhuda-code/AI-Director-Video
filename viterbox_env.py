"""
viterbox_env.py — Tạo & quản lý venv RIÊNG cho backend TTS "viterbox".

Xem giải thích đầy đủ trong viterbox_worker.py: package `viterbox` ghim
version numpy/transformers xung đột trực tiếp với requirements.txt chính của
project, nên phải cài trong 1 virtualenv tách biệt, không đụng tới
site-packages của tiến trình Python đang chạy pipeline. Module này chỉ được
gọi khi tts.engine == "viterbox" (xem tts.py) — không ảnh hưởng gì tới các
backend TTS khác hay tới môi trường Python chính.

Hoạt động giống hệt trên máy thật / Colab / VPS Linux / GitHub Actions: chỉ
cần `python3 -m venv` chạy được (luôn có sẵn trên mọi bản Python 3 chuẩn).

LƯU Ý CHO GITHUB ACTIONS: venv này KHÔNG có GPU trên runner miễn phí (không
có CUDA) -> tự động rơi về "cpu", chậm hơn nhiều so với chạy trên máy/Colab
có GPU. Việc cài đặt (tải torch + package viterbox + model AI ~3-4GB) cũng sẽ
phải làm LẠI TỪ ĐẦU mỗi lần chạy CI vì runner bị xoá sau mỗi job -- không
phải lỗi của cơ chế cache trong module này, mà là giới hạn vốn có của runner
dùng 1 lần.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_VENV_DIRNAME = ".viterbox_venv"
_VITERBOX_GIT_URL = "git+https://github.com/iamdinhthuan/viterbox-tts.git"
_MARKER_NAME = ".install_ok"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_viterbox_env(project_root: Path) -> Path:
    """
    Tạo venv riêng (nếu chưa có) + cài package `viterbox`, trả về đường dẫn
    python interpreter của venv đó.

    Idempotent: nếu đã cài xong lần trước (marker file tồn tại), bỏ qua luôn
    không cài lại -- tránh tải lại torch/model mỗi lần chạy pipeline trên
    cùng 1 máy (quan trọng khi chạy lặp lại trên máy thật/self-hosted runner;
    trên GitHub Actions hosted runner thì venv bị xoá cùng máy ảo sau mỗi
    job nên vẫn phải cài lại mỗi lần dù có marker hay không).
    """
    venv_dir = project_root / _VENV_DIRNAME
    marker = venv_dir / _MARKER_NAME
    py = _venv_python(venv_dir)

    if marker.exists() and py.exists():
        print(f"[viterbox-env] venv riêng đã sẵn sàng tại {venv_dir}")
        return py

    if not py.exists():
        print(f"[viterbox-env] Tạo venv riêng cho Viterbox tại {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    print("[viterbox-env] Cài package 'viterbox' (+ dependencies riêng: "
          "torch/transformers/numpy phiên bản do viterbox tự ghim, KHÔNG liên "
          "quan tới version trong requirements.txt chính) — lần đầu có thể mất "
          "vài phút vì phải tải torch + model AI (~3-4GB)...")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", _VITERBOX_GIT_URL], check=True)

    marker.write_text("ok", encoding="utf-8")
    print("[viterbox-env] Cài xong.")
    return py
