#!/usr/bin/env python3
"""
viterbox_worker.py — Worker CHẠY TRONG VENV RIÊNG (.viterbox_venv), tách biệt
hoàn toàn khỏi môi trường Python chính của project.

LÝ DO CẦN VENV RIÊNG: package `viterbox`
(https://github.com/iamdinhthuan/viterbox-tts) ghim CỨNG một số version xung
đột trực tiếp với requirements.txt của project chính:
  - viterbox cần numpy>=1.24.0,<1.26.0   | project chính cần numpy>=1.26.0
  - viterbox cần transformers==4.46.3    | project chính cần transformers>=4.57.0,<5.0.0
Cài chung 1 venv sẽ làm hỏng ÂM THẦM 1 trong 2 bên (pip resolver hạ/nâng
version mà không báo lỗi rõ ràng). Vì vậy script này KHÔNG được import bất kỳ
module nào của project chính (config.py, progress_utils.py, ...) — chỉ dùng
thư viện chuẩn + package viterbox, để chạy độc lập dưới venv riêng
(xem viterbox_env.py — nơi tạo venv này và gọi script này qua subprocess).

Giao thức (đơn giản, dễ debug qua log CI): script in ra STDOUT, mỗi dòng 1 sự
kiện, để process cha (tts.py) đọc realtime và cập nhật checkpoint/progress bar:
    "VITERBOX_MODEL_LOADED"                  -- sau khi load model xong
    "VITERBOX_OK <clip_id>"                  -- 1 câu tổng hợp xong
    "VITERBOX_ERR <clip_id> <thông báo lỗi>" -- 1 câu lỗi hẳn sau khi đã retry;
                                                 worker DỪNG LUÔN sau dòng này
                                                 (không tổng hợp tiếp các câu
                                                 sau) -- process cha coi đây là
                                                 lỗi stage, checkpoint đã lưu
                                                 mọi câu THÀNH CÔNG trước đó nên
                                                 lần chạy sau sẽ resume tiếp từ
                                                 đúng câu bị lỗi.
    "VITERBOX_DONE"                          -- toàn bộ job xong, không lỗi

Job JSON (đường dẫn truyền qua sys.argv[1]):
{
  "device": "cuda" | "cpu",
  "language": "vi",
  "audio_prompt": "path/to/ref.wav" | null,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "temperature": 0.8,
  "items": [{"clip_id": "...", "text": "...", "out_path": "..."}, ...]
}
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    if len(sys.argv) != 2:
        print("VITERBOX_ERR - Thiếu đường dẫn file job JSON.", flush=True)
        return 1

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        job = json.load(f)

    device = job.get("device", "cpu")

    # Checkpoint gốc (s3gen.pt) của package `viterbox` được lưu sẵn cho CUDA.
    # Trên máy KHÔNG có GPU (CPU-only — ví dụ GitHub Actions runner miễn phí,
    # hoặc laptop không có card NVIDIA), `torch.load()` mặc định trong code
    # gốc của viterbox không truyền map_location -> PyTorch cố nạp thẳng lên
    # CUDA và crash với lỗi "Attempting to deserialize object on a CUDA
    # device but torch.cuda.is_available() is False", bất kể chạy ở đâu.
    # Ta không sửa trực tiếp package viterbox (vì nó bị cài lại từ đầu mỗi
    # lần), mà "monkeypatch" torch.load ngay tại đây để luôn ép về CPU khi
    # không có GPU khả dụng, trước khi import/gọi tới viterbox.
    try:
        import torch
    except ImportError as e:
        print(f"VITERBOX_ERR - Chưa cài 'torch' trong venv này: {e}", flush=True)
        return 1

    if device == "cpu" or not torch.cuda.is_available():
        _orig_torch_load = torch.load

        def _torch_load_force_cpu(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_force_cpu
        print("[viterbox-worker] Không có GPU khả dụng -> ép torch.load map về CPU.",
              file=sys.stderr, flush=True)

    try:
        from viterbox import Viterbox
    except ImportError as e:
        print(f"VITERBOX_ERR - Chưa cài package 'viterbox' trong venv này: {e}", flush=True)
        return 1

    print(f"[viterbox-worker] Đang load model Viterbox (device={device})...",
          file=sys.stderr, flush=True)
    tts = Viterbox.from_pretrained(device)
    print("VITERBOX_MODEL_LOADED", flush=True)

    audio_prompt = job.get("audio_prompt") or None
    language = job.get("language", "vi")
    exaggeration = job.get("exaggeration", 0.5)
    cfg_weight = job.get("cfg_weight", 0.5)
    temperature = job.get("temperature", 0.8)

    for item in job["items"]:
        clip_id = item["clip_id"]
        text = item["text"]
        out_path = item["out_path"]

        gen_kwargs = dict(
            text=text, language=language,
            exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature,
        )
        if audio_prompt:
            gen_kwargs["audio_prompt"] = audio_prompt

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                audio = tts.generate(**gen_kwargs)
                tts.save_audio(audio, out_path)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 — worker cô lập, bắt hết để retry + báo rõ lỗi
                last_err = e
                print(f"[viterbox-worker] '{clip_id}' lỗi lần {attempt}/3: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(1.0 * attempt)

        if last_err is not None:
            print(f"VITERBOX_ERR {clip_id} {last_err}", flush=True)
            return 1
        print(f"VITERBOX_OK {clip_id}", flush=True)

    print("VITERBOX_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
