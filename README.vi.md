<div align="center">

# 🎬 AI Director Video Commentary

**Pipeline AI trọn gói: biến bất kỳ video nguồn nào thành video bình luận dẫn chuyện — AI phân tích từng cảnh, viết lời bình theo công thức viral, đọc voiceover tự nhiên và render đồng bộ hoàn hảo.**

[🇬🇧 English](README.md) · [Apache 2.0](LICENSE)

</div>

---

## Công cụ này làm được gì

Đưa vào 1 video (clip phim, tập phim, cảnh anime…), pipeline tạo ra 1 video hoàn toàn mới trong đó AI **xem** từng cảnh, **hiểu** cốt truyện, **viết** lời bình kiểu viral, và **đọc** thành giọng — đồng bộ với footage gốc, kèm phụ đề SRT.

### Các bước pipeline

| # | Stage | Chức năng |
|---|-------|-----------|
| 1 | **Preprocess** | Đọc thông số video, tách audio, phát hiện cảnh, trích keyframe |
| 2 | **ASR** | Nhận dạng giọng nói qua **FunASR** (mặc định), tự fallback sang **faster-whisper** |
| 3 | **Vision** | LLM phân tích nội dung thị giác từng cảnh (cảm xúc, hành động, độ đặc sắc) |
| 4 | **Semantic Graph** | Gộp phân tích audio + hình ảnh thành các block cốt truyện có cấu trúc |
| 5 | **Reference** *(tùy chọn)* | Lấy transcript từ video tham khảo/đối thủ để AI nắm chính xác cốt truyện |
| 6 | **Script** | Viết lời bình: câu hook, mạch truyện, góc nhìn, nhịp độ |
| 7 | **TTS** | Tổng hợp giọng đọc (piper / edge-tts / Gemini) |
| 8 | **Render** | Ghép video cuối cùng: audio + phụ đề đồng bộ |

Mỗi stage đều ghi **checkpoint gần như real-time**, nên nếu bị ngắt giữa chừng, lần chạy sau tự tiếp tục đúng chỗ dừng — kể cả trên máy khác (nhờ cloud sync).

---

## Bắt đầu nhanh

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt` chỉ chứa phần **lõi nhẹ** (scene detection, OpenCV, SRT, requests…). Các framework nặng được cài **đúng lúc cần**: `run.py` đọc backend bạn chọn trong `config.toml` rồi tự `pip install` đúng nhóm tùy chọn tương ứng (`requirements-asr.txt`, `requirements-vision.txt`, `requirements-llm.txt`, `requirements-tts.txt`, `requirements-cloud.txt`, `requirements-reference.txt`). Không GPU, không torch, không phải tải 8GB model nếu config của bạn không cần.

### 2. Cấu hình

Mở `config.toml` — toàn bộ cấu hình nằm trong file này (không dùng `.env`). Chọn backend cho từng stage:

| Section / key | Lựa chọn | Mặc định |
|---|---|---|
| `api.script_backend` | `local` (Gemma 4 12B QAT qua llama.cpp) · `cerebras` | `local` |
| `processing.asr_backend` | `funasr` · `whisper` | `funasr` |
| `processing.vision_backend` | `gemma4_video` (Ollama) · `local` (Qwen3-VL) · `moondream` · `cerebras` · `mistral` | `gemma4_video` |
| `tts.engine` | `piper` · `edge-tts` · `gemini` | `piper` |
| `cloud.enabled` | `true` / `false` (Tigris hoặc bất kỳ provider S3-compatible) | `true` |

### 3. Chạy

```bash
python run.py
```

Màn hình quản lý project hiện ra: tạo project, trỏ tới video, và chạy pipeline. Dùng `python run.py --no-menu` cho các lần chạy không cần tương tác (CI, cron).

---

## Quản lý project & cloud

- **Project** — mỗi video là 1 project với thư mục riêng (`projects/<project-id>/`) chứa metadata, checkpoint và output.
- **Cloud sync** — checkpoint và deliverables tự đồng bộ lên Tigris / bucket S3-compatible bất kỳ, để bạn tiếp tục project từ máy khác (hoặc runner GitHub Actions).
- **Smart resume** — tiếp tục từ stage cuối đã hoàn thành, hoặc đúng item đang dở trong stage, nhờ micro-checkpoint.

---

## GitHub Actions (chạy pipeline trên cloud không cần máy)

Workflow (`.github/workflows/run-pipeline.yml`) tự chạy pipeline hằng ngày trên runner GitHub:

1. Chọn project **mới nhất chưa hoàn thành** (local + cloud).
2. Tải về và tiếp tục từ checkpoint gần nhất.
3. Chạy toàn bộ stage rồi đồng bộ kết quả lên cloud.

Các điểm nổi bật:

- **Dependency theo backend** — workflow chỉ cài phần lõi; entry point Python tự cài phần còn lại theo backend *thực tế trên CI*. Khi `ci.force_lightweight_backends = true`, `config.py` tự đổi backend local nặng sang API tương đương (asr → whisper, vision → mistral, script → cerebras).
- **Tải model có điều kiện** — Ollama (Gemma 4 cho vision) chỉ được cài, pull và warm-up khi `config.toml` thực sự chọn `vision_backend = "gemma4_video"`.
- **Dashboard trực tiếp** — Cloudflare Quick Tunnel cấp 1 link công khai tạm thời tới dashboard tiến trình (% từng stage, ETA, tự refresh).
- **Chịu lỗi tốt** — giới hạn 6h/job của GitHub không phải vấn đề nhờ checkpoint; lần chạy bị ngắt tự tiếp tục ở lần chạy theo lịch kế tiếp.

> ⚠️ **Lưu ý bảo mật**: `config.toml` chứa API key thật được commit vào repo để workflow chạy không cần can thiệp. Đây là quyết định có chủ đích — xem ghi chú trong `config.toml` và nên đổi/giới hạn key nếu repo ở chế độ public.

---

## Các nhóm requirements

| File | Cần khi | Nặng? |
|---|---|---|
| `requirements.txt` | Luôn luôn (lõi) | Không |
| `requirements-asr.txt` | Stage ASR chạy (FunASR / faster-whisper) | torch qua funasr |
| `requirements-vision.txt` | `vision_backend = "local"` hoặc `"moondream"` (model transformers) | torch, transformers |
| `requirements-llm.txt` | `script_backend = "local"` (GGUF llama.cpp) hoặc `"cerebras"` (openai) | llama-cpp-python |
| `requirements-tts.txt` | Engine TTS `piper` / `edge-tts` / `gemini` | nhỏ |
| `requirements-cloud.txt` | `cloud.enabled = true` (boto3) | không |
| `requirements-reference.txt` | Có cung cấp URL video tham khảo | không |

`ensure_python_packages()` trong `run.py` chỉ cài đúng nhóm mà `config.toml` của bạn cần — config tối giản sẽ không bao giờ kéo theo torch/transformers.

---

## Cấu trúc project

```
AI-Director-Video/
├── run.py                    # Entry point chính (CLI + menu)
├── config.py                 # Bộ đọc config (TOML, truy cập kiểu dot-path)
├── config.toml               # Cấu hình (API keys, backend, đường dẫn)
├── requirements*.txt         # Lõi + các nhóm dependency tùy chọn
├── checkpoint.py             # Hệ thống checkpoint gần real-time
├── cloud_storage.py          # Đồng bộ Tigris / S3-compatible (boto3 lazy)
├── project_manager.py        # Quản lý project + điều phối cloud
├── platform_utils.py         # Tiện ích ffmpeg/torch
├── preprocess.py             # Stage 1 — cảnh, keyframe, audio
├── asr.py                    # Stage 2 — FunASR / faster-whisper
├── vision.py                 # Stage 3 — hiểu nội dung cảnh
├── semantic_graph.py         # Stage 4 — block cốt truyện
├── reference_video.py        # Stage 5 — transcript đối thủ
├── script_writer.py          # Stage 6 — lời bình
├── tts.py                    # Stage 7 — voiceover
├── render.py                 # Stage 8 — ghép video cuối
├── ci_run_latest_project.py  # Entry point cho GitHub Actions
└── .github/workflows/        # Pipeline CI theo lịch
```

---

## Xử lý sự cố

| Vấn đề | Cách xử lý |
|---|---|
| Video báo `No such file or directory` | Kiểm tra `paths.input_video` hoặc nhập đường dẫn khi được hỏi |
| Cloud sync lỗi | Kiểm tra key trong `[cloud]`; Tigris cần `region_name = "auto"`, `addressing_style = "virtual"` |
| CUDA OOM | Giảm `vision_batch_size`, dùng `vision_backend = "mistral"` / `"gemma4_video"`, hoặc nhỏ `asr_model_size` |
| Checkpoint hỏng | Xóa `checkpoints/<stage>.json` rồi chạy lại (resume từ stage trước) |
| Thiếu dependency tùy chọn | Chạy đúng lệnh `pip install -r requirements-*.txt` được ghi trong thông báo lỗi |

---

## Giấy phép

[Apache License 2.0](LICENSE) — © 2026 dungtrantriphimphenhuda-code
