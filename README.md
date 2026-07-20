# AI Director Video Commentary — Enhanced Version

An AI-powered video commentary pipeline that automatically generates narrated commentary videos from any input video. Features project management, cloud sync (Tigris, S3-compatible), and near real-time checkpointing.

## What This Does

Input: Any video file (movie clip, drama scene, etc.)
Output: A new video with AI-generated voiceover commentary, properly synced with the original footage, plus SRT subtitles.

**Pipeline stages:**
1. **Preprocess** — Probe video, extract audio, detect scenes, extract keyframes
2. **ASR** — Speech-to-text transcription using FunASR (primary, `processing.asr_backend`), automatically falling back to faster-whisper if FunASR fails to load or errors out
3. **Vision** — AI analyzes each scene's visual content
4. **Semantic Graph** — Combines audio + visual analysis into structured blocks
5. **Reference** — (optional) fetches transcripts from reference/competitor video links, so the script writer knows the plot accurately
6. **Script** — AI writes viral-style narration commentary
7. **TTS** — Text-to-speech voiceover generation
8. **Render** — Final video assembly with synced commentary

### Reference video links (optional)

To help the AI get the plot right (character names, twists, event order) instead of guessing from the source clip alone, you can give it one or more reference/competitor video links (e.g. a full recap already on YouTube):

- When running a project you'll be prompted: `Link video tham khảo (đối thủ, cách nhau bởi dấu phẩy, tuỳ chọn)`. Paste one or more comma-separated YouTube links, or leave blank.
- Or pre-fill them in `config.toml` under `[reference] urls = [...]` so they're used automatically without prompting.
- It tries `youtube-transcript-api` first (lightweight, less likely to be blocked), and falls back to `yt-dlp --write-auto-sub` only if that fails. Both are optional dependencies — if neither is installed or a video has no captions, that URL is just skipped and the pipeline keeps going.
- If YouTube blocks `yt-dlp` with "Sign in to confirm you're not a bot", set `reference.ytdlp_cookies_from_browser = "chrome"` (or `"firefox"`, `"edge"`) in `config.toml` to use your real browser session's cookies.
- The fetched text is only used to fact-check names/plot points for the narration — the prompt explicitly tells the model not to copy the reference's wording, to avoid plagiarizing another channel's script.

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.toml` with your API keys:

```toml
[api]
cerebras_api_key = "your-cerebras-key"
mistral_api_key = "your-mistral-key"

[cloud]
access_key = "your-tigris-access-key"
secret_key = "your-tigris-secret-key"
bucket_name = "ai-director-video"
endpoint_url = "https://t3.storage.dev"
region_name = "auto"
addressing_style = "virtual"
enabled = true
```

### 3. Run

```bash
python run.py
```

You'll see the project management menu:

```
======================================================================
  AI DIRECTOR VIDEO — PROJECT MANAGER
======================================================================
  1. Create new project
  2. Continue existing project
  3. List all projects
  4. Delete a project
  5. Sync project to cloud (Tigris)
  6. Download project from cloud
  7. Run pipeline on a project
  0. Exit
======================================================================
```

---

## Project Management

### Creating a Project

1. Select **1. Create new project**
2. Enter a project ID (e.g., `my-movie-v1`)
3. Enter video file path
4. Enter project title

Each project gets its own directory:
```
projects/
  my-movie-v1/
    _project_meta.json      # Project metadata
    checkpoints/            # Pipeline checkpoints
    output/
      pipeline/            # Intermediate files
      deliverables/        # Final video + subtitles
```

### Continuing a Project

1. Select **2. Continue existing project**
2. Choose from the list
3. Pipeline resumes from last completed stage

### Cloud Sync

Projects are automatically synced to cloud storage. You can also:
- **5. Sync project to cloud** — Manual upload
- **6. Download project from cloud** — Download from another machine

---

## Features

### Near Real-Time Checkpoints

The pipeline saves checkpoints at multiple levels:
- **Stage checkpoints** — After each major stage (preprocess, ASR, vision, etc.)
- **Micro-checkpoints** — Within stages (per-scene in vision, per-clip in TTS/render)
- **Emergency checkpoints** — On SIGINT/SIGTERM (Ctrl+C)

Configure frequency in `config.toml`:
```toml
[processing]
micro_checkpoint_interval = 1  # Save every item (most frequent)
```

### Cloud Storage (Tigris)

All project data is synced to Tigris, or any other S3-compatible storage provider:
- Checkpoints sync automatically after each save
- Full project upload/download for cross-machine workflow
- Credentials stored in `config.toml` (not environment variables)

### Smart Resume

When you continue a project:
1. Scans all checkpoints to determine current stage
2. Skips completed stages automatically
3. Resumes from exact point of interruption
4. Works across different machines (with cloud sync)

---

## Configuration Reference

### `[api]` — API Keys

| Key | Description |
|-----|-------------|
| `cerebras_api_key` | Cerebras API key for script writing |
| `cerebras_model` | Model name (default: `zai-glm-4.7`) |
| `hf_token` | Hugging Face token (optional, for local vision) |
| `mistral_api_key` | Mistral API key for vision analysis |

### `[tts]` — Voice Settings

| Key | Description | Default |
|-----|-------------|---------|
| `voice` | Edge-TTS voice name | `vi-VN-HoangMinhNeural` |
| `rate` | Speech rate | `+0%` |
| `volume` | Volume adjustment | `+0%` |
| `pitch` | Pitch adjustment | `+0Hz` |

**Available voices:**
- Vietnamese: `vi-VN-HoangMinhNeural`, `vi-VN-NamMinhNeural`
- English: `en-US-JennyNeural`, `en-US-GuyNeural`
- Chinese: `zh-CN-XiaoxiaoNeural`, `zh-CN-YunxiNeural`

### `[processing]` — Pipeline Settings

| Key | Description | Default |
|-----|-------------|---------|
| `asr_backend` | ASR engine: `funasr` (primary) or `whisper` | `funasr` |
| `asr_model_size` | Whisper model size (used when `asr_backend="whisper"`, or as automatic fallback if FunASR fails) | `small` |
| `vision_backend` | Vision analysis backend | `mistral` |
| `micro_checkpoint_interval` | Checkpoint frequency | `1` |
| `narration_pov` | Narration point of view | `third_person` |
| `content_type` | Content type | `movie` |
| `target_duration_sec` | Target output duration | `180` |

### `[reference]` — Reference/Competitor Video Links

| Key | Description | Default |
|-----|--------------|---------|
| `urls` | Default reference video links (used if you leave the prompt blank) | `[]` |
| `languages` | Preferred subtitle languages, tried in order | `["vi", "en"]` |
| `max_chars_per_video` | Max characters kept per reference video (truncated beyond this) | `6000` |
| `use_ytdlp_fallback` | Fall back to yt-dlp when youtube-transcript-api fails | `true` |
| `ytdlp_cookies_from_browser` | Browser to pull cookies from for yt-dlp (`chrome`/`firefox`/`edge`), fixes bot-check errors | `""` |

### `[cloud]` — Cloud Storage

| Key | Description | Default |
|-----|-------------|---------|
| `access_key` | Cloud storage access key | — |
| `secret_key` | Cloud storage secret key | — |
| `bucket_name` | Storage bucket name | `ai-director-video` |
| `endpoint_url` | S3 endpoint URL | `https://t3.storage.dev` (Tigris) |
| `region_name` | Region (`auto` for Tigris) | `auto` |
| `addressing_style` | `path` or `virtual` (Tigris requires `virtual`) | `virtual` |
| `enabled` | Enable cloud sync | `true` |

### `[paths]` — File Paths

| Key | Description | Default |
|-----|-------------|---------|
| `input_video` | Input video path | `./input/source.mp4` |
| `output_dir` | Output directory | `./output` |
| `checkpoint_dir` | Checkpoint directory | `./checkpoints` |
| `projects_dir` | Projects directory | `./projects` |

---

## Command Line Usage

### Interactive Mode (Default)

```bash
python run.py
```

Shows project menu for managing and running projects.

### Non-Interactive Mode

```bash
# Run directly without menu (uses default project)
python run.py --no-menu
```

Or set in `config.toml`:
```toml
[project]
show_project_menu_on_start = false
```

---

## Chạy tự động bằng GitHub Actions (thay cho Colab)

Repo có sẵn workflow `.github/workflows/run-pipeline.yml` để chạy pipeline
trên máy ảo của GitHub thay vì phải mở Colab. Máy ảo này **không có GPU**,
nên nên đặt `vision_backend = "mistral"` (gọi API thay vì chạy model
Qwen3-VL local) trong `config.toml` — để `"local"` pipeline sẽ rất chậm hoặc
treo vì chạy vision model bằng CPU.

**Cách hoạt động:** vì máy ảo bị xoá sạch sau mỗi lần chạy, project phải
được lưu trên cloud (Tigris/S3) để "sống sót" qua các lần chạy. Mỗi lần
workflow chạy, nó tự động:
1. Liệt kê tất cả project trên cloud.
2. Bỏ qua project đã hoàn thành (`status = completed`).
3. Chọn project **mới nhất chưa xong**, tải về, và chạy tiếp từ checkpoint
   gần nhất (script: `ci_run_latest_project.py`).
4. Tự sync kết quả lên lại cloud.

Nếu không có project nào đang dang dở, workflow tự thoát êm — không báo lỗi.

### Thiết lập (làm 1 lần)

> **CẢNH BÁO:** repo này đang **public** và `config.toml` (chứa API key
> thật) đã được commit thẳng vào repo theo yêu cầu — bất kỳ ai xem repo
> đều thấy được các key này. Nếu key bị người khác lấy và dùng, bạn sẽ
> chịu chi phí/quota bị dùng trộm. Nên dùng key riêng cho việc này (dễ thu
> hồi) và cân nhắc đổi repo sang Private hoặc chuyển lại sang dùng GitHub
> Secrets nếu muốn an toàn hơn.

1. Điền đầy đủ key thật vào `config.toml` ở máy bạn (mục `[api]` và
   `[cloud]`), rồi commit + push file này lên GitHub:
   ```bash
   git add -f config.toml   # -f cần thiết vì trước đó file này bị .gitignore
   git commit -m "Thêm config.toml cho GitHub Actions"
   git push origin main
   ```

2. Tạo project và upload video như bình thường (`python run.py` ở máy cá
   nhân, hoặc Colab), rồi chọn **"5. Sync project to cloud"** để đẩy project
   lên cloud — đây là project mà workflow sẽ tìm thấy và chạy tiếp.

3. Push code lên nhánh `main` (workflow sẽ tự chạy), hoặc vào tab
   **Actions → Run AI Director Video Pipeline → Run workflow** để bấm chạy
   tay, hoặc chờ đến 8:00 sáng hôm sau (lịch chạy tự động mỗi ngày).

### Lưu ý quan trọng

- **Giới hạn 6 giờ/job**: GitHub Actions tự ngắt job sau 6 giờ dù chưa xong
  (giới hạn cứng của GitHub, không thể tăng). Nhờ checkpoint gần như
  real-time, lần chạy sau sẽ tự tiếp tục đúng chỗ dừng, không mất tiến độ.
- **Không có GPU**: mọi bước cần GPU (vision local, ASR nếu chọn model lớn)
  sẽ chạy bằng CPU, chậm hơn Colab có GPU khá nhiều — phù hợp cho video
  ngắn/vừa. Nếu cần nhanh hơn, cân nhắc self-hosted runner có GPU riêng.
- **Chỉ chạy 1 project/lần**: nếu có nhiều project dang dở, mỗi lần chạy chỉ
  xử lý 1 project (mới nhất); các project còn lại sẽ được xử lý ở lần chạy
  kế tiếp.
- **Mỗi lần sửa key**: sửa `config.toml` ở máy bạn rồi commit + push lại —
  workflow luôn dùng đúng file `config.toml` mới nhất trong repo.

### Ưu tiên chạy backend "local" ngay trên CI

Mặc định, `config.py` tự đổi các backend "local" (tốn disk/RAM) sang API khi
phát hiện chạy trong GitHub Actions. Nếu bạn đã tự xác nhận runner đủ tài
nguyên (vd 4 vCPU / 16GB RAM + disk đã dọn rác qua bước "Free disk space"),
đặt trong `config.toml`:

```toml
[ci]
force_lightweight_backends = false
```

Rủi ro: nếu ước tính sai, job có thể bị runner huỷ giữa chừng vì hết disk/
RAM — nhờ checkpoint gần real-time, chạy lại workflow sẽ tự tiếp tục đúng
chỗ dừng.

### Theo dõi tiến trình trực tiếp (dashboard qua Cloudflare Quick Tunnel)

Mỗi lần workflow chạy, nó tự mở 1 server tiến trình nội bộ (cổng 8787),
dùng `cloudflared tunnel --url http://localhost:8787` để cấp 1 link công
khai tạm thời (`https://xxxx.trycloudflare.com`) — **không cần tài khoản
Cloudflare hay API token**. Link này xuất hiện trong tab **Actions** →
lần chạy tương ứng → phần **Summary**, hiện % hoàn tất từng stage kèm dự
đoán thời gian hoàn thành (ETA), tự làm mới mỗi 3 giây. Link chỉ tồn tại
trong lúc job đang chạy.

### Dọn rác tự động

Workflow dọn rác 2 lần: (1) gỡ bớt công cụ cài sẵn không dùng tới trên
runner (.NET, Android SDK, GHC...) để lấy lại disk trước khi tải model, và
(2) sau khi pipeline chạy xong (`cleanup.py`), dọn `__pycache__`, pip
cache, và file model tải dở dang — không đụng tới checkpoint hay output
thật của project.

---

## File Structure

```
AI-Director-Video/
├── run.py                    # Main entry point
├── config.toml               # Configuration (your API keys)
├── config.toml.example       # Example configuration
├── requirements.txt          # Python dependencies
│
├── checkpoint.py             # Enhanced checkpoint system
├── cloud_storage.py          # Cloud storage (Tigris / S3-compatible)
├── project_manager.py        # Project management
│
├── preprocess.py             # Video preprocessing
├── asr.py                    # Speech-to-text
├── vision.py                 # Visual analysis
├── semantic_graph.py         # Data combination
├── script_writer.py          # Narration generation
├── tts.py                    # Text-to-speech
├── render.py                 # Video rendering
│
├── config.py                 # Config loader
├── platform_utils.py         # Platform utilities
├── progress_utils.py         # Progress tracking
│
├── projects/                 # Project directories
│   └── <project-id>/
│       ├── _project_meta.json
│       ├── checkpoints/
│       └── output/
│
├── input/                    # Input videos
├── output/                   # Default output
├── checkpoints/              # Default checkpoints
└── model_cache/              # Downloaded AI models
```

---

## Troubleshooting

### "No such file or directory" for video
- Check `paths.input_video` in `config.toml`
- Or enter video path when prompted

### Cloud sync fails
- Check `access_key` and `secret_key` in `[cloud]`
- Ensure bucket name is correct and, for Tigris, that `region_name = "auto"` and `addressing_style = "virtual"`
- Pipeline continues locally even if cloud sync fails

### Checkpoint not found
- Delete the corrupted checkpoint: `rm checkpoints/<stage>.json`
- Or create fresh project and re-run

### Out of memory (CUDA)
- Reduce `vision_batch_size` in config
- Use `vision_backend = "mistral"` instead of `"local"`
- Use smaller `asr_model_size`

---

## API Keys Setup

### Cerebras (for script writing)
1. Sign up at https://cerebras.ai
2. Get API key from dashboard
3. Add to `config.toml`: `cerebras_api_key = "your-key"`

### Mistral (for vision analysis)
1. Sign up at https://mistral.ai
2. Get API key from dashboard
3. Add to `config.toml`: `mistral_api_key = "your-key"`

### Tigris (for cloud storage)
1. Sign up at https://www.tigrisdata.com and create an access key + bucket
2. Fill in `[cloud]` in `config.toml` with the access key, secret key, and bucket name
3. Set `cloud.enabled = true` to use cloud sync (endpoint/region/addressing_style already default to Tigris)

---

## Tips

1. **Start simple** — Use short videos (1-3 minutes) first
2. **Check checkpoints** — View `checkpoints/` folder to see progress
3. **Use cloud sync** — Sync important projects to continue on another machine
4. **Customize voice** — Try different TTS voices in config
5. **Adjust duration** — Set `target_duration_sec` for desired output length

---

## License

This project is provided as-is for educational and personal use.
