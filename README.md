<div align="center">

# 🎬 AI Director Video Commentary

**End-to-end AI pipeline that turns any source video into a narrated commentary video — scene-by-scene AI analysis, viral-style narration, natural TTS voiceover, and perfectly synced rendering.**

[🇻🇳 Tiếng Việt](README.vi.md) · [Apache 2.0](LICENSE)

</div>

---

## What it does

Drop in a video (a movie clip, a drama episode, an anime scene…) and the pipeline produces a brand-new video where an AI **watches** every scene, **understands** the plot, **writes** viral-style narration, and **speaks** it — synced to the original footage, with SRT subtitles included.

### Pipeline stages

| # | Stage | What happens |
|---|-------|--------------|
| 1 | **Preprocess** | Probe the video, extract audio, detect scenes, extract keyframes |
| 2 | **ASR** | Speech-to-text via **FunASR** (default), auto-fallback to **faster-whisper** |
| 3 | **Vision** | An LLM analyzes each scene's visual content (emotions, actions, intensity) |
| 4 | **Semantic Graph** | Merges audio + visual analysis into structured story blocks |
| 5 | **Reference** *(optional)* | Pulls transcripts from reference/competitor videos to get the plot exactly right |
| 6 | **Script** | Writes the narration: hook lines, story arc, POV, pacing |
| 7 | **TTS** | Synthesizes voiceover (piper / edge-tts / Gemini) |
| 8 | **Render** | Assembles the final video with synced audio + subtitles |

Every stage writes **near real-time checkpoints**, so an interrupted run resumes exactly where it stopped — even on a different machine (via cloud sync).

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

`requirements.txt` contains only the **lightweight core** (scene detection, OpenCV, SRT, requests…). Heavy frameworks are installed **on demand**: `run.py` inspects the backends you selected in `config.toml` and `pip install`s exactly the matching optional group (`requirements-asr.txt`, `requirements-vision.txt`, `requirements-llm.txt`, `requirements-tts.txt`, `requirements-cloud.txt`, `requirements-reference.txt`). No GPU, no torch, no 8 GB of models if your config doesn't need them.

### 2. Configure

Edit `config.toml` — every setting lives there (no `.env` files). Pick a backend per stage:

| Section / key | Options | Default |
|---|---|---|
| `api.script_backend` | `local` (Gemma 4 12B QAT via llama.cpp) · `cerebras` | `local` |
| `processing.asr_backend` | `funasr` · `whisper` | `funasr` |
| `processing.vision_backend` | `gemma4_video` (Ollama) · `local` (Qwen3-VL) · `moondream` · `cerebras` · `mistral` | `gemma4_video` |
| `tts.engine` | `piper` · `edge-tts` · `gemini` | `piper` |
| `cloud.enabled` | `true` / `false` (Tigris or any S3-compatible provider) | `true` |

### 3. Run

```bash
python run.py
```

You get a project manager menu: create a project, point it at a video, and run the pipeline. Use `python run.py --no-menu` for unattended runs (CI, cron).

---

## Project management & cloud

- **Projects** — each video is a project with its own folder (`projects/<project-id>/`) holding metadata, checkpoints, and outputs.
- **Cloud sync** — checkpoints and deliverables auto-sync to Tigris / any S3-compatible bucket, so you can continue a project from another machine (or a GitHub Actions runner).
- **Smart resume** — resumes from the last completed stage, or the exact item within a stage, using micro-checkpoints.

---

## GitHub Actions (headless cloud pipeline)

A workflow (`.github/workflows/run-pipeline.yml`) runs the pipeline daily on GitHub-hosted runners:

1. Picks the most recent **unfinished** project (local + cloud).
2. Downloads it and resumes from the last checkpoint.
3. Runs all stages and syncs results back to cloud.

Key behaviors:

- **Backend-aware dependencies** — the workflow installs only the core requirements; the Python entry point auto-installs the rest based on the *effective* CI backends. When `ci.force_lightweight_backends = true`, `config.py` transparently swaps heavy local backends for API equivalents (asr → whisper, vision → mistral, script → cerebras).
- **Gated model downloads** — Ollama (Gemma 4 for vision) is only installed, pulled, and warmed up when `config.toml` actually selects `vision_backend = "gemma4_video"`.
- **Live dashboard** — a Cloudflare Quick Tunnel gives a temporary public link to a progress dashboard (per-stage %, ETA, auto-refresh).
- **Resilient** — 6 h/job GitHub limit is fine thanks to checkpoints; interrupted runs resume on the next scheduled run.

> ⚠️ **Security note**: `config.toml` with real API keys is committed to the repo so the workflow can run unattended. This is a deliberate choice — see the note in `config.toml` and rotate/limit keys if the repo is public.

---

## Requirements groups

| File | Needed when | Heavy? |
|---|---|---|
| `requirements.txt` | Always (core) | No |
| `requirements-asr.txt` | ASR stage runs (FunASR / faster-whisper) | torch via funasr |
| `requirements-vision.txt` | `vision_backend = "local"` or `"moondream"` (transformers models) | torch, transformers |
| `requirements-llm.txt` | `script_backend = "local"` (llama.cpp GGUF) or `"cerebras"` (openai) | llama-cpp-python |
| `requirements-tts.txt` | TTS engine `piper` / `edge-tts` / `gemini` | small |
| `requirements-cloud.txt` | `cloud.enabled = true` (boto3) | no |
| `requirements-reference.txt` | reference video URLs provided | no |

`ensure_python_packages()` in `run.py` installs exactly the groups your `config.toml` needs — so a minimal config never drags in torch/transformers.

---

## Project structure

```
AI-Director-Video/
├── run.py                    # Main entry point (CLI + menu)
├── config.py                 # Config loader (TOML, dot-path access)
├── config.toml               # Configuration (API keys, backends, paths)
├── requirements*.txt         # Core + optional dependency groups
├── checkpoint.py             # Near real-time checkpoint system
├── cloud_storage.py          # Tigris / S3-compatible sync (lazy boto3)
├── project_manager.py        # Project CRUD + cloud orchestration
├── platform_utils.py         # ffmpeg/torch helpers
├── preprocess.py             # Stage 1 — scenes, keyframes, audio
├── asr.py                    # Stage 2 — FunASR / faster-whisper
├── vision.py                 # Stage 3 — scene understanding
├── semantic_graph.py         # Stage 4 — story blocks
├── reference_video.py        # Stage 5 — competitor transcripts
├── script_writer.py          # Stage 6 — narration
├── tts.py                    # Stage 7 — voiceover
├── render.py                 # Stage 8 — final assembly
├── ci_run_latest_project.py  # GitHub Actions entry point
└── .github/workflows/        # Scheduled CI pipeline
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No such file or directory` for the video | Check `paths.input_video` or enter the path when prompted |
| Cloud sync fails | Verify `[cloud]` keys; Tigris needs `region_name = "auto"`, `addressing_style = "virtual"` |
| CUDA OOM | Lower `vision_batch_size`, use `vision_backend = "mistral"` / `"gemma4_video"`, or smaller `asr_model_size` |
| Corrupted checkpoint | Delete `checkpoints/<stage>.json` and re-run (resumes from previous stage) |
| Missing optional dependency | Run the matching `pip install -r requirements-*.txt` listed in the error message |

---

## License

[Apache License 2.0](LICENSE) — © 2026 dungtrantriphimphenhuda-code
