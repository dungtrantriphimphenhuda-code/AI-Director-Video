"""
reference_video.py — Lấy transcript/tóm tắt từ video tham khảo (đối thủ) trên
YouTube để bổ sung ngữ cảnh cốt truyện cho script_writer (qua director_brief),
giúp giảm sai lệch nội dung (tên nhân vật, tình tiết, thứ tự sự kiện) khi AI
chỉ dựa vào ASR + vision của chính video nguồn (vốn có thể thiếu ngữ cảnh nếu
chỉ là 1 đoạn clip ngắn).

CHIẾN LƯỢC 2 TẦNG (ưu tiên nhẹ -> nặng), để "ít lỗi vặt":
  1. youtube-transcript-api — chỉ lấy phụ đề có sẵn qua API nội bộ của
     YouTube, KHÔNG giả lập trình duyệt tải file -> ít bị bot-detection soi
     hơn yt-dlp. Đây là lựa chọn ưu tiên.
  2. yt-dlp (--write-auto-sub --skip-download) — fallback khi video không có
     transcript qua API trên (phụ đề bị tắt, ngôn ngữ không khớp...), hoặc
     URL không phải YouTube. Hỗ trợ cookies từ trình duyệt (`reference.
     ytdlp_cookies_from_browser` trong config.toml) để giảm lỗi "Sign in to
     confirm you're not a bot" — xem README mục Troubleshooting.

AN TOÀN / KHÔNG LÀM VỠ PIPELINE:
  - Không có network tại đây (nếu proxy egress bị chặn) -> lỗi được bắt và
    log rõ, KHÔNG raise ra run.py. Nếu cả 2 tầng đều thất bại cho 1 URL,
    module trả về text rỗng cho URL đó và pipeline chạy tiếp bình thường
    (giống như khi người dùng không nhập link tham khảo nào).
  - Nếu thiếu thư viện (`youtube-transcript-api` hoặc `yt-dlp` chưa cài),
    tầng tương ứng tự bỏ qua thay vì crash — pipeline vẫn chạy được, chỉ
    thiếu phần bổ sung ngữ cảnh.

BẢN QUYỀN: chỉ dùng nội dung tham khảo để ĐỐI CHIẾU sự kiện/tên nhân vật.
Prompt gửi cho script_writer luôn kèm cảnh báo không sao chép nguyên văn
lời bình của nguồn tham khảo (xem `fetch_reference_brief`).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/|live/))([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    m = _YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def _via_transcript_api(url: str, languages: list[str]) -> str | None:
    """Tầng 1 (ưu tiên): youtube-transcript-api — nhẹ, ít bị chặn."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError:
        print(
            "[reference_video] Thư viện 'youtube-transcript-api' chưa cài "
            "(pip install youtube-transcript-api) — bỏ qua tầng 1, thử yt-dlp."
        )
        return None

    video_id = _extract_video_id(url)
    if not video_id:
        return None  # không phải link YouTube nhận dạng được -> để yt-dlp xử lý

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        text = " ".join(seg.text.strip() for seg in fetched if seg.text.strip())
        return text.strip() or None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"[reference_video] Video {video_id}: không có transcript phù hợp ({type(e).__name__}).")
        return None
    except Exception as e:
        # Bắt mọi lỗi khác (mạng, rate-limit, đổi API...) — không để crash pipeline.
        print(f"[reference_video] Lỗi youtube-transcript-api cho {video_id}: {e}")
        return None


def _parse_vtt(path: Path) -> str:
    """Trích text thô từ file phụ đề .vtt: bỏ timestamp/tag/số thứ tự, gộp
    dòng trùng lặp liền kề (auto-sub của YouTube hay lặp dòng do caption cuộn)."""
    lines_out: list[str] = []
    last = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:")) or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)  # bỏ tag <c>, timestamp inline kiểu <00:00:01.200>
        line = line.strip()
        if line and line != last:
            lines_out.append(line)
            last = line
    return " ".join(lines_out).strip()


def _via_ytdlp(url: str, languages: list[str], cookies_from_browser: str = "") -> str | None:
    """Tầng 2 (fallback): tải PHỤ ĐỀ (không tải video) bằng yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        print(
            "[reference_video] Thư viện 'yt-dlp' chưa cài (pip install yt-dlp) "
            "— không thể fallback, bỏ qua URL này."
        )
        return None

    with tempfile.TemporaryDirectory() as tmp:
        ydl_opts: dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": languages or ["vi", "en"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmp) / "%(id)s"),
            "quiet": True,
            "no_warnings": True,
            "retries": 2,
        }
        if cookies_from_browser:
            # Giúp né lỗi "Sign in to confirm you're not a bot" — xem README.
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"[reference_video] yt-dlp lỗi khi tải phụ đề cho {url}: {e}")
            return None

        vtt_files = sorted(Path(tmp).glob("*.vtt"))
        if not vtt_files:
            print(f"[reference_video] yt-dlp: không tìm thấy phụ đề nào cho {url}.")
            return None
        # Ưu tiên file khớp ngôn ngữ đầu tiên trong danh sách ưu tiên nếu có nhiều file.
        chosen = vtt_files[0]
        for lang in languages or []:
            for f in vtt_files:
                if f".{lang}." in f.name or f.name.endswith(f".{lang}.vtt"):
                    chosen = f
                    break
        return _parse_vtt(chosen)


def fetch_one(url: str, cfg) -> dict[str, Any]:
    """Lấy transcript cho 1 URL. KHÔNG BAO GIỜ raise — trả text rỗng khi thất bại."""
    languages = cfg.get("reference.languages", ["vi", "en"])
    max_chars = cfg.get("reference.max_chars_per_video", 6000)
    cookies_from_browser = cfg.get("reference.ytdlp_cookies_from_browser", "")
    use_ytdlp_fallback = cfg.get("reference.use_ytdlp_fallback", True)

    text = None
    method = None
    try:
        text = _via_transcript_api(url, languages)
        if text:
            method = "youtube_transcript_api"
    except Exception as e:
        print(f"[reference_video] Lỗi không mong đợi ở tầng 1 cho {url}: {e}")

    if not text and use_ytdlp_fallback:
        try:
            text = _via_ytdlp(url, languages, cookies_from_browser)
            if text:
                method = "yt-dlp"
        except Exception as e:
            print(f"[reference_video] Lỗi không mong đợi ở tầng 2 (yt-dlp) cho {url}: {e}")

    if not text:
        print(f"[reference_video] KHÔNG lấy được nội dung tham khảo từ: {url} "
              f"(pipeline vẫn tiếp tục bình thường, chỉ thiếu phần bổ sung này).")
        return {"url": url, "method": None, "text": "", "char_count": 0}

    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += " [...]"

    print(f"[reference_video] OK ({method}): {url} — {len(truncated)} ký tự.")
    return {"url": url, "method": method, "text": truncated, "char_count": len(truncated)}


def fetch_reference_brief(cfg, urls: list[str]) -> dict[str, Any]:
    """Lấy transcript cho nhiều URL tham khảo, gộp thành 1 khối text kèm cảnh
    báo bản quyền, dùng làm `director_brief` bổ sung cho script_writer."""
    urls = [u.strip() for u in (urls or []) if u.strip()]
    results = []
    for i, url in enumerate(urls, start=1):
        print(f"[reference_video] ({i}/{len(urls)}) Đang lấy nội dung tham khảo: {url}")
        results.append(fetch_one(url, cfg))

    parts = [f"[Nguồn tham khảo: {r['url']}]\n{r['text']}" for r in results if r["text"]]
    combined_brief = "\n\n".join(parts)

    note = ""
    if combined_brief:
        note = (
            "LƯU Ý QUAN TRỌNG: nội dung tham khảo ở trên CHỈ dùng để xác minh tên "
            "nhân vật, tình tiết, thứ tự sự kiện cho đúng — TUYỆT ĐỐI KHÔNG sao chép "
            "hay diễn giải sát nguyên văn lời bình của nguồn tham khảo. Narration phải "
            "được viết hoàn toàn bằng văn phong và câu chữ mới."
        )

    return {"sources": results, "combined_brief": combined_brief, "note": note}


def run_reference_stage(cfg, task_config: dict[str, Any], checkpoint_mgr=None) -> dict[str, Any]:
    """Entry point cho stage 'reference' trong run.py. Ghi reference_brief.json
    vào pipeline/ để resume không phải fetch lại (đỡ tốn thời gian và tránh
    bị rate-limit nếu gọi lại nhiều lần)."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    urls = task_config.get("reference_urls") or []
    if not urls:
        print("[reference_video] Không có link tham khảo nào — bỏ qua stage này.")
        result: dict[str, Any] = {"sources": [], "combined_brief": "", "note": ""}
    else:
        result = fetch_reference_brief(cfg, urls)

    with open(pipeline_dir / "reference_brief.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("reference", result)

    return result
