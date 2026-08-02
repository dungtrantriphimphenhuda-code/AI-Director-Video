"""
cleanup.py — dọn rác workspace trước/sau khi chạy pipeline trên CI.

Runner GitHub Actions miễn phí chỉ có ~14GB disk trống thật sự dùng được
(xem ghi chú trong config.py) — nếu ưu tiên chạy các backend "local" (tải
model AI về máy, có thể cộng dồn ~15-20GB), phần dư đó cần được tối đa hoá
bằng cách dọn sạch những thứ không cần thiết TRƯỚC khi tải model, và dọn
sạch rác tạm thời SAU mỗi lần chạy để lần sau (nếu có cache giữa các lần
chạy) không bị phình to dần.

Cố tình KHÔNG đụng tới:
  - Bất kỳ file nào trong `projects_dir` (project thật của người dùng) trừ
    file tạm rõ ràng là rác (vd `*.tmp`, `*.partial`, `*.incomplete`).
  - Output cuối cùng (final_preview.mp4, storyboard.json, ...).
  - Checkpoint — đây là cơ chế resume, xoá nhầm sẽ mất tiến độ.

An toàn: mọi bước đều bọc try/except riêng lẻ và chỉ in cảnh báo nếu lỗi,
không bao giờ raise ra ngoài — dọn rác thất bại không được phép làm hỏng
pipeline chính.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _rm(path: Path, label: str) -> int:
    """Xoá 1 file/thư mục, trả về số byte đã giải phóng (ước tính thô)."""
    try:
        if not path.exists():
            return 0
        size = _du(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        return size
    except Exception as exc:
        print(f"[cleanup] Bỏ qua '{label}' ({path}): {exc}")
        return 0


def _du(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def _fmt_mb(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.0f} MB"


def cleanup_workspace(repo_root: Path | None = None, projects_dir: Path | None = None,
                       model_cache_dir: Path | None = None) -> None:
    """
    Dọn rác an toàn, gọi được nhiều lần (đầu và cuối mỗi lần chạy CI):
      1. __pycache__ / *.pyc trong repo code (không phải trong projects_dir).
      2. pip cache (`pip cache purge`).
      3. File tải dở dang trong model_cache_dir (HuggingFace hub để lại
         *.incomplete / *.lock khi bị ngắt giữa chừng lúc tải).
      4. File tạm rõ ràng là rác trong projects_dir (*.tmp, *.partial).
    """
    repo_root = repo_root or Path(__file__).parent
    freed = 0

    print("[cleanup] Bắt đầu dọn rác workspace...")

    # 1) __pycache__ / *.pyc trong code repo (bỏ qua projects_dir để không
    #    đụng vào dữ liệu người dùng nếu projects_dir nằm trong repo_root).
    for pyc_dir in repo_root.rglob("__pycache__"):
        if projects_dir and projects_dir in pyc_dir.parents:
            continue
        freed += _rm(pyc_dir, "__pycache__")

    # 2) pip cache — không ảnh hưởng gì tới package ĐÃ CÀI, chỉ xoá cache tải.
    try:
        subprocess.run(["pip", "cache", "purge"], check=False,
                        capture_output=True, text=True)
        print("[cleanup] Đã purge pip cache.")
    except Exception as exc:
        print(f"[cleanup] Không purge được pip cache: {exc}")

    # 3) File model tải dở dang (an toàn để xoá — lần load model sau sẽ tự
    #    tải lại phần này, không mất checkpoint pipeline).
    if model_cache_dir and model_cache_dir.exists():
        for pattern in ("*.incomplete", "*.lock", "*.tmp"):
            for f in model_cache_dir.rglob(pattern):
                freed += _rm(f, f"model cache dở dang ({pattern})")

    # 4) File tạm rõ ràng là rác trong projects_dir (KHÔNG đụng checkpoint
    #    hay output thật).
    if projects_dir and projects_dir.exists():
        for pattern in ("*.tmp", "*.partial"):
            for f in projects_dir.rglob(pattern):
                freed += _rm(f, f"file tạm project ({pattern})")

    print(f"[cleanup] Xong — giải phóng khoảng {_fmt_mb(freed)}.")


def print_disk_usage(path: Path | None = None) -> None:
    """In dung lượng đĩa còn trống (tiện đối chiếu trước/sau khi dọn rác)."""
    try:
        usage = shutil.disk_usage(path or Path("/"))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(f"[cleanup] Disk: còn trống {free_gb:.1f}GB / tổng {total_gb:.1f}GB.")
    except Exception as exc:
        print(f"[cleanup] Không đọc được dung lượng đĩa: {exc}")


if __name__ == "__main__":
    print_disk_usage()
    cleanup_workspace()
    print_disk_usage()
