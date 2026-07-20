#!/usr/bin/env python3
"""
ci_run_latest_project.py — Entry point dùng riêng cho GitHub Actions (hoặc
bất kỳ môi trường chạy không tương tác nào khác, vd cron trên VPS).

Khác với `run.py --no-menu` (luôn chạy trên 1 project cố định tên "default"),
script này:
  1. Liệt kê TẤT CẢ project (cục bộ + cloud) qua ProjectManager.
  2. Bỏ qua project đã "completed" (đã render xong final_preview.mp4).
  3. Trong số project CHƯA xong, chọn project có "last_modified" gần đây
     nhất (dựa trên LastModified thật của object trên cloud — xem
     cloud_storage.list_remote_projects — nên vẫn đúng dù máy chạy CI
     không có ổ đĩa cũ để so mtime cục bộ).
  4. Nếu project đó chỉ có trên cloud (hoặc chưa có ở máy đang chạy), tải
     nó về rồi mới chạy tiếp từ checkpoint gần nhất.
  5. Nếu không có project nào đang chờ xử lý, thoát êm (exit code 0) —
     KHÔNG coi là lỗi, để lịch chạy tự động (schedule) mỗi 8h sáng không
     bị đánh dấu "failed" trên GitHub chỉ vì chưa có việc để làm.

Yêu cầu: config.toml đã tồn tại (workflow tạo sẵn từ secrets trước khi gọi
script này) và mục [cloud] enabled = true, vì đây là kênh DUY NHẤT để
project "sống sót" qua các lần chạy (runner của GitHub Actions bị xoá sạch
sau mỗi lần chạy).

Chạy:
    python ci_run_latest_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config  # noqa: E402
from cloud_storage import get_cloud_storage_from_config  # noqa: E402
from project_manager import ProjectManager  # noqa: E402
from platform_utils import ensure_ffmpeg  # noqa: E402
from cleanup import cleanup_workspace, print_disk_usage  # noqa: E402
import progress_server  # noqa: E402
import run as run_module  # noqa: E402  (tái dùng run_pipeline_on_project có sẵn)


def pick_next_project(pm: ProjectManager) -> dict | None:
    """Chọn project cần chạy tiếp: chưa 'completed', mới sửa gần đây nhất."""
    projects = pm.list_all_projects(include_cloud=True)
    candidates = [p for p in projects if p.get("status") != "completed"]
    if not candidates:
        return None
    # last_modified dạng "YYYY-MM-DD HH:MM:SS" -> so sánh chuỗi là đủ (ISO-like).
    candidates.sort(key=lambda p: p.get("last_modified") or "", reverse=True)
    return candidates[0]


def main() -> int:
    print("=" * 70)
    print("  AI DIRECTOR VIDEO — CI runner (GitHub Actions)")
    print("=" * 70)

    # Dọn rác TRƯỚC khi tải/load bất kỳ model nào — quan trọng nhất khi
    # ưu tiên backend "local" trên CI (ci.force_lightweight_backends=false),
    # vì disk trống ban đầu quyết định model có tải xong được hay không.
    print_disk_usage()
    cleanup_workspace()

    ensure_ffmpeg()
    cfg = load_config("config.toml")
    cloud = get_cloud_storage_from_config(cfg)

    # Mở dashboard tiến trình (server nội bộ) — bước riêng trong workflow sẽ
    # dùng cloudflared để lấy link công khai tạm thời trỏ vào cổng này.
    progress_http_server = progress_server.start_server(
        port=int(cfg.get("ci.progress_port", 8787))
    )

    if cloud is None or not getattr(cloud, "bucket_ready", False):
        print("[ci] CẢNH BÁO: cloud storage chưa sẵn sàng (kiểm tra secrets phần "
              "[cloud] trong config.toml). Không có cloud, mỗi lần chạy CI sẽ "
              "luôn bắt đầu project rỗng vì máy ảo bị xoá sau mỗi lần chạy.")

    projects_dir = cfg.resolve_path("paths.projects_dir")
    checkpoint_subdir = run_module._checkpoint_subdir_name(cfg)
    pm = ProjectManager(projects_dir, cloud, checkpoint_subdir=checkpoint_subdir)

    chosen = pick_next_project(pm)
    if chosen is None:
        print("[ci] Không có project nào đang chờ xử lý (tất cả đã 'completed' "
              "hoặc chưa có project nào). Bỏ qua lần chạy này — không phải lỗi.")
        progress_server.STATE.mark_done()
        progress_http_server.shutdown()
        return 0

    project_id = chosen["project_id"]
    print(f"[ci] Đã chọn project: '{project_id}' "
          f"(status={chosen.get('status')}, last_modified={chosen.get('last_modified')})")
    progress_server.STATE.project_id = project_id

    local_dir = pm.base_dir / project_id
    if not local_dir.exists() or chosen.get("source") == "cloud_only":
        ok = pm.sync_from_cloud(project_id)
        if not ok:
            print(f"[ci] LỖI: không tải được project '{project_id}' từ cloud.")
            progress_server.STATE.mark_error(f"Không tải được project '{project_id}' từ cloud.")
            progress_http_server.shutdown()
            return 1

    try:
        run_module.run_pipeline_on_project(cfg, pm, project_id, cloud)
    except Exception as exc:
        progress_server.STATE.mark_error(str(exc))
        raise
    else:
        progress_server.STATE.mark_done()
    finally:
        # Đoạn dọn rác THỨ HAI: sau khi chạy xong (dù thành công hay lỗi),
        # dọn lại model cache dở dang/pip cache trước khi job kết thúc —
        # giúp lần chạy kế tiếp (nếu runner tái sử dụng cache giữa các lần
        # chạy qua actions/cache) không bị phình to dần theo thời gian.
        print_disk_usage()
        cleanup_workspace(model_cache_dir=cfg.resolve_path("paths.model_cache_dir"),
                           projects_dir=cfg.resolve_path("paths.projects_dir"))
        print_disk_usage()
        progress_http_server.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
