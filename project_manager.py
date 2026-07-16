"""
project_manager.py — Hệ thống quản lý project thông minh.

Quét các project cục bộ, đồng bộ với Filebase, và cho phép người dùng:
  - Liệt kê tất cả project (local + cloud)
  - Tiếp tục 1 project (resume từ checkpoint)
  - Xoá 1 project (local và/hoặc cloud)
  - Tạo project mới
  - Nhập video vào 1 project

Mỗi project có thư mục riêng chứa video gốc (input/), checkpoint, và output.
Video gốc được COPY vào bên trong project_dir/input/ ngay khi tạo project
(thay vì chỉ lưu đường dẫn tuyệt đối bên ngoài) — để khi sync/tải project
qua máy khác, video luôn đi kèm và pipeline có thể chạy tiếp ngay, không
cần người dùng tìm lại/nhập lại đường dẫn video gốc.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


class ProjectManager:
    """Quản lý các project của AI Director Video, cả cục bộ lẫn trên Filebase."""

    def __init__(self, base_dir: Path, filebase_storage=None):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.filebase = filebase_storage
        self.projects_index_path = self.base_dir / "_projects_index.json"

    def scan_local_projects(self) -> list[dict[str, Any]]:
        """Quét toàn bộ thư mục project bên trong base_dir."""
        projects = []
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
                continue
            meta = self._load_project_meta(d)
            if meta is None:
                meta = self._create_project_meta(d)
            else:
                # QUAN TRỌNG: dù đã có _project_meta.json (ghi lần đầu lúc
                # create_project()), vẫn phải quét lại checkpoint/output mỗi
                # lần liệt kê để cập nhật status/stages_completed thật. Nếu
                # chỉ đọc thẳng file cache như trước, project sẽ hiện mãi ở
                # trạng thái "new"/0 stage dù pipeline đã chạy xong nhiều
                # bước — vì không có gì khiến cache đó được ghi đè lại.
                meta = self._refresh_project_meta(d, meta)
            projects.append(meta)
        return projects

    def _load_project_meta(self, project_dir: Path) -> dict[str, Any] | None:
        """Đọc metadata project từ _project_meta.json."""
        meta_file = project_dir / "_project_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_project_meta(self, project_dir: Path, meta: dict[str, Any]) -> None:
        """Ghi metadata project."""
        meta_file = project_dir / "_project_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _refresh_project_meta(self, project_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
        """Cập nhật các trường suy ra được từ đĩa (status, stages_completed,
        has_input_video, has_final_output, last_modified...) dựa trên
        checkpoint/input/output thật hiện có, đè lên metadata cũ đã cache.
        Các trường không thể suy ra từ đĩa (title, created_at...) được giữ
        nguyên từ metadata cũ.
        """
        # Quét checkpoint để xác định trạng thái
        ckpt_dir = project_dir / "checkpoints"
        if ckpt_dir.exists():
            stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
            completed = [s for s in stages if (ckpt_dir / f"{s}.json").exists()]
            meta["stages_completed"] = completed
            meta["total_stages"] = len(stages)
            if len(completed) == len(stages):
                meta["status"] = "completed"
            elif len(completed) > 0:
                meta["status"] = "in_progress"
            else:
                meta["status"] = "new"

        # Kiểm tra có config riêng cho project không
        cfg_file = project_dir / "config.toml"
        meta["has_config"] = cfg_file.exists()

        # Kiểm tra video gốc nằm bên trong project_dir/input/
        input_dir = project_dir / "input"
        if input_dir.exists():
            video_files = [
                p for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi", ".webm")
            ]
            if video_files:
                meta["video_path"] = f"input/{video_files[0].name}"
                meta["has_input_video"] = True
            else:
                meta["has_input_video"] = False

        # Kiểm tra video đầu ra cuối cùng
        out_dir = project_dir / "output"
        if out_dir.exists():
            final = out_dir / "deliverables" / "final_preview.mp4"
            if final.exists():
                meta["has_final_output"] = True
                meta["status"] = "completed"

        meta["last_modified"] = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(max(project_dir.stat().st_mtime, 0))
        )

        self._save_project_meta(project_dir, meta)
        return meta

    def _create_project_meta(self, project_dir: Path) -> dict[str, Any]:
        """Tạo metadata bằng cách quét thư mục project."""
        meta = {
            "project_id": project_dir.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": "",
            "title": project_dir.name,
            "status": "unknown",
            "last_modified": "",
            "stages_completed": [],
            "total_stages": 7,
        }
        return self._refresh_project_meta(project_dir, meta)

    def create_project(self, project_id: str, video_path: str = "", title: str = "") -> Path:
        """Tạo thư mục project mới với đầy đủ cấu trúc con."""
        project_dir = self.base_dir / project_id
        if project_dir.exists():
            raise ValueError(f"Project '{project_id}' đã tồn tại.")

        project_dir.mkdir(parents=True)
        (project_dir / "input").mkdir()
        (project_dir / "checkpoints").mkdir()
        (project_dir / "output" / "pipeline").mkdir(parents=True)
        (project_dir / "output" / "deliverables").mkdir(parents=True)

        # Copy video gốc vào BÊN TRONG project_dir/input/ thay vì chỉ lưu
        # đường dẫn tuyệt đối bên ngoài. Nếu chỉ lưu đường dẫn, khi
        # sync_to_cloud() upload project_dir thì video (nằm ngoài thư mục
        # này) sẽ không được upload -> tải project về máy khác sẽ có đủ
        # checkpoint nhưng thiếu video gốc để chạy tiếp các bước cần lại nó.
        rel_video_path = ""
        if video_path:
            src = Path(video_path).expanduser()
            if src.exists() and src.is_file():
                dest = project_dir / "input" / src.name
                shutil.copy2(src, dest)
                rel_video_path = f"input/{src.name}"
                print(f"[project] Đã copy video gốc vào project: {dest}")
            else:
                print(f"[project] Cảnh báo: không tìm thấy video tại '{video_path}', "
                      f"bỏ qua bước copy (có thể thêm video vào thư mục "
                      f"'{project_dir / 'input'}' sau).")

        meta = {
            "project_id": project_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": rel_video_path,
            "title": title or project_id,
            "status": "new",
            "last_modified": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages_completed": [],
            "total_stages": 7,
            "has_config": False,
            "has_input_video": bool(rel_video_path),
        }
        self._save_project_meta(project_dir, meta)
        print(f"[project] Đã tạo project: {project_id}")
        return project_dir

    def delete_project(self, project_id: str, cloud: bool = False) -> bool:
        """Xoá 1 project cục bộ, và tuỳ chọn xoá luôn trên Filebase."""
        project_dir = self.base_dir / project_id
        if not project_dir.exists():
            print(f"[project] Không tìm thấy project '{project_id}' cục bộ.")
            return False

        shutil.rmtree(project_dir)
        print(f"[project] Đã xoá project cục bộ: {project_id}")

        if cloud and self.filebase:
            ok = self.filebase.delete_project(project_id)
            if ok:
                print(f"[project] Đã xoá project trên cloud: {project_id}")
            return ok
        return True

    def get_project_status(self, project_id: str) -> dict[str, Any]:
        """Lấy trạng thái chi tiết của 1 project."""
        project_dir = self.base_dir / project_id
        if not project_dir.exists():
            return {"exists": False, "project_id": project_id}

        meta = self._load_project_meta(project_dir)
        if meta is None:
            meta = self._create_project_meta(project_dir)

        meta["exists"] = True
        meta["local_path"] = str(project_dir)

        # Kiểm tra checkpoint
        ckpt_dir = project_dir / "checkpoints"
        if ckpt_dir.exists():
            stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
            for s in stages:
                ckpt_file = ckpt_dir / f"{s}.json"
                meta[f"stage_{s}_done"] = ckpt_file.exists()

        return meta

    def list_all_projects(self, include_cloud: bool = True) -> list[dict[str, Any]]:
        """Liệt kê tất cả project (cục bộ + cloud nếu được bật)."""
        local_projects = self.scan_local_projects()
        local_ids = {p["project_id"] for p in local_projects}

        if include_cloud and self.filebase:
            remote_projects = self.filebase.list_remote_projects()
            for rp in remote_projects:
                if rp["project_id"] not in local_ids:
                    rp["source"] = "cloud_only"
                    rp["status"] = "cloud_only"
                    local_projects.append(rp)

        return local_projects

    def display_projects(self, projects: list[dict[str, Any]]) -> None:
        """In danh sách project ra màn hình dạng bảng."""
        if not projects:
            print("\n  Không tìm thấy project nào.")
            return

        print(f"\n  {'#':<4} {'Project ID':<30} {'Trạng thái':<15} {'Stage':<10} {'Sửa lần cuối':<20} {'Nguồn':<10}")
        print("  " + "-" * 95)

        for i, p in enumerate(projects, 1):
            pid = p.get("project_id", "?")
            status = p.get("status", "unknown")
            completed = len(p.get("stages_completed", []))
            total = p.get("total_stages", 7)
            modified = p.get("last_modified", "?")
            source = p.get("source", "local")
            status_icon = {
                "completed": "DONE",
                "in_progress": "PROG",
                "new": "NEW",
                "unknown": "???",
                "cloud_only": "CLD",
            }.get(status, "???")
            print(f"  {i:<4} {pid:<30} {status_icon:<15} {completed}/{total:<10} {modified:<20} {source:<10}")

    def prompt_select_project(self, projects: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Hỏi người dùng chọn 1 project từ danh sách."""
        if not projects:
            return None

        self.display_projects(projects)
        print(f"\n  Nhập số thứ tự project (1-{len(projects)}) hoặc Enter để huỷ:")

        try:
            choice = input("  > ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(projects):
                return projects[idx]
            print("  Lựa chọn không hợp lệ.")
        except (ValueError, EOFError):
            print("  Giá trị nhập không hợp lệ.")
        return None

    def sync_to_cloud(self, project_id: str) -> dict[str, Any]:
        """Đồng bộ 1 project cục bộ lên Filebase Storage."""
        if not self.filebase:
            return {"error": "Chưa cấu hình Filebase"}

        project_dir = self.base_dir / project_id
        if not project_dir.exists():
            return {"error": f"Không tìm thấy project '{project_id}' cục bộ"}

        print(f"[filebase] Đang upload project '{project_id}' lên cloud "
              f"(bao gồm cả video gốc trong input/, checkpoint, và output)...")
        result = self.filebase.upload_project(project_dir, project_id)
        print(f"[filebase] Upload xong: {result['uploaded']} file, "
              f"{result.get('skipped', 0)} file bỏ qua (đã có sẵn, không đổi), "
              f"{result['errors']} lỗi.")
        return result

    def sync_from_cloud(self, project_id: str) -> bool:
        """Tải 1 project từ Filebase Storage về máy."""
        if not self.filebase:
            print("[filebase] Chưa cấu hình Filebase.")
            return False

        project_dir = self.base_dir / project_id
        print(f"[filebase] Đang tải project '{project_id}' từ cloud...")
        ok = self.filebase.download_project(project_id, project_dir)
        if ok:
            self._create_project_meta(project_dir)
            print(f"[filebase] Tải xong: {project_dir}")
        return ok

    def prompt_action(self) -> str | None:
        """Hiện menu hành động và nhận lựa chọn của người dùng."""
        print("\n" + "=" * 60)
        print("  AI DIRECTOR VIDEO — QUẢN LÝ PROJECT")
        print("=" * 60)
        print("  1. Tạo project mới")
        print("  2. Tiếp tục project có sẵn")
        print("  3. Liệt kê tất cả project")
        print("  4. Xoá 1 project")
        print("  5. Đồng bộ project lên cloud (Filebase)")
        print("  6. Tải project từ cloud về")
        print("  7. Chạy pipeline trên 1 project")
        print("  0. Thoát")
        print("=" * 60)

        try:
            choice = input("  Chọn (0-7): ").strip()
            return choice
        except (EOFError, KeyboardInterrupt):
            return None
