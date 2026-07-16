#!/usr/bin/env python3
"""
run.py — Entry point nâng cao với quản lý project, sync Filebase,
checkpoint gần như real-time, và UX cải tiến.

Tính năng:
  - Menu project thông minh: tạo, liệt kê, tiếp tục, xoá, đồng bộ
  - Checkpoint gần như real-time (mỗi scene/batch)
  - Tự động sync lên Filebase cloud storage
  - Xử lý ngắt an toàn (resume từ checkpoint gần nhất)
  - Theo dõi tiến độ kèm ETA

Mỗi project được cách ly dữ liệu input/output riêng: khi chạy pipeline trên
1 project, paths.input_video và paths.output_dir trong cfg được trỏ lại vào
đúng project_dir đó (xem run_pipeline_on_project) — nếu không, dù menu cho
chọn project khác nhau, pipeline vẫn luôn đọc/ghi vào đường dẫn toàn cục
trong config.toml, khiến việc "chọn project" không thực sự cách ly gì cả.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config  # noqa: E402
from checkpoint import CheckpointManager  # noqa: E402
from filebase_storage import get_filebase_storage_from_config, FilebaseStorage  # noqa: E402
from project_manager import ProjectManager  # noqa: E402
from platform_utils import ensure_ffmpeg  # noqa: E402
from progress_utils import StepTracker, print_progress_bar  # noqa: E402


def ensure_python_packages(cfg=None) -> None:
    """Kiểm tra và cài đặt các package cần thiết."""
    checks = {
        "faster_whisper": "faster-whisper",
        "scenedetect": "scenedetect",
        "cv2": "opencv-python",
        "transformers": "transformers",
        "torch": "torch",
        "openai": "openai",
        "edge_tts": "edge-tts",
        "srt": "srt",
        "boto3": "boto3",
    }
    missing = []
    print(f"[deps] Đang kiểm tra {len(checks)} package...")
    for i, (module_name, pip_name) in enumerate(checks.items(), start=1):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
        print_progress_bar(i, len(checks), prefix="[deps] checking", suffix=module_name)

    if missing:
        print(f"[deps] Đang cài đặt: {missing}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=True,
        )
        print("[deps] Xong.")
    else:
        print("[deps] Tất cả package đã sẵn sàng.")


def ask_task_config(cfg) -> dict:
    """Thu thập cấu hình nội dung từ người dùng."""
    task_config = {
        "narration_pov": cfg.get("processing.narration_pov", "third_person"),
        "content_type": cfg.get("processing.content_type", "movie"),
        "genre": cfg.get("processing.genre", "drama"),
        "target_duration_sec": cfg.get("processing.target_duration_sec", 180),
    }

    interactive = sys.stdin.isatty()
    if not interactive:
        print("[task] Non-interactive: dùng giá trị mặc định từ config.toml")
        task_config["title"] = ""
        task_config["plot_summary"] = ""
        return task_config

    print("\n" + "=" * 60)
    print("  CẤU HÌNH NỘI DUNG (Enter để dùng mặc định)")
    print("=" * 60)
    task_config["title"] = input("  Tên phim/video: ").strip()
    task_config["plot_summary"] = input("  Tóm tắt cốt truyện (tuỳ chọn): ").strip()
    return task_config


def choose_hook(cfg, task_config: dict) -> str | None:
    """Sinh và chọn câu hook mở đầu."""
    from script_writer import generate_hooks

    try:
        hooks = generate_hooks(cfg, task_config, task_config.get("plot_summary", ""))
    except Exception as e:
        print(f"[hook] Lỗi khi sinh hook: {e}")
        return None

    if not hooks:
        return None

    interactive = sys.stdin.isatty()
    print("\n=== Các câu Hook mở đầu ===")
    for i, h in enumerate(hooks, start=1):
        print(f"  {i}. [{h.get('style', '')}] {h.get('text', '')}")

    if not interactive:
        chosen = hooks[0]["text"]
        print(f"\n[hook] Tự động chọn #1: {chosen}")
        return chosen

    choice = input("\n  Chọn số thứ tự hook (Enter cho #1): ").strip()
    if not choice:
        return hooks[0]["text"]
    try:
        idx = int(choice) - 1
        return hooks[idx]["text"]
    except (ValueError, IndexError):
        print("  Lựa chọn không hợp lệ, dùng #1.")
        return hooks[0]["text"]


def _list_projects_including_cloud(pm: ProjectManager, filebase: FilebaseStorage | None) -> list[dict]:
    """Liệt kê project cục bộ + cloud, để không bỏ sót project chỉ còn trên
    Filebase (vd: ổ đĩa cục bộ vừa bị xoá/mất do máy cloud khởi động lại)."""
    return pm.list_all_projects(include_cloud=filebase is not None)


def _resolve_selected_project(pm: ProjectManager, filebase, selected: dict) -> dict | None:
    """Nếu project được chọn chỉ tồn tại trên cloud (source == 'cloud_only'),
    tự động tải nó về trước khi thao tác tiếp. Trả về meta mới nhất, hoặc
    None nếu tải thất bại."""
    if selected.get("source") != "cloud_only":
        return selected

    project_id = selected["project_id"]
    print(f"\n  [project] '{project_id}' chỉ có trên cloud — đang tự động tải về...")
    ok = pm.sync_from_cloud(project_id)
    if not ok:
        print(f"  [project] Tải project '{project_id}' từ cloud thất bại.")
        return None
    return pm.get_project_status(project_id)


def run_project_menu(cfg, filebase: FilebaseStorage | None) -> None:
    """Menu quản lý project chính."""
    projects_dir = cfg.resolve_path("paths.projects_dir")
    pm = ProjectManager(projects_dir, filebase)

    while True:
        action = pm.prompt_action()

        if action == "0" or action is None:
            print("\n  Tạm biệt!")
            break

        elif action == "1":
            # Tạo project mới
            print("\n  --- Tạo Project Mới ---")
            project_id = input("  Project ID (vd: 'my-movie-v1'): ").strip()
            if not project_id:
                print("  Đã huỷ.")
                continue
            video_path = input("  Đường dẫn file video (Enter để bỏ qua): ").strip()
            title = input("  Tên project (Enter để dùng ID): ").strip()
            try:
                project_dir = pm.create_project(project_id, video_path, title)
                print(f"  Đã tạo: {project_dir}")
            except ValueError as e:
                print(f"  Lỗi: {e}")

        elif action == "2":
            # Liệt kê và tiếp tục project (bao gồm cả project chỉ có trên cloud)
            projects = _list_projects_including_cloud(pm, filebase)
            if not projects:
                print("\n  Không tìm thấy project nào (cả cục bộ lẫn cloud). Hãy tạo 1 project trước.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                selected = _resolve_selected_project(pm, filebase, selected)
                if not selected:
                    continue
                print(f"\n  Đã chọn: {selected['project_id']}")
                print(f"  Trạng thái: {selected.get('status', 'unknown')}")
                run_pipeline_on_project(cfg, pm, selected["project_id"], filebase)

        elif action == "3":
            # Liệt kê tất cả project
            projects = pm.list_all_projects(include_cloud=filebase is not None)
            print(f"\n  Tìm thấy {len(projects)} project:")
            pm.display_projects(projects)

        elif action == "4":
            # Xoá project (bao gồm cả project chỉ có trên cloud)
            projects = _list_projects_including_cloud(pm, filebase)
            if not projects:
                print("\n  Không có project nào để xoá.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                confirm = input(f"  Xoá '{selected['project_id']}'? (y/N): ").strip().lower()
                if confirm == "y":
                    if selected.get("source") == "cloud_only":
                        # Không có bản cục bộ để xoá, chỉ có thể xoá trên cloud.
                        if filebase:
                            filebase.delete_project(selected["project_id"])
                        else:
                            print("  Chưa cấu hình Filebase, không thể xoá.")
                    else:
                        cloud = input("  Xoá luôn trên cloud? (y/N): ").strip().lower() == "y"
                        pm.delete_project(selected["project_id"], cloud=cloud)

        elif action == "5":
            # Đồng bộ lên cloud
            projects = pm.scan_local_projects()
            if not projects:
                print("\n  Không có project nào để đồng bộ.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                result = pm.sync_to_cloud(selected["project_id"])
                print(f"  Kết quả đồng bộ: {result}")

        elif action == "6":
            # Tải từ cloud
            if not filebase:
                print("\n  Chưa cấu hình Filebase.")
                continue
            remote_projects = filebase.list_remote_projects()
            if not remote_projects:
                print("\n  Không tìm thấy project nào trên cloud.")
                continue
            print("\n  Project trên cloud:")
            for i, rp in enumerate(remote_projects, 1):
                print(f"    {i}. {rp['project_id']} ({rp.get('file_count', '?')} file)")
            choice = input("  Chọn số thứ tự project: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(remote_projects):
                    pid = remote_projects[idx]["project_id"]
                    pm.sync_from_cloud(pid)
            except (ValueError, IndexError):
                print("  Lựa chọn không hợp lệ.")

        elif action == "7":
            # Chạy pipeline trên project (bao gồm cả project chỉ có trên cloud)
            projects = _list_projects_including_cloud(pm, filebase)
            if not projects:
                print("\n  Không tìm thấy project nào. Hãy tạo 1 project trước.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                selected = _resolve_selected_project(pm, filebase, selected)
                if not selected:
                    continue
                run_pipeline_on_project(cfg, pm, selected["project_id"], filebase)

        else:
            print("  Lựa chọn không hợp lệ. Thử lại.")


def _scope_paths_to_project(cfg, project_dir: Path, meta: dict) -> None:
    """Trỏ paths.output_dir và paths.input_video vào đúng project_dir.

    Nếu không làm bước này, dù menu cho chọn project khác nhau, mọi stage
    (preprocess/asr/vision/...) vẫn đọc/ghi vào paths.output_dir và
    paths.input_video TOÀN CỤC trong config.toml — nghĩa là "chọn project A
    rồi chọn project B" thực chất vẫn chạy trên cùng 1 bộ input/output, chỉ
    khác nhau ở thư mục checkpoints/. Việc này khiến tính năng multi-project
    không thực sự cách ly dữ liệu.
    """
    cfg.set("paths.output_dir", str(project_dir / "output"))

    video_rel = meta.get("video_path", "")
    if video_rel:
        cfg.set("paths.input_video", str(project_dir / video_rel))
    # Nếu project chưa có video (video_rel rỗng), giữ nguyên paths.input_video
    # mặc định — preprocess.py sẽ tự hỏi người dùng nhập đường dẫn nếu đang
    # chạy tương tác (xem preprocess._resolve_input_video).


def _ensure_video_in_project(cfg, project_dir: Path, meta: dict, pm: "ProjectManager") -> None:
    """Đảm bảo video input đang dùng nằm bên trong project_dir/input/.

    Trường hợp cần: project được tạo mà chưa gán video (bỏ qua ở bước tạo),
    sau đó preprocess.py hỏi và người dùng nhập 1 đường dẫn video mới. Nếu
    không copy video đó vào project_dir/input/ và cập nhật lại meta, lần
    sync lên Filebase sau đó sẽ lại thiếu video gốc — lặp lại đúng lỗi đã
    sửa ở project_manager.create_project().
    """
    input_dir = project_dir / "input"
    input_dir.mkdir(exist_ok=True)

    current = Path(cfg.get("paths.input_video", "")).expanduser()
    if not current.exists() or not current.is_file():
        return

    try:
        current.relative_to(input_dir)
        return  # Video đã nằm trong project rồi, không cần làm gì thêm.
    except ValueError:
        pass

    dest = input_dir / current.name
    if not dest.exists():
        shutil.copy2(current, dest)
        print(f"[project] Đã copy video vào project để đảm bảo đi kèm khi sync: {dest}")

    meta["video_path"] = f"input/{current.name}"
    meta["has_input_video"] = True
    pm._save_project_meta(project_dir, meta)
    cfg.set("paths.input_video", str(dest))


def run_pipeline_on_project(cfg, pm: ProjectManager, project_id: str, filebase: FilebaseStorage | None) -> None:
    """Chạy toàn bộ pipeline trên 1 project cụ thể."""
    project_dir = pm.base_dir / project_id
    if not project_dir.exists():
        print(f"[pipeline] Không tìm thấy project '{project_id}'.")
        return

    meta = pm.get_project_status(project_id)
    print(f"\n[pipeline] Project: {project_id}")
    print(f"[pipeline] Trạng thái: {meta.get('status', 'unknown')}")
    print(f"[pipeline] Các stage đã xong: {meta.get('stages_completed', [])}")

    # Cách ly input/output theo đúng project này (xem docstring _scope_paths_to_project).
    _scope_paths_to_project(cfg, project_dir, meta)

    # Thiết lập checkpoint manager riêng cho project, hỗ trợ micro-checkpoint
    ckpt_dir = project_dir / "checkpoints"
    auto_sync_cloud = cfg.get("processing.auto_sync_cloud", True)
    ckpt = CheckpointManager(ckpt_dir, project_id=project_id, filebase_storage=filebase,
                              auto_sync_cloud=auto_sync_cloud)

    print("\n[pipeline] Trạng thái từng stage:")
    for stage, done in ckpt.status().items():
        icon = "XONG" if done else "CHƯA"
        print(f"  [{icon}] {stage}")
    print()

    import preprocess, asr, vision, semantic_graph, script_writer, tts, render

    stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
    tracker = StepTracker(stages)

    def run_stage(stage: str, compute_fn, has_checkpoint: bool = True):
        tracker.start(stage)
        if has_checkpoint and ckpt.is_done(stage):
            result = ckpt.load(stage)
            print(f"[main] Bỏ qua {stage} (đã có checkpoint).")
            tracker.finish(stage, skipped=True)
        else:
            result = compute_fn()
            tracker.finish(stage)
        return result

    preprocess_result = run_stage("preprocess", lambda: preprocess.run_preprocess(cfg, ckpt))
    # Sau preprocess, paths.input_video chắc chắn trỏ tới video hợp lệ đang
    # dùng (có thể vừa được người dùng nhập lại) -> đảm bảo nó nằm trong
    # project_dir/input/ để lần sync sau không bị thiếu.
    _ensure_video_in_project(cfg, project_dir, meta, pm)

    asr_timeline = run_stage("asr", lambda: asr.run_asr(cfg, preprocess_result, ckpt))
    vision_analysis = run_stage("vision", lambda: vision.run_vision_analysis(cfg, preprocess_result, ckpt))

    semantic_blocks = run_stage(
        "semantic_graph",
        lambda: semantic_graph.run_semantic_graph(cfg, preprocess_result, asr_timeline, vision_analysis, ckpt),
        has_checkpoint=False,
    )

    def _run_script():
        task_config = ask_task_config(cfg)
        hook = choose_hook(cfg, task_config)
        return script_writer.run_script_writer(
            cfg, task_config, semantic_blocks, asr_timeline, vision_analysis,
            hook=hook, director_brief=task_config.get("plot_summary", ""), checkpoint_mgr=ckpt,
        )

    storyboard = run_stage("script", _run_script)
    tts_result = run_stage("tts", lambda: tts.run_tts(cfg, storyboard, ckpt))
    render_result = run_stage("render", lambda: render.run_render(cfg, storyboard, tts_result, ckpt))

    # Cập nhật trạng thái project
    meta["status"] = "completed"
    meta["stages_completed"] = stages
    pm._save_project_meta(project_dir, meta)

    # Sync cuối cùng lên cloud
    if filebase and cfg.get("filebase.enabled", True):
        print("\n[filebase] Đồng bộ cuối cùng lên cloud...")
        pm.sync_to_cloud(project_id)

    print("\n" + "=" * 60)
    print("  HOÀN TẤT PIPELINE")
    print("=" * 60)
    print(f"  Video hoàn chỉnh: {render_result['final_preview_path']}")
    print(f"  Phụ đề:           {render_result['srt_path']}")
    print(f"  Kiểm tra:         {'ĐẠT' if render_result['validation_report']['passed'] else 'KHÔNG ĐẠT'}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--no-menu", action="store_true",
        help="Bỏ qua menu quản lý project, chạy thẳng pipeline trên project mặc định "
             "(non-interactive), kể cả khi đang chạy trong 1 terminal có TTY.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  AI DIRECTOR VIDEO COMMENTARY — Pipeline nâng cao")
    print("  Kèm Quản lý Project, Đồng bộ Filebase, Checkpoint Real-time")
    print("=" * 70)

    ensure_ffmpeg()

    cfg = load_config("config.toml")
    ensure_python_packages(cfg)

    # Nạp HF token
    hf_token = cfg.get("api.hf_token", "")
    if hf_token and not hf_token.startswith("PASTE_"):
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("[main] Đã nạp HF token từ config.toml.")

    # Khởi tạo Filebase
    filebase = get_filebase_storage_from_config(cfg)

    # Kiểm tra có chạy ở chế độ menu project không.
    # Chế độ non-interactive được kích hoạt bởi: cờ --no-menu (ưu tiên cao
    # nhất, đúng như README mô tả), HOẶC không có TTY (vd chạy trong script/
    # cron), HOẶC show_project_menu_on_start=false trong config.toml.
    show_menu = cfg.get("project.show_project_menu_on_start", True)
    interactive = sys.stdin.isatty() and not args.no_menu

    if interactive and show_menu:
        run_project_menu(cfg, filebase)
    else:
        # Non-interactive: chạy thẳng trên project mặc định
        print("[main] Đang chạy ở chế độ non-interactive...")
        projects_dir = cfg.resolve_path("paths.projects_dir")
        pm = ProjectManager(projects_dir, filebase)

        # Tạo project mặc định nếu chưa có
        default_id = "default"
        project_dir = pm.base_dir / default_id
        if not project_dir.exists():
            pm.create_project(default_id, cfg.get("paths.input_video", ""), "Default Project")

        run_pipeline_on_project(cfg, pm, default_id, filebase)


if __name__ == "__main__":
    main()
