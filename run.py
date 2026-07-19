#!/usr/bin/env python3
"""
run.py — Entry point nâng cao với quản lý project, sync cloud (Tigris),
checkpoint gần như real-time, và UX cải tiến.

Tính năng:
  - Menu project thông minh: tạo, liệt kê, tiếp tục, xoá, đồng bộ
  - Checkpoint gần như real-time (mỗi scene/batch)
  - Tự động sync lên cloud storage (Tigris, hoặc provider S3-compatible khác)
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
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config  # noqa: E402
from checkpoint import CheckpointManager  # noqa: E402
from cloud_storage import get_cloud_storage_from_config, CloudStorage  # noqa: E402
from project_manager import ProjectManager  # noqa: E402
from platform_utils import ensure_ffmpeg  # noqa: E402
from progress_utils import StepTracker, print_progress_bar  # noqa: E402
import reference_video  # noqa: E402


def _checkpoint_subdir_name(cfg) -> str:
    """Tên thư mục con chứa checkpoint bên trong mỗi project_dir, đọc từ
    paths.checkpoint_dir trong config.toml (mặc định "checkpoints"). Dùng
    CHUNG cho cả CheckpointManager (run_pipeline_on_project) và
    ProjectManager (quét trạng thái project) để 2 nơi không bao giờ lệch tên
    thư mục với nhau."""
    return Path(cfg.get("paths.checkpoint_dir", "./checkpoints")).name or "checkpoints"


def ensure_python_packages(cfg=None) -> None:
    """Kiểm tra và cài đặt các package cần thiết."""
    checks = {
        "faster_whisper": "faster-whisper",
        "scenedetect": "scenedetect",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "transformers": "transformers",
        "torch": "torch",
        "openai": "openai",
        "edge_tts": "edge-tts",
        "srt": "srt",
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

    # rclone là 1 BINARY (dùng bởi cloud_storage.py), không phải gói pip
    # nên không nằm trong `checks` ở trên và KHÔNG tự cài được qua pip
    # install -- chỉ kiểm tra và nhắc người dùng cài tay nếu thiếu. Thiếu
    # rclone KHÔNG chặn pipeline chạy, chỉ tắt tính năng cloud sync (xem
    # get_cloud_storage_from_config trong cloud_storage.py).
    if shutil.which("rclone") is None:
        print(
            "[deps] CẢNH BÁO: chưa cài rclone -> tính năng cloud sync sẽ bị tắt. "
            "Cài: curl https://rclone.org/install.sh | sudo bash "
            "(xem thêm https://rclone.org/downloads/)"
        )


def ask_task_config(cfg, project_reference_urls: list[str] | None = None) -> dict:
    """Thu thập cấu hình nội dung từ người dùng.

    project_reference_urls: link đối thủ đã được nhập RIÊNG cho project này
    lúc tạo project (xem menu "1. Tạo project mới"). Nếu đã có, dùng luôn,
    KHÔNG hỏi lại và KHÔNG rơi về danh sách chung trong config.toml — vì mỗi
    project/video thường có đối thủ khác nhau.
    """
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
        task_config["reference_urls"] = project_reference_urls or cfg.get("reference.urls", [])
        return task_config

    print("\n" + "=" * 60)
    print("  CẤU HÌNH NỘI DUNG (Enter để dùng mặc định)")
    print("=" * 60)
    task_config["title"] = input("  Tên phim/video: ").strip()
    task_config["plot_summary"] = input("  Tóm tắt cốt truyện (tuỳ chọn): ").strip()
    if project_reference_urls:
        # Đã nhập lúc tạo project -> dùng luôn, không hỏi lại lần nữa.
        print(f"  Link đối thủ: dùng {len(project_reference_urls)} link đã nhập lúc tạo project.")
        task_config["reference_urls"] = project_reference_urls
        return task_config
    ref_input = input(
        "  Link video tham khảo (đối thủ, cách nhau bởi dấu phẩy, tuỳ chọn): "
    ).strip()
    if ref_input:
        task_config["reference_urls"] = [u.strip() for u in ref_input.split(",") if u.strip()]
    else:
        task_config["reference_urls"] = cfg.get("reference.urls", [])
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


def _list_projects_including_cloud(pm: ProjectManager, cloud: CloudStorage | None) -> list[dict]:
    """Liệt kê project cục bộ + cloud, để không bỏ sót project chỉ còn trên
    cloud (vd: ổ đĩa cục bộ vừa bị xoá/mất do máy cloud khởi động lại)."""
    return pm.list_all_projects(include_cloud=cloud is not None)


def _resolve_selected_project(pm: ProjectManager, cloud, selected: dict) -> dict | None:
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


def run_project_menu(cfg, cloud: CloudStorage | None) -> None:
    """Menu quản lý project chính."""
    projects_dir = cfg.resolve_path("paths.projects_dir")
    pm = ProjectManager(projects_dir, cloud, checkpoint_subdir=_checkpoint_subdir_name(cfg))

    # auto_project_scan (project.auto_project_scan trong config.toml, mặc định
    # true): tự động quét + hiện nhanh số lượng project ngay khi vào menu,
    # thay vì bắt người dùng phải chọn "3. Liệt kê" trước mới biết có gì.
    # Trước đây key này hoàn toàn không được đọc ở đâu.
    if cfg.get("project.auto_project_scan", True):
        try:
            projects = _list_projects_including_cloud(pm, cloud)
            print(f"\n  [project] Đã quét: {len(projects)} project tìm thấy "
                  f"(cục bộ + cloud).")
        except Exception as e:
            print(f"\n  [project] CẢNH BÁO: quét project tự động thất bại: {e}")

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
            ref_input = input(
                "  Link video tham khảo (đối thủ, cách nhau bởi dấu phẩy, tuỳ chọn): "
            ).strip()
            reference_urls = [u.strip() for u in ref_input.split(",") if u.strip()]
            try:
                project_dir = pm.create_project(project_id, video_path, title, reference_urls)
                print(f"  Đã tạo: {project_dir}")
                # Đẩy ngay lên cloud khi vừa tạo — không bắt người dùng
                # phải nhớ vào lại menu "5. Đồng bộ" mới có project trên
                # cloud. Nếu chưa cấu hình cloud thì bỏ qua im lặng
                # (sync_to_cloud tự trả lỗi rõ ràng trong trường hợp đó).
                if cloud:
                    print("  [cloud] Đang đẩy project mới lên cloud...")
                    result = pm.sync_to_cloud(project_id)
                    if result.get("error") or result.get("aborted"):
                        print(f"  [cloud] CẢNH BÁO: đẩy lên cloud thất bại: {result}")
                    else:
                        print(f"  [cloud] Đã có trên cloud: {result['uploaded']} file tải lên.")
                else:
                    print("  [cloud] Chưa cấu hình cloud storage — project chỉ đang ở local. "
                          "Điền access_key/secret_key trong config.toml để tự động đẩy lên cloud.")
            except ValueError as e:
                print(f"  Lỗi: {e}")

        elif action == "2":
            # Liệt kê và tiếp tục project (bao gồm cả project chỉ có trên cloud)
            projects = _list_projects_including_cloud(pm, cloud)
            if not projects:
                print("\n  Không tìm thấy project nào (cả cục bộ lẫn cloud). Hãy tạo 1 project trước.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                selected = _resolve_selected_project(pm, cloud, selected)
                if not selected:
                    continue
                print(f"\n  Đã chọn: {selected['project_id']}")
                print(f"  Trạng thái: {selected.get('status', 'unknown')}")
                run_pipeline_on_project(cfg, pm, selected["project_id"], cloud)

        elif action == "3":
            # Liệt kê tất cả project
            projects = pm.list_all_projects(include_cloud=cloud is not None)
            print(f"\n  Tìm thấy {len(projects)} project:")
            pm.display_projects(projects)

        elif action == "4":
            # Xoá project (bao gồm cả project chỉ có trên cloud)
            projects = _list_projects_including_cloud(pm, cloud)
            if not projects:
                print("\n  Không có project nào để xoá.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                confirm = input(f"  Xoá '{selected['project_id']}'? (y/N): ").strip().lower()
                if confirm == "y":
                    if selected.get("source") == "cloud_only":
                        # Không có bản cục bộ để xoá, chỉ có thể xoá trên cloud.
                        if cloud:
                            cloud.delete_project(selected["project_id"])
                        else:
                            print("  Chưa cấu hình cloud storage, không thể xoá.")
                    else:
                        delete_on_cloud = input("  Xoá luôn trên cloud? (y/N): ").strip().lower() == "y"
                        pm.delete_project(selected["project_id"], cloud=delete_on_cloud)

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
            if not cloud:
                print("\n  Chưa cấu hình cloud storage.")
                continue
            remote_projects = cloud.list_remote_projects()
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
            projects = _list_projects_including_cloud(pm, cloud)
            if not projects:
                print("\n  Không tìm thấy project nào. Hãy tạo 1 project trước.")
                continue
            selected = pm.prompt_select_project(projects)
            if selected:
                selected = _resolve_selected_project(pm, cloud, selected)
                if not selected:
                    continue
                run_pipeline_on_project(cfg, pm, selected["project_id"], cloud)

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
    sync lên cloud sau đó sẽ lại thiếu video gốc — lặp lại đúng lỗi đã
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
    if dest.exists() and dest.stat().st_size == current.stat().st_size:
        # Trùng tên VÀ trùng kích thước -> coi như cùng 1 file, không copy lại.
        # (So sánh size là kiểm tra rẻ; không phải hash đầy đủ, nhưng đủ để
        # tránh trường hợp phổ biến nhất: chạy lại pipeline trên đúng video cũ.)
        pass
    elif dest.exists():
        # Trùng tên nhưng KHÁC kích thước -> đây là 1 video MỚI, không được
        # âm thầm dùng file cũ trong input/. Đặt tên khác để không mất dữ
        # liệu và đảm bảo pipeline dùng đúng video người dùng vừa cung cấp.
        stem, suffix = current.stem, current.suffix
        i = 1
        while (input_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        dest = input_dir / f"{stem}_{i}{suffix}"
        shutil.copy2(current, dest)
        print(f"[project] CẢNH BÁO: input/ đã có file trùng tên nhưng khác nội dung — "
              f"đã copy video mới vào project với tên khác để không dùng nhầm file cũ: {dest}")
    else:
        shutil.copy2(current, dest)
        print(f"[project] Đã copy video vào project để đảm bảo đi kèm khi sync: {dest}")

    meta["video_path"] = f"input/{dest.name}"
    meta["has_input_video"] = True
    pm._save_project_meta(project_dir, meta)
    cfg.set("paths.input_video", str(dest))


def run_pipeline_on_project(cfg, pm: ProjectManager, project_id: str, cloud: CloudStorage | None) -> None:
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

    # Thiết lập checkpoint manager riêng cho project, hỗ trợ micro-checkpoint.
    # Thư mục checkpoint LUÔN nằm trong project_dir (bắt buộc cho cách ly
    # multi-project — xem _scope_paths_to_project ở trên), nhưng TÊN thư mục
    # con vẫn tôn trọng paths.checkpoint_dir trong config.toml thay vì hard-code
    # "checkpoints" (trước đây key này hoàn toàn bị bỏ qua).
    ckpt_subdir_name = _checkpoint_subdir_name(cfg)
    ckpt_dir = project_dir / ckpt_subdir_name
    auto_sync_cloud = cfg.get("processing.auto_sync_cloud", True)
    auto_save_interval = cfg.get("project.auto_save_interval", 0)
    ckpt = CheckpointManager(ckpt_dir, project_id=project_id, cloud_storage=cloud,
                              auto_sync_cloud=auto_sync_cloud, auto_save_interval=auto_save_interval)

    print("\n[pipeline] Trạng thái từng stage:")
    for stage, done in ckpt.status().items():
        icon = "XONG" if done else "CHƯA"
        print(f"  [{icon}] {stage}")
    print()

    import preprocess, asr, vision, semantic_graph, script_writer, tts, render

    stages = ["preprocess", "asr", "vision", "semantic_graph", "reference", "script", "tts", "render"]
    tracker = StepTracker(stages)

    # BUGFIX: checkpoint JSON được tự động đẩy lên cloud gần như real-time
    # (xem checkpoint.py:_sync_to_cloud), NHƯNG file output thật của stage đó
    # (audio.wav, vision_analysis.json, voiceover.mp3, final_preview.mp4...)
    # trước đây chỉ được đồng bộ lên cloud khi người dùng bấm menu "5. Đồng bộ"
    # hoặc khi TOÀN BỘ pipeline chạy xong. Nếu phiên chạy bị ngắt giữa chừng
    # (mất mạng, hết token, máy cloud restart...) sau khi 1 stage xong nhưng
    # trước khi kịp đồng bộ, cloud sẽ có checkpoint nói "đã xong" trong khi
    # KHÔNG có file thật đi kèm -> tải project về máy khác sẽ crash ngay khi
    # đọc file không tồn tại (đã xảy ra thật với audio.wav).
    #
    # Danh sách file output "bắt buộc phải có" cho mỗi stage, dùng để:
    #   1) không tin checkpoint "đã xong" nếu file thật đã biến mất -> tự
    #      chạy lại stage đó thay vì crash ở stage sau.
    #   2) biết cần đồng bộ project lên cloud sau khi stage nào đó chạy THẬT
    #      (không phải bị skip), để checkpoint trên cloud luôn đi kèm dữ liệu
    #      thật, không bao giờ "nói dối" nữa.
    required_outputs = {
        "preprocess": ["output/pipeline/audio.wav", "output/pipeline/scenes.json"],
        "asr": ["output/pipeline/asr_timeline.json"],
        "vision": ["output/pipeline/vision_analysis.json"],
        "semantic_graph": ["output/pipeline/semantic_blocks.json"],
        "reference": ["output/pipeline/reference_brief.json"],
        "script": ["output/pipeline/storyboard.json"],
        "tts": ["output/pipeline/voiceover.mp3"],
        "render": ["output/deliverables/final_preview.mp4", "output/deliverables/narration_subtitle.srt"],
    }

    def _missing_outputs(stage: str) -> list[str]:
        return [rel for rel in required_outputs.get(stage, []) if not (project_dir / rel).exists()]

    def run_stage(stage: str, compute_fn, has_checkpoint: bool = True):
        tracker.start(stage)
        missing = _missing_outputs(stage) if has_checkpoint else []
        if has_checkpoint and ckpt.is_done(stage) and not missing:
            result = ckpt.load(stage)
            print(f"[main] Bỏ qua {stage} (đã có checkpoint).")
            tracker.finish(stage, skipped=True)
        else:
            if has_checkpoint and ckpt.is_done(stage) and missing:
                print(f"[main] CẢNH BÁO: checkpoint '{stage}' nói đã xong nhưng thiếu file "
                      f"output thật ({', '.join(missing)}) — chạy lại stage này.")
            result = compute_fn()
            tracker.finish(stage)
            # Đồng bộ NGAY project thật (không chỉ checkpoint JSON) lên cloud
            # sau mỗi stage chạy thật, để cloud không bao giờ có checkpoint
            # "mồ côi" (không có file thật đi kèm) như đã xảy ra trước đây.
            if cloud and cfg.get("cloud.enabled", True) and auto_sync_cloud:
                print(f"[cloud] Đồng bộ output của '{stage}' lên cloud...")
                sync_result = pm.sync_to_cloud(project_id)
                if sync_result.get("error") or sync_result.get("aborted"):
                    print(f"[cloud] CẢNH BÁO: đồng bộ '{stage}' lên cloud thất bại: {sync_result}")
        return result

    preprocess_result = run_stage("preprocess", lambda: preprocess.run_preprocess(cfg, ckpt))
    # Sau preprocess, paths.input_video chắc chắn trỏ tới video hợp lệ đang
    # dùng (có thể vừa được người dùng nhập lại) -> đảm bảo nó nằm trong
    # project_dir/input/ để lần sync sau không bị thiếu.
    _ensure_video_in_project(cfg, project_dir, meta, pm)
    # _ensure_video_in_project() có thể vừa copy 1 video mới vào input/ SAU
    # khi run_stage("preprocess", ...) đã đồng bộ lên cloud lần đầu (nếu
    # preprocess chạy thật) -> đồng bộ thêm 1 lần nữa để đảm bảo video luôn
    # đi kèm, không phải chờ tới stage kế tiếp chạy thật mới được đồng bộ.
    if cloud and cfg.get("cloud.enabled", True) and auto_sync_cloud:
        pm.sync_to_cloud(project_id)

    asr_timeline = run_stage("asr", lambda: asr.run_asr(cfg, preprocess_result, ckpt))
    vision_analysis = run_stage("vision", lambda: vision.run_vision_analysis(cfg, preprocess_result, ckpt))

    semantic_blocks = run_stage(
        "semantic_graph",
        lambda: semantic_graph.run_semantic_graph(cfg, preprocess_result, asr_timeline, vision_analysis, ckpt),
        has_checkpoint=False,
    )

    # task_config chỉ nên hỏi người dùng 1 LẦN dù được dùng ở cả 2 stage
    # ("reference" và "script") — nếu 1 trong 2 đã có checkpoint và bị skip,
    # ask_task_config() sẽ không được gọi lần nào; nếu cả 2 đều chạy thật,
    # cache dưới đây đảm bảo chỉ hỏi 1 lần.
    _task_config_cache: dict[str, dict] = {}

    def _get_task_config() -> dict:
        if "value" not in _task_config_cache:
            _task_config_cache["value"] = ask_task_config(cfg, meta.get("reference_urls"))
        return _task_config_cache["value"]

    def _run_reference():
        task_config = _get_task_config()
        return reference_video.run_reference_stage(cfg, task_config, checkpoint_mgr=ckpt)

    reference_result = run_stage("reference", _run_reference)

    def _run_script():
        task_config = _get_task_config()
        hook = choose_hook(cfg, task_config)

        director_brief = task_config.get("plot_summary", "")
        ref_brief = (reference_result or {}).get("combined_brief", "")
        ref_note = (reference_result or {}).get("note", "")
        if ref_brief:
            # Gộp tóm tắt tự nhập (nếu có) + transcript tham khảo + cảnh báo
            # chống đạo văn thành 1 director_brief duy nhất cho script_writer.
            pieces = [p for p in [director_brief, ref_brief, ref_note] if p]
            director_brief = "\n\n".join(pieces)

        return script_writer.run_script_writer(
            cfg, task_config, semantic_blocks, asr_timeline, vision_analysis,
            hook=hook, director_brief=director_brief, checkpoint_mgr=ckpt,
        )

    storyboard = run_stage("script", _run_script)
    tts_result = run_stage("tts", lambda: tts.run_tts(cfg, storyboard, ckpt))
    render_result = run_stage("render", lambda: render.run_render(cfg, storyboard, tts_result, ckpt))

    # Cập nhật trạng thái project
    meta["status"] = "completed"
    meta["stages_completed"] = stages
    pm._save_project_meta(project_dir, meta)

    # Sync cuối cùng lên cloud
    if cloud and cfg.get("cloud.enabled", True):
        print("\n[cloud] Đồng bộ cuối cùng lên cloud...")
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
    print("  Kèm Quản lý Project, Đồng bộ Cloud (Tigris), Checkpoint Real-time")
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

    # Khởi tạo cloud storage
    cloud = get_cloud_storage_from_config(cfg)

    # Kiểm tra có chạy ở chế độ menu project không.
    # Chế độ non-interactive được kích hoạt bởi: cờ --no-menu (ưu tiên cao
    # nhất, đúng như README mô tả), HOẶC không có TTY (vd chạy trong script/
    # cron), HOẶC show_project_menu_on_start=false trong config.toml.
    show_menu = cfg.get("project.show_project_menu_on_start", True)
    interactive = sys.stdin.isatty() and not args.no_menu

    if interactive and show_menu:
        run_project_menu(cfg, cloud)
    else:
        # Non-interactive: chạy thẳng trên project mặc định
        print("[main] Đang chạy ở chế độ non-interactive...")
        projects_dir = cfg.resolve_path("paths.projects_dir")
        pm = ProjectManager(projects_dir, cloud, checkpoint_subdir=_checkpoint_subdir_name(cfg))

        # Tạo project mặc định nếu chưa có
        default_id = "default"
        project_dir = pm.base_dir / default_id
        if not project_dir.exists():
            pm.create_project(default_id, cfg.get("paths.input_video", ""), "Default Project")

        run_pipeline_on_project(cfg, pm, default_id, cloud)


if __name__ == "__main__":
    main()
