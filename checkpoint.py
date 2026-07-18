"""
checkpoint.py — Enhanced checkpoint system with near real-time frequency.

Every significant operation now creates a checkpoint, allowing resume from
almost any point in the pipeline. Checkpoints are automatically synced to
cloud storage (Tigris, or any other S3-compatible provider) when configured.

Changes from original:
  - Micro-checkpoints within stages (every scene, every batch)
  - Cloud sync after each checkpoint
  - Project-aware checkpointing
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Enhanced checkpoint manager with micro-checkpoints and cloud sync."""

    def __init__(self, checkpoint_dir: str | Path, project_id: str = "", cloud_storage=None,
                 auto_sync_cloud: bool = True, auto_save_interval: int = 0):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self.cloud = cloud_storage
        # auto_sync_cloud: đọc từ config processing.auto_sync_cloud. Nếu False,
        # KHÔNG tự động sync checkpoint lên cloud sau mỗi lần lưu (người
        # dùng có thể vẫn chủ động sync tay qua menu "5. Đồng bộ lên cloud").
        self.auto_sync_cloud = auto_sync_cloud
        # auto_save_interval: đọc từ project.auto_save_interval trong config.toml
        # (giây). 0 (mặc định) = giữ nguyên hành vi cũ: throttle sync
        # micro-checkpoint lên cloud theo SỐ LẦN gọi (mỗi 5 lần, xem
        # save_micro). Nếu > 0, chuyển throttle sang theo THỜI GIAN: sync mỗi
        # khi đã trôi qua ít nhất auto_save_interval giây kể từ lần sync gần
        # nhất, bất kể số lần save_micro() đã gọi.
        self.auto_save_interval = max(0, auto_save_interval)
        self._save_count = 0
        # Đếm riêng số lần save_micro() được gọi, dùng cho auto_save_interval
        # (throttle THEO THỜI GIAN, tuỳ chọn — xem save_micro()).
        self._micro_save_count = 0
        self._last_save_time = time.time()

        # BUGFIX (nghiêm trọng): trước đây _sync_to_cloud() nuốt luôn kết quả
        # thất bại (không retry, không log lại), và save_micro() chỉ sync
        # theo bội số cố định "mỗi 5 lần gọi" bất kể mỗi item tốn bao nhiêu
        # công GPU/API thật để tạo ra. Hậu quả thực tế: 1 phiên vision chạy
        # xong 689/1609 scene cục bộ, nhưng cloud chỉ có 273/1609 — khi
        # Colab bị ngắt (ổ đĩa /content là bộ nhớ tạm, mất sạch khi restart),
        # 416 scene đã tốn API/GPU bị mất trắng vì chưa kịp rơi vào bội số
        # sync hoặc vì 1 lần upload lỗi không bao giờ được thử lại.
        #
        # Sửa: mỗi micro-checkpoint giờ LUÔN được thử sync ngay (không throttle
        # theo số lần mặc định nữa — upload 1 JSON vài KB rẻ hơn rất nhiều so
        # với rủi ro mất công đã tốn để tạo ra nó). Nếu 1 lần sync thất bại,
        # item đó được ghi vào _pending_syncs và sẽ được TỰ ĐỘNG thử lại ở lần
        # save_micro() kế tiếp, và bắt buộc phải "trả hết nợ" khi
        # flush_pending_syncs() được gọi ở cuối mỗi stage.
        self._pending_syncs: dict[str, Path] = {}  # remote_relative -> local_path
        # BUGFIX: save_micro()/_sync_to_cloud() không thread-safe trước đây vì
        # luôn được gọi TUẦN TỰ từ 1 coroutine/luồng duy nhất. Từ khi tts.py
        # chạy song song nhiều luồng qua asyncio.to_thread(), nhiều luồng có
        # thể gọi save_micro() cùng lúc -> lock để tránh 2 luồng cùng
        # đọc/ghi self._pending_syncs / bộ đếm cùng lúc (race condition).
        self._lock = threading.Lock()

        # Register signal handlers for graceful shutdown
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Save emergency checkpoint on SIGINT/SIGTERM."""
        print(f"\n[checkpoint] Received signal {signum}, saving emergency checkpoint...")
        self.save_emergency()
        # Restore original signal handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        sys.exit(1)

    def _path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def _micro_path(self, stage: str, item_id: str) -> Path:
        return self.dir / f"{stage}_{item_id}.json"

    def is_done(self, stage: str) -> bool:
        """Check if a stage is completed."""
        p = self._path(stage)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("_done", False))
        except (json.JSONDecodeError, OSError):
            return False

    def is_micro_done(self, stage: str, item_id: str) -> bool:
        """Check if a micro-checkpoint is done."""
        p = self._micro_path(stage, item_id)
        return p.exists()

    def list_micro_done(self, stage: str) -> set[str]:
        """Trả về tập hợp item_id đã có micro-checkpoint cho 1 stage.

        Dùng để resume: gọi hàm này TRƯỚC khi chạy lại 1 stage, để biết
        item nào (scene, clip...) đã xong và có thể bỏ qua, thay vì luôn
        chạy lại từ đầu dù micro-checkpoint đã được ghi.
        """
        prefix = f"{stage}_"
        done = set()
        for p in self.dir.glob(f"{prefix}*.json"):
            if p.name.endswith(".json.tmp"):
                continue
            item_id = p.stem[len(prefix):]
            done.add(item_id)
        return done

    def force_sync_micro(self, stage: str, item_id: str) -> None:
        """Ép sync 1 micro-checkpoint lên cloud ngay, bỏ qua throttle.

        Dùng ở item cuối cùng của vòng lặp để đảm bảo item cuối luôn được
        đẩy lên cloud dù chưa rơi đúng vào bội số của chu kỳ throttle.
        """
        p = self._micro_path(stage, item_id)
        if p.exists():
            self._retry_pending_syncs()
            self._sync_to_cloud(p, f"checkpoints/{stage}_{item_id}.json")

    def save(self, stage: str, payload: Any) -> None:
        """Save stage checkpoint and sync to cloud."""
        p = self._path(stage)
        wrapper = {
            "_done": True,
            "_stage": stage,
            "_project_id": self.project_id,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_save_count": self._save_count,
            "data": payload,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

        self._save_count += 1
        self._last_save_time = time.time()
        print(f"[checkpoint] Saved stage '{stage}' -> {p.name}")

        # Sync to cloud
        self._sync_to_cloud(p, f"checkpoints/{stage}.json")

    def save_micro(self, stage: str, item_id: str, payload: Any) -> None:
        """Save micro-checkpoint within a stage (per-scene, per-batch)."""
        p = self._micro_path(stage, item_id)
        wrapper = {
            "_done": True,
            "_stage": stage,
            "_item_id": item_id,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": payload,
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

        with self._lock:
            self._micro_save_count += 1

        # Trước khi làm gì mới, luôn cố "trả nợ" các lần sync đã thất bại
        # trước đó -- nếu không làm bước này, 1 lần lỗi mạng thoáng qua sẽ
        # khiến item đó biến mất khỏi cloud VĨNH VIỄN (bug cũ).
        self._retry_pending_syncs()

        if self.auto_save_interval > 0:
            # Chế độ throttle THEO THỜI GIAN (tuỳ chọn, đọc từ
            # project.auto_save_interval trong config.toml): chỉ dùng khi
            # người dùng chủ động muốn giảm tần suất round-trip mạng cho các
            # item RẺ (vd rất nhiều item nhỏ trong thời gian ngắn). Ngay cả
            # khi bỏ qua sync ở đây, item vẫn đã có trong _pending_syncs nếu
            # có sự cố, và sẽ được flush_pending_syncs() dọn sạch cuối stage.
            with self._lock:
                should_sync_now = (time.time() - self._last_save_time) >= self.auto_save_interval
                if not should_sync_now:
                    self._pending_syncs[f"checkpoints/{stage}_{item_id}.json"] = p
                    return

        ok = self._sync_to_cloud(p, f"checkpoints/{stage}_{item_id}.json")
        if ok:
            with self._lock:
                self._last_save_time = time.time()

    def load(self, stage: str) -> Any:
        """Load stage checkpoint data."""
        p = self._path(stage)
        with open(p, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        return wrapper["data"]

    def load_micro(self, stage: str, item_id: str) -> Any:
        """Load micro-checkpoint data."""
        p = self._micro_path(stage, item_id)
        with open(p, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        return wrapper["data"]

    def clear(self, stage: str | None = None) -> None:
        """Clear checkpoint(s)."""
        if stage is None:
            for p in self.dir.glob("*.json"):
                p.unlink()
            print("[checkpoint] Cleared all checkpoints.")
        else:
            # Clear main + micro + partial
            for p in self.dir.glob(f"{stage}*.json"):
                p.unlink()
            print(f"[checkpoint] Cleared checkpoints for '{stage}'.")

    def clear_all_for_project(self) -> None:
        """Clear all checkpoints for this project."""
        self.clear()

    def status(self) -> dict[str, bool]:
        """Return status of all known stages."""
        known_stages = ["preprocess", "asr", "vision", "semantic_graph", "script", "tts", "render"]
        return {s: self.is_done(s) for s in known_stages}

    def save_emergency(self) -> None:
        """Save emergency marker so we know where we stopped."""
        p = self.dir / "_emergency_stop.json"
        wrapper = {
            "_emergency": True,
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_completed_stages": list(self.status().items()),
        }
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
        print("[checkpoint] Emergency checkpoint saved.")

    def _sync_to_cloud(self, local_path: Path, remote_relative: str) -> bool:
        """Sync a single file to cloud storage.

        BUGFIX: trước đây hàm này nuốt luôn kết quả (kể cả thất bại) và
        không bao giờ báo cho save_micro() biết để thử lại -> 1 lần lỗi mạng
        là mất checkpoint đó trên cloud vĩnh viễn dù file cục bộ vẫn còn.
        Giờ trả về True/False thật, và tự ghi nhận vào self._pending_syncs
        khi thất bại để lần gọi sau (hoặc flush_pending_syncs() cuối stage)
        có thể thử lại.
        """
        if self.cloud is None or not self.auto_sync_cloud:
            return True  # không cấu hình cloud -> không có gì để "nợ"
        ok = False
        try:
            remote_key = f"projects/{self.project_id}/{remote_relative}"
            # Gọi mạng THẬT SỰ nằm ngoài lock -- nếu khoá ở đây, 40 luồng tts
            # song song sẽ bị dồn về chạy tuần tự ngay tại bước upload, phá
            # hỏng mục đích chạy song song. Lock chỉ cần bảo vệ chỗ đọc/ghi
            # self._pending_syncs (thao tác dict, rất nhanh) bên dưới.
            ok = bool(self.cloud._upload_file(local_path, remote_key))
        except Exception:
            ok = False
        with self._lock:
            if ok:
                self._pending_syncs.pop(remote_relative, None)
            else:
                self._pending_syncs[remote_relative] = local_path
        return ok

    def _retry_pending_syncs(self) -> None:
        """Thử lại MỌI lần sync trước đó đã thất bại.

        Gọi ở đầu mỗi save_micro() (rẻ: dict rỗng thì no-op ngay) để các lỗi
        mạng thoáng qua tự phục hồi càng sớm càng tốt, không phải đợi tới
        cuối stage mới phát hiện ra là đang "nợ" cloud.
        """
        if self.cloud is None or not self.auto_sync_cloud:
            return
        with self._lock:
            snapshot = list(self._pending_syncs.items())
        if not snapshot:
            return
        for remote_relative, local_path in snapshot:
            if not local_path.exists():
                # File cục bộ không còn (hiếm khi xảy ra) -> bỏ khỏi hàng đợi,
                # không có gì để upload lại nữa.
                with self._lock:
                    self._pending_syncs.pop(remote_relative, None)
                continue
            self._sync_to_cloud(local_path, remote_relative)

    def flush_pending_syncs(self, max_retries: int = 5, retry_wait_sec: float = 3.0) -> bool:
        """Ép mọi checkpoint còn đang 'nợ' cloud phải lên cloud trước khi coi
        1 stage là an toàn để kết thúc.

        BẮT BUỘC gọi hàm này ở cuối mỗi stage có dùng save_micro() (vision,
        tts, render, preprocess) — đây chính là lưới an toàn cuối cùng đảm
        bảo "đồng bộ chính xác 100%": dù throttle/lỗi mạng có làm trễ vài
        lần trong lúc chạy, cuối stage KHÔNG ĐƯỢC còn item nào sót lại chỉ
        tồn tại cục bộ.

        Trả về True nếu mọi thứ đã lên cloud sạch sẽ, False nếu vẫn còn sót
        (vd mất mạng hẳn) — khi đó pipeline nên in cảnh báo rõ ràng cho
        người dùng thay vì im lặng như trước.
        """
        if self.cloud is None or not self.auto_sync_cloud:
            return True
        for attempt in range(1, max_retries + 1):
            self._retry_pending_syncs()
            with self._lock:
                still_pending = bool(self._pending_syncs)
            if not still_pending:
                return True
            if attempt < max_retries:
                time.sleep(retry_wait_sec)
        with self._lock:
            remaining = len(self._pending_syncs)
        print(f"[checkpoint] CẢNH BÁO: {remaining} checkpoint vẫn CHƯA lên được cloud sau "
              f"{max_retries} lần thử (có thể do mất mạng). Dữ liệu vẫn AN TOÀN trên đĩa cục bộ, "
              f"nhưng sẽ mất nếu máy/Colab bị reset trước khi bạn chạy lại hoặc bấm "
              f"'5. Đồng bộ project lên cloud' để thử lại.")
        return False

    def sync_all_to_cloud(self, force: bool = False) -> dict[str, int]:
        """Sync all checkpoint files to cloud storage.

        force=True bỏ qua cờ auto_sync_cloud (dùng khi người dùng chủ động
        bấm "đồng bộ" từ menu, chứ không phải auto-sync ngầm)."""
        if self.cloud is None:
            return {"uploaded": 0, "errors": 0}
        if not self.auto_sync_cloud and not force:
            return {"uploaded": 0, "errors": 0}

        uploaded = 0
        errors = 0
        for p in self.dir.glob("*.json"):
            remote_key = f"projects/{self.project_id}/checkpoints/{p.name}"
            ok = self.cloud._upload_file(p, remote_key)
            if ok:
                uploaded += 1
            else:
                errors += 1

        print(f"[checkpoint] Cloud sync: {uploaded} uploaded, {errors} errors")
        return {"uploaded": uploaded, "errors": errors}
