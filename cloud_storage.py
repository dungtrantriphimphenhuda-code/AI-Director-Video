"""
cloud_storage.py — Tích hợp lưu trữ cloud tương thích S3 (hiện dùng Tigris),
qua rclone.

TẠI SAO ĐỔI TỪ boto3 SANG rclone (2026-07):
Bản boto3 cũ (~900 dòng) phải tự tay xử lý mọi thứ: retry, backoff,
multipart, giới hạn số connection song song, và một bug deadlock nội bộ
của s3transfer khiến job "treo hàng giờ" mà không ném exception nào cả —
phải chống bằng 1 lớp watchdog thread tự chế khá phức tạp và vẫn rò rỉ
thread khi gặp ca xấu nhất. rclone là 1 binary CLI được cả cộng đồng dùng
nhiều năm để sync dữ liệu lớn lên S3-compatible storage: retry, backoff,
multipart, giới hạn số connection song song, và timeout theo "không còn
tiến triển" (idle timeout, tự reset mỗi khi có byte mới, không giết oan
job đang tải file nhiều GB) đều đã được xử lý bên trong rclone, rất
trưởng thành và ổn định. Module này giờ chỉ còn là 1 lớp mỏng gọi rclone
qua subprocess — ít code tự viết hơn nhiều => ít "lỗi vặt" hơn.

Logic hoàn toàn generic nên vẫn dùng được với Filebase, AWS S3,
Cloudflare R2, hay bất kỳ provider tương thích S3 nào khác — chỉ cần đổi
endpoint_url/region_name/addressing_style trong config.toml (mục [cloud]),
giống hệt như trước.

YÊU CẦU: cần cài rclone (không phải gói pip) — https://rclone.org/downloads/
  Linux/macOS: curl https://rclone.org/install.sh | sudo bash
  Windows: xem hướng dẫn tại https://rclone.org/downloads/

Tigris endpoint: https://t3.storage.dev
Giao thức: tương thích S3 API
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

RCLONE_BIN = shutil.which("rclone")
HAS_RCLONE = RCLONE_BIN is not None

# Tên remote rclone dùng nội bộ — KHÔNG ghi ra file rclone.conf nào trên
# đĩa, toàn bộ cấu hình (access_key, secret_key, endpoint...) được truyền
# qua biến môi trường RCLONE_CONFIG_<TÊN>_<KEY> cho từng tiến trình con,
# nên không đụng tới (và không bị đụng bởi) rclone.conf thật của người
# dùng nếu họ có sẵn 1 cái, cũng như trước đây secret chỉ nằm trong
# config.toml chứ không ghi thêm ra chỗ khác.
_REMOTE_NAME = "aidirectorcloud"

# Ngưỡng thời gian chờ (giây) áp cho MỖI lần gọi rclone cho 1 file ĐƠN LẺ
# (dùng bởi _upload_file, gọi rất nhiều lần bởi checkpoint.py, có lúc tới
# hàng chục lần song song từ tts.py). Đây CHỈ là lưới an toàn ở tầng
# Python, phòng trường hợp hiếm bản thân tiến trình rclone bị treo hẳn —
# khác với watchdog cũ, subprocess.run(timeout=...) có thể KILL tiến
# trình con 1 cách sạch sẽ (không rò rỉ thread) khi hết giờ.
# Không áp dụng cho upload_project()/download_project() (file/thư mục lớn,
# chạy lâu là bình thường) — 2 hàm đó để rclone tự quản lý qua --timeout
# (idle timeout, tự reset mỗi khi vẫn còn nhận/gửi byte).
_SINGLE_FILE_TIMEOUT_FLOOR_SEC = 120
_SINGLE_FILE_ASSUMED_MIN_SPEED_BYTES_PER_SEC = 1 * 1024 * 1024  # 1MB/s


def _single_file_timeout(size_bytes: int) -> float:
    return _SINGLE_FILE_TIMEOUT_FLOOR_SEC + max(size_bytes, 0) / _SINGLE_FILE_ASSUMED_MIN_SPEED_BYTES_PER_SEC


# Regex khớp dòng tổng kết số file rclone luôn in ở cuối (trừ khi chạy
# --quiet), dạng: "Transferred:            8 / 8, 100%". Có 2 dòng bắt đầu
# bằng "Transferred:" — 1 dòng theo BYTE (có dấu phẩy thập phân + đơn vị,
# vd "10.567 MiB / 10.567 MiB"), 1 dòng theo SỐ FILE (số nguyên thuần).
# Regex này chỉ khớp dòng SỐ FILE nhờ \d+ (không cho phép dấu chấm/chữ).
_RE_TRANSFERRED_COUNT = re.compile(r"Transferred:\s*(\d+)\s*/\s*(\d+)\s*,")
_RE_ERRORS_COUNT = re.compile(r"Errors:\s*(\d+)\b")


def _rclone_env(access_key: str, secret_key: str, endpoint_url: str,
                 region_name: str, addressing_style: str) -> dict:
    """Dựng biến môi trường cấu hình 1 remote rclone kiểu S3 hoàn toàn
    trong bộ nhớ tiến trình con — không ghi rclone.conf ra đĩa."""
    env = os.environ.copy()
    prefix = f"RCLONE_CONFIG_{_REMOTE_NAME.upper()}_"
    env[prefix + "TYPE"] = "s3"
    # "Other" = provider S3-compatible generic, dùng được cho Tigris,
    # Filebase, Cloudflare R2, MinIO... (không phải AWS thật, không cần
    # các quirk riêng của preset AWS trong rclone).
    env[prefix + "PROVIDER"] = "Other"
    env[prefix + "ACCESS_KEY_ID"] = access_key
    env[prefix + "SECRET_ACCESS_KEY"] = secret_key
    env[prefix + "ENDPOINT"] = endpoint_url
    env[prefix + "REGION"] = region_name
    # addressing_style="virtual" (Tigris) -> force_path_style=false.
    # addressing_style="path" (Filebase, MinIO...) -> force_path_style=true.
    env[prefix + "FORCE_PATH_STYLE"] = "false" if addressing_style == "virtual" else "true"
    # Không dùng bất kỳ file rclone.conf nào trên đĩa — mọi cấu hình đã đủ
    # qua biến môi trường ở trên.
    env["RCLONE_CONFIG"] = os.devnull
    return env


class CloudStorage:
    """Quản lý upload/download dữ liệu project lên/từ 1 dịch vụ lưu trữ
    tương thích S3 (mặc định: Tigris; cũng dùng được với Filebase, AWS S3,
    Cloudflare R2, hoặc bất kỳ provider S3-compatible nào khác) qua rclone.
    Logic bên trong hoàn toàn generic — chỉ cấu hình lại endpoint_url/
    region_name/addressing_style trong config.toml (mục [cloud]) là dùng
    được provider khác."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket_name: str = "ai-director-video",
        endpoint_url: str = "https://t3.storage.dev",
        region_name: str = "auto",
        addressing_style: str = "virtual",
        auto_create_bucket: bool = True,
        max_retries: int = 3,
    ):
        if not HAS_RCLONE:
            raise ImportError(
                "Cần cài rclone để dùng cloud storage (không phải gói pip). "
                "Cài đặt: curl https://rclone.org/install.sh | sudo bash "
                "(xem thêm https://rclone.org/downloads/)"
            )

        self.bucket_name = bucket_name
        # Số lần thử lại rclone tự làm cho MỖI file khi gặp lỗi mạng tạm
        # thời, đọc từ project.cloud_sync_retries trong config.toml (xem
        # get_cloud_storage_from_config) — cùng ý nghĩa như bản boto3 cũ,
        # nhưng giờ retry nằm hẳn trong rclone (--retries), không phải vòng
        # lặp Python tự viết.
        self.max_retries = max(0, max_retries)
        self._env = _rclone_env(access_key, secret_key, endpoint_url, region_name, addressing_style)
        self._remote = f"{_REMOTE_NAME}:{bucket_name}"

        # bucket_ready = bucket đã xác nhận tồn tại (hoặc vừa được tự tạo)
        # và dùng được. Mọi hàm upload/list/download bên dưới kiểm tra cờ
        # này trước, để báo lỗi 1 lần rõ ràng thay vì thất bại lặp lại
        # từng file.
        self.bucket_ready = False
        if auto_create_bucket:
            self.bucket_ready = self.ensure_bucket()

    # ------------------------------------------------------------------
    # Tiện ích gọi rclone
    # ------------------------------------------------------------------

    def _rclone_args(self, args: list[str]) -> list[str]:
        """Ghép flag retry/timeout mặc định (áp dụng cho MỌI lệnh) với các
        flag riêng của từng thao tác."""
        return [
            RCLONE_BIN,
            "--retries", str(self.max_retries + 1),
            "--low-level-retries", "10",
            "--contimeout", "30s",
            # Idle timeout: rclone tự huỷ 1 kết nối nếu KHÔNG NHẬN thêm
            # byte nào trong 300s — khác timeout "tổng thời gian" cứng
            # nhắc của bản cũ, nên file vài GB tải chậm nhưng vẫn đang
            # tiến triển sẽ KHÔNG bị giết oan.
            "--timeout", "300s",
            "--stats-log-level", "NOTICE",
            *args,
        ]

    def _rclone_capture(self, args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
        """Chạy 1 lệnh rclone ngắn (list/mkdir/size/xoá...), thu toàn bộ
        stdout/stderr để đọc kết quả."""
        return subprocess.run(
            self._rclone_args(args), env=self._env,
            capture_output=True, text=True, timeout=timeout,
        )

    # Nếu rclone không in thêm bất kỳ dòng log nào trong ngần này giây,
    # chủ động coi là "treo" và kill tiến trình — lưới an toàn Ở TẦNG
    # PYTHON, độc lập với --timeout (idle timeout) của rclone, phòng hờ
    # MỌI nguyên nhân treo khác chưa lường hết (không chỉ riêng bug
    # --progress đã sửa ở trên). An toàn tuyệt đối để làm điều này ở đây:
    # khác hẳn bug watchdog thời boto3 (phải "bỏ mặc" 1 thread Python kẹt
    # trong code C của s3transfer vì không kill được), ở đây ta chỉ kill
    # 1 TIẾN TRÌNH HỆ ĐIỀU HÀNH con do chính ta spawn ra — luôn kill sạch
    # được, không rò rỉ gì cả.
    _STREAM_STALL_TIMEOUT_SEC = 900  # 15 phút không có dòng log mới -> coi là treo

    def _rclone_stream(self, args: list[str]) -> tuple[int, list[str]]:
        """Chạy 1 lệnh rclone dài (copy project cả GB dữ liệu), IN TRỰC
        TIẾP log tiến độ của rclone ra console, đồng thời gom lại các dòng
        log để đọc số liệu tổng kết ở cuối (xem _RE_TRANSFERRED_COUNT).

        QUAN TRỌNG: KHÔNG dùng --progress ở đây. --progress vẽ 1 khối hiển
        thị bằng mã điều khiển terminal (\\r để đè dòng tại chỗ, không phải
        \\n xuống dòng thật) — hợp lý khi chạy trên terminal thật, nhưng
        khi stdout bị pipe (như subprocess.PIPE ở đây, hoặc khi chạy trên
        CI/GitHub Actions), rclone vẫn liên tục ghi \\r mà KHÔNG có \\n.
        for line in proc.stdout ở dưới đợi \\n để tách dòng -> không đọc
        được gì, trong khi buffer của pipe (thường 64KB) đầy dần rồi ghi
        bị chặn phía rclone -> CẢ HAI BÊN khoá chờ nhau vô thời hạn (nhìn
        như "treo", dù transfer thật có thể vẫn đang chạy tốt dưới nền).
        Đây CHÍNH LÀ nguyên nhân của lần treo 30+ phút đã gặp.
        Thay vào đó dùng "-v --stats-one-line": rclone in các dòng LOG
        thống kê định kỳ (mỗi --stats), MỖI dòng luôn kết thúc bằng \\n
        thật qua framework log thông thường -> an toàn tuyệt đối khi bị
        pipe/redirect, không có nguy cơ deadlock.

        Ngoài ra còn có lưới an toàn _STREAM_STALL_TIMEOUT_SEC (xem
        docstring của hằng số đó) để chủ động kill nếu vẫn treo vì lý do
        khác — đọc dòng qua 1 thread nền + queue để không tự chặn chính
        vòng lặp đọc bằng timeout được."""
        proc = subprocess.Popen(
            self._rclone_args(["-v", "--stats-one-line", *args]), env=self._env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None

        q: queue.Queue[str | None] = queue.Queue()

        def _reader():
            try:
                for raw_line in proc.stdout:
                    q.put(raw_line.rstrip("\n"))
            finally:
                q.put(None)  # đánh dấu stdout đã đóng (process đã thoát)

        threading.Thread(target=_reader, daemon=True).start()

        lines: list[str] = []
        stalled = False
        while True:
            try:
                item = q.get(timeout=self._STREAM_STALL_TIMEOUT_SEC)
            except queue.Empty:
                stalled = True
                print(f"[cloud] CẢNH BÁO: không có tiến triển nào trong "
                      f"{self._STREAM_STALL_TIMEOUT_SEC}s -> chủ động huỷ lệnh rclone "
                      f"(coi như thất bại, sẽ không chặn pipeline vô thời hạn).")
                proc.kill()
                break
            if item is None:
                break
            print(f"[cloud] {item}")
            lines.append(item)

        returncode = proc.wait() if not stalled else -9
        return returncode, lines

    # ------------------------------------------------------------------
    # Bucket
    # ------------------------------------------------------------------

    def ensure_bucket(self) -> bool:
        """Kiểm tra bucket đã tồn tại chưa; nếu chưa, tự tạo trên storage.

        Trả về True nếu bucket sẵn sàng dùng được (đã có sẵn hoặc vừa tạo
        thành công), False nếu không dùng được (vd tên bucket đã bị người
        khác dùng — tên bucket là DUY NHẤT TOÀN CỤC trên hầu hết dịch vụ
        tương thích S3, kể cả Filebase lẫn Tigris).
        """
        r = self._rclone_capture(["lsd", self._remote])
        if r.returncode == 0:
            return True

        r2 = self._rclone_capture(["mkdir", self._remote])
        if r2.returncode == 0:
            print(f"[cloud] Bucket '{self.bucket_name}' chưa có trên cloud — đã tự tạo mới.")
            return True

        err = (r2.stderr or r.stderr or "").strip()
        if "already" in err.lower() and ("own" in err.lower() or "exist" in err.lower()):
            # Đã tồn tại (của chính mình, hoặc lsd ở trên bị lỗi tạm thời)
            # -> coi như sẵn sàng, giống BucketAlreadyOwnedByYou của S3.
            return True
        if "bucketalreadyexists" in err.lower():
            print(
                f"[cloud] LỖI: Tên bucket '{self.bucket_name}' đã bị tài khoản "
                f"khác dùng (tên bucket trên storage tương thích S3 là duy nhất toàn cục, giống "
                f"AWS S3). Hãy đổi 'bucket_name' trong config.toml sang 1 tên khác, "
                f"riêng biệt cho bạn (vd 'ai-director-video-<ten-cua-ban>'), rồi chạy lại."
            )
        else:
            print(f"[cloud] Không tạo/kiểm tra được bucket '{self.bucket_name}': {err[-800:] or 'lỗi không rõ'}")
        return False

    # ------------------------------------------------------------------
    # File đơn lẻ (dùng bởi checkpoint.py — gọi rất thường xuyên,
    # có thể chạy song song hàng chục lần)
    # ------------------------------------------------------------------

    def _upload_file(self, local_path: Path, remote_key: str) -> bool:
        """Upload 1 file lên cloud. rclone tự retry nội bộ theo
        self.max_retries (xem _rclone_args) nên không cần vòng lặp
        backoff tự viết như bản cũ."""
        if not self.bucket_ready:
            return False
        try:
            size = local_path.stat().st_size
        except OSError:
            size = 0
        dest = f"{self._remote}/{remote_key}"
        try:
            r = self._rclone_capture(["copyto", str(local_path), dest],
                                      timeout=_single_file_timeout(size))
        except subprocess.TimeoutExpired:
            print(f"[cloud] Upload TIMEOUT cho '{remote_key}' (tiến trình rclone bị huỷ sạch, "
                  f"sẽ được retry ở checkpoint tiếp theo hoặc flush_pending_syncs()).")
            return False
        if r.returncode == 0:
            return True
        print(f"[cloud] Upload lỗi cho '{remote_key}': {(r.stderr or '').strip()[-500:]}")
        return False

    def _download_file(self, remote_key: str, local_path: Path) -> bool:
        """Tải 1 file từ cloud về."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        size = self._remote_size(remote_key) or 0
        src = f"{self._remote}/{remote_key}"
        try:
            r = self._rclone_capture(["copyto", src, str(local_path)],
                                      timeout=_single_file_timeout(size))
        except subprocess.TimeoutExpired:
            print(f"[cloud] Download TIMEOUT cho '{remote_key}'.")
            return False
        if r.returncode == 0:
            return True
        print(f"[cloud] Tải lỗi cho '{remote_key}': {(r.stderr or '').strip()[-500:]}")
        return False

    def _remote_size(self, remote_key: str) -> int | None:
        """Lấy kích thước (byte) của 1 object trên cloud, None nếu chưa tồn tại."""
        r = self._rclone_capture(["size", f"{self._remote}/{remote_key}", "--json"])
        if r.returncode != 0:
            return None
        try:
            data = json.loads(r.stdout)
            return int(data.get("bytes", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def sync_checkpoint(self, project_dir: Path, project_id: str, stage: str,
                         checkpoint_subdir: str = "checkpoints") -> bool:
        """Upload riêng 1 file checkpoint (dùng cho sync nhanh 1 stage).

        LƯU Ý: hiện KHÔNG có nơi nào trong codebase gọi hàm này (checkpoint
        sync thật đang đi qua CheckpointManager._sync_to_cloud() trong
        checkpoint.py). `checkpoint_subdir` phải khớp với tên thư mục
        checkpoint thật (xem paths.checkpoint_dir trong config.toml) nếu
        được dùng trong tương lai."""
        if not self.bucket_ready:
            return False
        ckpt_file = project_dir / checkpoint_subdir / f"{stage}.json"
        if ckpt_file.exists():
            remote_key = f"projects/{project_id}/checkpoints/{stage}.json"
            return self._upload_file(ckpt_file, remote_key)
        return False

    # ------------------------------------------------------------------
    # Cả project (upload/download hàng loạt)
    # ------------------------------------------------------------------

    def upload_project(self, project_dir: Path, project_id: str, max_workers: int = 16) -> dict[str, Any]:
        """Upload toàn bộ thư mục project (bao gồm input/ chứa video gốc,
        checkpoints/, output/) lên cloud bằng 1 lệnh `rclone copy` duy
        nhất.

        rclone TỰ so sánh kích thước + thời gian sửa đổi (và có thể bật
        checksum) để bỏ qua file đã có sẵn giống hệt trên cloud — thay cho
        toàn bộ logic so ETag/MD5 thủ công ~80 dòng của bản boto3 cũ.
        rclone cũng tự chạy song song nhiều file cùng lúc qua
        --transfers/--checkers, và tự động cân bằng hợp lý giữa file nhỏ
        (nhiều, cần độ trễ thấp) và file nặng (ít, cần băng thông) mà
        không cần chia 2 nhóm thủ công như trước."""
        if not self.bucket_ready:
            print(f"[cloud] Bỏ qua đồng bộ: bucket '{self.bucket_name}' chưa sẵn sàng "
                  f"(xem lỗi ensure_bucket ở trên).")
            return {"uploaded": 0, "skipped": 0, "errors": 0, "error_files": [],
                    "aborted": True}

        dest = f"{self._remote}/projects/{project_id}"
        print(f"[cloud] Bắt đầu đồng bộ project '{project_id}' lên cloud...")
        start = time.time()
        returncode, lines = self._rclone_stream([
            "copy", str(project_dir), dest,
            # Bỏ qua file metadata nội bộ tạm (vd _upload_meta.json cũ),
            # giống hệt điều kiện "startswith('_') and endswith('.json')"
            # của bản cũ. Pattern không có "/" nên rclone tự khớp ở MỌI
            # cấp thư mục, không chỉ gốc.
            "--exclude", "_*.json",
            "--transfers", str(max_workers),
            "--checkers", str(max_workers),
            "--stats", "5s",
        ])
        elapsed = time.time() - start

        transferred, total = self._parse_transferred_count(lines)
        errors = self._parse_errors_count(lines)
        ok = returncode == 0

        # Upload metadata của lần sync này (file riêng, KHÔNG bị lệnh copy
        # ở trên tự loại vì --exclude chỉ áp cho chính lệnh copy đó).
        meta = {
            "project_id": project_id,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_count": total,
            "error_count": errors,
        }
        meta_file = project_dir / "_upload_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        self._upload_file(meta_file, f"projects/{project_id}/_upload_meta.json")
        meta_file.unlink(missing_ok=True)

        print(f"[cloud] Đồng bộ xong trong {elapsed:.1f}s: {transferred}/{total} file cần cập nhật "
              f"đã lên cloud, {errors} lỗi" + ("." if ok else " (rclone thoát với lỗi)."))

        return {
            # "uploaded": số file rclone THẬT SỰ đã transfer (đã trừ file
            # giống hệt bản cloud, do rclone tự bỏ qua). Không còn tách
            # riêng "skipped" như bản cũ vì rclone không tiện lộ ra số này
            # qua output tổng kết đơn giản — giữ field "skipped" = 0 để
            # tương thích ngược với code gọi result['skipped'].
            "uploaded": transferred,
            "skipped": 0,
            "errors": errors if errors else (0 if ok else 1),
            "error_files": [] if ok else [f"rclone thoát với mã lỗi {returncode} — xem log ở trên"],
        }

    def download_project(self, project_id: str, local_dir: Path, max_workers: int = 16) -> bool:
        """Tải 1 project từ cloud về thư mục cục bộ (bao gồm input/video)
        bằng 1 lệnh `rclone copy` duy nhất."""
        if not self.bucket_ready:
            print(f"[cloud] Không tải được: bucket '{self.bucket_name}' chưa sẵn sàng.")
            return False

        remote_prefix = f"projects/{project_id}"
        check = self._rclone_capture(["lsjson", "--recursive", f"{self._remote}/{remote_prefix}"])
        if check.returncode != 0 or not (check.stdout or "").strip() or check.stdout.strip() == "[]":
            print(f"[cloud] Không tìm thấy project '{project_id}'.")
            return False

        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cloud] Bắt đầu tải project '{project_id}' về {local_dir}...")
        start = time.time()
        returncode, lines = self._rclone_stream([
            "copy", f"{self._remote}/{remote_prefix}", str(local_dir),
            "--transfers", str(max_workers),
            "--checkers", str(max_workers),
            "--stats", "5s",
        ])
        elapsed = time.time() - start

        transferred, total = self._parse_transferred_count(lines)
        ok = returncode == 0
        print(f"[cloud] Đã tải {transferred}/{total} file về {local_dir} "
              f"trong {elapsed:.1f}s" + ("." if ok else " (rclone thoát với lỗi)."))
        return ok

    @staticmethod
    def _parse_transferred_count(lines: list[str]) -> tuple[int, int]:
        """Đọc số file (đã transfer / tổng số cần xử lý) từ dòng tổng kết
        cuối cùng rclone in ra ("Transferred: X / Y, 100%"). Có thể có
        nhiều dòng dạng này (in định kỳ mỗi --stats) — lấy dòng CUỐI CÙNG
        vì đó là số liệu sau khi hoàn tất."""
        transferred, total = 0, 0
        for line in lines:
            m = _RE_TRANSFERRED_COUNT.search(line)
            if m:
                transferred, total = int(m.group(1)), int(m.group(2))
        return transferred, total

    @staticmethod
    def _parse_errors_count(lines: list[str]) -> int:
        errors = 0
        for line in lines:
            m = _RE_ERRORS_COUNT.search(line)
            if m:
                errors = int(m.group(1))
        return errors

    # ------------------------------------------------------------------
    # Liệt kê / xoá project trên cloud
    # ------------------------------------------------------------------

    def list_remote_projects(self) -> list[dict[str, Any]]:
        """Liệt kê tất cả project đang lưu trên cloud.

        Kèm theo "last_modified" (thời điểm object mới nhất được ghi) và
        "status" suy luận (completed/in_progress/new) dựa trên các key
        thấy được — CẦN THIẾT cho môi trường chạy không có ổ đĩa cũ (vd
        GitHub Actions), nơi không thể dựa vào mtime file cục bộ như
        ProjectManager._refresh_project_meta vẫn làm cho project chạy trên
        máy có ổ đĩa bền vững.
        """
        if not self.bucket_ready:
            return []
        r = self._rclone_capture(["lsjson", "--recursive", f"{self._remote}/projects"])
        if r.returncode != 0:
            # Prefix "projects/" chưa từng có object nào -> không phải lỗi
            # thật, chỉ là chưa có project nào trên cloud.
            if r.stdout and r.stdout.strip() not in ("", "[]", "null"):
                print(f"[cloud] Lỗi khi liệt kê project: {(r.stderr or '').strip()[-500:]}")
            return []
        try:
            entries = json.loads(r.stdout or "[]")
        except json.JSONDecodeError as e:
            print(f"[cloud] Lỗi khi liệt kê project: {e}")
            return []

        projects: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if entry.get("IsDir"):
                continue
            path = entry.get("Path", "")
            parts = path.split("/")
            if len(parts) < 2:
                continue
            project_id = parts[0]
            p = projects.setdefault(project_id, {
                "project_id": project_id,
                "file_count": 0,
                "status": "new",
                "_last_modified": "",
            })
            p["file_count"] += 1
            mod_time = str(entry.get("ModTime", ""))
            if mod_time > p["_last_modified"]:
                p["_last_modified"] = mod_time
            if path.endswith("output/deliverables/final_preview.mp4"):
                p["status"] = "completed"
            elif "/checkpoints/" in path and p["status"] != "completed":
                p["status"] = "in_progress"

        result = []
        for p in projects.values():
            raw = p.pop("_last_modified")
            # ModTime của rclone lsjson theo chuẩn RFC3339 (vd
            # "2026-07-19T10:30:00.000000000+07:00") -> cắt về dạng ngắn
            # gọn giống bản cũ để hiển thị trong menu.
            p["last_modified"] = raw[:19].replace("T", " ") if raw else ""
            result.append(p)
        return result

    def delete_project(self, project_id: str) -> bool:
        """Xoá toàn bộ 1 project trên cloud."""
        if not self.bucket_ready:
            print(f"[cloud] Không xoá được: bucket '{self.bucket_name}' chưa sẵn sàng.")
            return False
        r = self._rclone_capture(["purge", f"{self._remote}/projects/{project_id}"])
        if r.returncode == 0:
            print(f"[cloud] Đã xoá project '{project_id}' trên cloud.")
            return True
        print(f"[cloud] Xoá project '{project_id}' thất bại: {(r.stderr or '').strip()[-500:]}")
        return False


def get_cloud_storage_from_config(cfg) -> CloudStorage | None:
    """Tạo instance CloudStorage từ section [cloud] trong config.toml.

    Đọc thêm region_name/addressing_style (mặc định là cấu hình của Tigris
    nếu không có trong config) — cho phép dùng bất kỳ dịch vụ tương thích S3
    nào (Tigris, Filebase, AWS S3, R2, ...) chỉ bằng cách đổi giá trị trong
    config.toml, không cần sửa code."""
    if not cfg.get("cloud.enabled", True):
        print("[cloud] cloud.enabled = false trong config.toml — bỏ qua cloud sync.")
        return None

    access_key = cfg.get("cloud.access_key", "")
    secret_key = cfg.get("cloud.secret_key", "")
    bucket = cfg.get("cloud.bucket_name", "ai-director-video")
    endpoint = cfg.get("cloud.endpoint_url", "https://t3.storage.dev")
    region_name = cfg.get("cloud.region_name", "auto")
    addressing_style = cfg.get("cloud.addressing_style", "virtual")
    max_retries = cfg.get("project.cloud_sync_retries", 3)

    if not access_key or access_key.startswith("PASTE_"):
        print("[cloud] Chưa cấu hình access_key — tắt tính năng sync cloud.")
        return None

    try:
        storage = CloudStorage(
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket,
            endpoint_url=endpoint,
            region_name=region_name,
            addressing_style=addressing_style,
            max_retries=max_retries,
        )
        if storage.bucket_ready:
            print(f"[cloud] Bucket '{bucket}' sẵn sàng ({endpoint}).")
        else:
            print(f"[cloud] CẢNH BÁO: bucket '{bucket}' KHÔNG dùng được — "
                  f"tính năng cloud sync sẽ bị bỏ qua cho tới khi bucket được sửa "
                  f"(xem lỗi ensure_bucket ở trên).")
        return storage
    except ImportError as e:
        print(f"[cloud] {e}")
        return None
    except Exception as e:
        print(f"[cloud] Khởi tạo thất bại: {e}")
        return None
