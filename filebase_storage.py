"""
filebase_storage.py — Tích hợp lưu trữ Filebase (tương thích S3).

Dùng boto3 để giao tiếp với Filebase (object storage tương thích S3) nhằm
sao lưu/khôi phục dữ liệu project lên/từ cloud.

Filebase endpoint: https://s3.filebase.com
Giao thức: tương thích S3 API
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from progress_utils import print_progress_bar
except ImportError:
    def print_progress_bar(current, total, prefix="", suffix="", bar_len=30):
        pass  # fallback im lặng nếu progress_utils không có sẵn (không nên xảy ra trong project này)


# File có kích thước >= ngưỡng này (video gốc, checkpoint model...) được coi
# là "nặng": tải RIÊNG (không xen vào ThreadPoolExecutor của các file nhỏ)
# và có dòng tiến độ RIÊNG tính theo byte thật, thay vì lẫn vào thanh đếm
# số file — trước đây 1 file 1.3GB cũng chỉ tính là "+1" như 1 file keyframe
# vài chục KB, nên thanh đếm file gần như đứng im ở vòng lặp cuối dù vẫn
# đang tải, nhìn như bị treo.
HEAVY_FILE_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20MB


def _format_size(n: float) -> str:
    """Định dạng số byte thành chuỗi dễ đọc (KB/MB/GB)."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


class _ByteProgress:
    """Callback cho boto3 download_file/upload_file dùng cho 1 file nặng:
    in tiến độ theo byte thật (MB đã tải/tổng MB) trên dòng log riêng, có
    tên file để phân biệt với thanh đếm file của các file nhỏ chạy song song."""

    def __init__(self, label: str, total_bytes: int):
        self.label = label
        self.total_bytes = max(total_bytes, 1)  # tránh chia cho 0 nếu size=0
        self.seen = 0
        self.start = time.time()

    def __call__(self, bytes_amount: int) -> None:
        # boto3 gọi lại nhiều lần với số byte MỚI nhận mỗi lần (không phải
        # tổng luỹ kế), nên phải cộng dồn ở đây.
        self.seen += bytes_amount
        elapsed = time.time() - self.start
        speed = self.seen / elapsed if elapsed > 0 else 0
        print_progress_bar(
            self.seen, self.total_bytes,
            prefix=f"[filebase] {self.label}",
            suffix=f"{_format_size(self.seen)}/{_format_size(self.total_bytes)}, {_format_size(speed)}/s",
        )


class FilebaseStorage:
    """Quản lý upload/download dữ liệu project lên/từ 1 dịch vụ lưu trữ
    tương thích S3 (Filebase, Tigris, hoặc bất kỳ provider S3-compatible nào)
    qua S3 API. Tên class/module giữ nguyên là "Filebase" vì lý do lịch sử,
    nhưng logic bên trong hoàn toàn generic — chỉ cấu hình lại endpoint_url/
    region_name/addressing_style trong config.toml là dùng được provider khác."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket_name: str = "ai-director-video",
        endpoint_url: str = "https://s3.filebase.com",
        region_name: str = "us-east-1",
        addressing_style: str = "path",
        auto_create_bucket: bool = True,
    ):
        if not HAS_BOTO3:
            raise ImportError("Cần cài boto3 để dùng cloud storage. Chạy: pip install boto3")

        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url

        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            # region_name/addressing_style giờ đọc từ config.toml (mục
            # [filebase]) thay vì hard-code riêng cho Filebase, vì mỗi
            # provider S3-compatible yêu cầu khác nhau để tính chữ ký request
            # (SigV4) và dựng URL đúng cách:
            #   - Filebase: region_name="us-east-1", addressing_style="path"
            #   - Tigris:   region_name="auto",      addressing_style="virtual"
            # Đặt sai sẽ khiến request bị ký sai hoặc gọi nhầm URL một cách
            # âm thầm (lỗi 403/404 khó hiểu) thay vì lỗi rõ ràng.
            region_name=region_name,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": addressing_style},
            ),
        )

        # bucket_ready = bucket đã xác nhận tồn tại (hoặc vừa được tự tạo) và
        # dùng được. Mọi hàm upload/list/download bên dưới sẽ kiểm tra cờ này
        # trước, để báo lỗi 1 lần rõ ràng thay vì thất bại lặp lại từng file.
        self.bucket_ready = False
        if auto_create_bucket:
            self.bucket_ready = self.ensure_bucket()
            if self.bucket_ready:
                self.ensure_cors()

    def ensure_bucket(self) -> bool:
        """Kiểm tra bucket đã tồn tại chưa; nếu chưa, tự tạo trên storage.

        Trả về True nếu bucket sẵn sàng dùng được (đã có sẵn hoặc vừa tạo
        thành công), False nếu không dùng được (vd tên bucket đã bị người
        khác dùng — tên bucket là DUY NHẤT TOÀN CỤC trên hầu hết dịch vụ
        tương thích S3, kể cả Filebase lẫn Tigris).
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            error_code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
            if error_code not in ("404", "NoSuchBucket", "NotFound", ""):
                print(f"[filebase] Không kiểm tra được bucket '{self.bucket_name}': {e}")
                return False

        # Bucket chưa tồn tại -> thử tự tạo
        try:
            self.client.create_bucket(Bucket=self.bucket_name)
            print(f"[filebase] Bucket '{self.bucket_name}' chưa có trên Filebase — đã tự tạo mới.")
            return True
        except Exception as e:
            error_code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
            if error_code == "BucketAlreadyOwnedByYou":
                return True
            if error_code == "BucketAlreadyExists":
                print(
                    f"[filebase] LỖI: Tên bucket '{self.bucket_name}' đã bị tài khoản "
                    f"khác dùng (tên bucket trên Filebase là duy nhất toàn cục, giống "
                    f"AWS S3). Hãy đổi 'bucket_name' trong config.toml sang 1 tên khác, "
                    f"riêng biệt cho bạn (vd 'ai-director-video-<ten-cua-ban>'), rồi chạy lại."
                )
            else:
                print(f"[filebase] Không tự tạo được bucket '{self.bucket_name}': {e}")
            return False

    def _upload_file(self, local_path: Path, remote_key: str) -> bool:
        """Upload 1 file lên Filebase."""
        try:
            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                remote_key,
            )
            return True
        except Exception as e:
            print(f"[filebase] Upload thất bại cho {remote_key}: {e}")
            return False

    def upload_public(self, local_path: Path, remote_key: str) -> bool:
        """Upload 1 file với ACL public-read — CHỈ dùng cho status.json (để
        dashboard web/mobile đọc trực tiếp qua URL công khai, không cần
        access_key/secret_key). Mọi file khác (video, checkpoint, output)
        vẫn phải dùng _upload_file() (private) như cũ."""
        try:
            self.client.upload_file(
                str(local_path), self.bucket_name, remote_key,
                ExtraArgs={"ACL": "public-read"},
            )
            return True
        except Exception as e:
            print(f"[filebase] Upload public thất bại cho {remote_key}: {e}")
            return False

    def ensure_cors(self) -> bool:
        """Bật CORS (allow GET từ mọi origin) cho bucket, để trình duyệt
        (dashboard web chạy trên điện thoại) fetch được status.json công khai
        trực tiếp từ Filebase. Không ảnh hưởng quyền riêng tư của các file
        khác — CORS chỉ quyết định trình duyệt nào được ĐỌC file ĐÃ public,
        không tự động public hoá file private."""
        try:
            self.client.put_bucket_cors(
                Bucket=self.bucket_name,
                CORSConfiguration={
                    "CORSRules": [{
                        "AllowedOrigins": ["*"],
                        "AllowedMethods": ["GET"],
                        "AllowedHeaders": ["*"],
                        "MaxAgeSeconds": 3600,
                    }]
                },
            )
            return True
        except Exception as e:
            print(f"[filebase] Không bật được CORS (dashboard web có thể không đọc được status.json): {e}")
            return False

    def _download_file(self, remote_key: str, local_path: Path, callback=None) -> bool:
        """Tải 1 file từ Filebase về. `callback`, nếu có, được boto3 gọi lại
        nhiều lần trong lúc tải với số byte MỚI nhận (không phải luỹ kế) —
        dùng cho file nặng để in tiến độ theo byte thật (xem download_project)."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                self.bucket_name,
                remote_key,
                str(local_path),
                Callback=callback,
            )
            return True
        except Exception as e:
            print(f"[filebase] Tải thất bại cho {remote_key}: {e}")
            return False

    def _list_files(self, prefix: str) -> list[str]:
        """Liệt kê tất cả file dưới 1 prefix (thư mục) trên Filebase."""
        return [key for key, _size in self._list_files_with_size(prefix)]

    def _list_files_with_size(self, prefix: str) -> list[tuple[str, int]]:
        """Liệt kê tất cả file dưới 1 prefix, kèm kích thước (byte).

        list_objects_v2 đã trả sẵn "Size" cho mỗi object trong response nên
        không cần gọi head_object riêng cho từng file (đỡ tốn round-trip) —
        dùng để phân loại file nhỏ/nặng trước khi tải ở download_project()."""
        try:
            files = []
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    files.append((obj["Key"], obj.get("Size", 0)))
            return files
        except Exception as e:
            print(f"[filebase] Lỗi liệt kê {prefix}: {e}")
            return []

    def _delete_file(self, remote_key: str) -> bool:
        """Xoá 1 file trên Filebase."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except Exception:
            return False

    def _remote_size_and_etag(self, remote_key: str) -> tuple[int, str] | None:
        """Lấy (kích thước, ETag) của 1 object trên Filebase, None nếu chưa tồn tại."""
        try:
            resp = self.client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return resp.get("ContentLength"), str(resp.get("ETag", "")).strip('"')
        except Exception:
            return None

    def _remote_size(self, remote_key: str) -> int | None:
        """Lấy kích thước (byte) của 1 object trên Filebase, None nếu chưa tồn tại."""
        try:
            resp = self.client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return resp.get("ContentLength")
        except Exception:
            return None

    @staticmethod
    def _local_md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def upload_project(self, project_dir: Path, project_id: str, max_workers: int = 16) -> dict[str, Any]:
        """Upload toàn bộ thư mục project (bao gồm input/ chứa video gốc,
        checkpoints/, output/) lên Filebase.

        Video gốc giờ nằm trong project_dir/input/ (xem project_manager.py:
        create_project()) nên rglob("*") ở đây tự động cuốn nó theo — không
        cần logic upload riêng cho video. Vì video có thể rất lớn và hiếm khi
        đổi giữa các lần sync, ta so kích thước với bản đã có trên cloud
        (head_object) và bỏ qua nếu trùng, để tránh upload lại hàng GB không
        cần thiết mỗi lần đồng bộ.

        BUGFIX: chỉ so KÍCH THƯỚC là không đủ — 2 file khác nội dung nhưng
        tình cờ trùng byte-size (khá dễ xảy ra với các file JSON/keyframe nhỏ,
        vd asr_timeline.json trước và sau khi sửa 1 câu thoại có thể trùng
        size) sẽ bị coi là "đã có sẵn" và KHÔNG được upload bản mới, khiến
        cloud giữ mãi bản cũ mà không ai biết. Với file NHỎ (dưới ngưỡng
        HEAVY_FILE_THRESHOLD_BYTES), khi size trùng ta so thêm MD5 cục bộ với
        ETag trên Filebase trước khi quyết định bỏ qua. Với file NẶNG (video
        gốc, hiếm đổi, hash cả GB mỗi lần sync sẽ rất chậm), vẫn chỉ so size
        như cũ — đây là đánh đổi có chủ đích, không phải bug.

        Upload SONG SONG (giống download_project()) vì project thường có
        hàng ngàn file keyframe nhỏ — tuần tự từng file sẽ bị chặn bởi độ
        trễ mạng (latency) chứ không phải băng thông, dù mỗi file tự nó
        upload nhanh."""
        if not self.bucket_ready:
            print(f"[filebase] Bỏ qua đồng bộ: bucket '{self.bucket_name}' chưa sẵn sàng "
                  f"(xem lỗi ensure_bucket ở trên).")
            return {"uploaded": 0, "skipped": 0, "errors": 0, "error_files": [],
                    "aborted": True}

        remote_prefix = f"projects/{project_id}/"

        candidates = []
        for local_file in project_dir.rglob("*"):
            if local_file.is_dir():
                continue
            if local_file.name.startswith("_") and local_file.name.endswith(".json"):
                continue
            rel = local_file.relative_to(project_dir)
            remote_key = f"{remote_prefix}{rel.as_posix()}"
            candidates.append((local_file, remote_key))

        total = len(candidates)
        uploaded: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        lock = threading.Lock()
        start = time.time()

        print_progress_bar(0, total, prefix="[filebase] upload", suffix=f"0/{total} file")

        def _job(local_file: Path, remote_key: str) -> tuple[str, str]:
            """Trả về (remote_key, 'uploaded'|'skipped'|'error')."""
            local_size = local_file.stat().st_size
            remote = self._remote_size_and_etag(remote_key)
            if remote is not None:
                remote_size, remote_etag = remote
                if remote_size == local_size:
                    if local_size >= HEAVY_FILE_THRESHOLD_BYTES:
                        # File nặng: chấp nhận đánh đổi, chỉ so size (xem docstring).
                        return remote_key, "skipped"
                    # File nhỏ/vừa: multipart ETag (chứa '-') không phải MD5
                    # thuần -> không so được, đành upload lại cho an toàn.
                    if "-" not in remote_etag:
                        try:
                            if self._local_md5(local_file) == remote_etag:
                                return remote_key, "skipped"
                        except OSError:
                            pass
            ok = self._upload_file(local_file, remote_key)
            return remote_key, ("uploaded" if ok else "error")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_job, lf, rk) for lf, rk in candidates]
            done = 0
            for future in as_completed(futures):
                remote_key, status = future.result()
                with lock:
                    done += 1
                    if status == "uploaded":
                        uploaded.append(remote_key)
                    elif status == "skipped":
                        skipped.append(remote_key)
                    else:
                        errors.append(remote_key)
                    elapsed = time.time() - start
                    speed = done / elapsed if elapsed > 0 else 0
                    print_progress_bar(
                        done, total, prefix="[filebase] upload",
                        suffix=f"{done}/{total} file, {speed:.1f} file/s",
                    )

        # Upload metadata của lần sync này
        meta = {
            "project_id": project_id,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_count": len(uploaded),
            "skipped_count": len(skipped),
            "error_count": len(errors),
        }
        meta_file = project_dir / "_upload_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        self._upload_file(meta_file, f"{remote_prefix}_upload_meta.json")
        meta_file.unlink(missing_ok=True)

        elapsed = time.time() - start
        print(f"[filebase] Upload xong trong {elapsed:.1f}s: {len(uploaded)} file tải lên, "
              f"{len(skipped)} bỏ qua (đã có sẵn), {len(errors)} lỗi.")

        return {
            "uploaded": len(uploaded),
            "skipped": len(skipped),
            "errors": len(errors),
            "error_files": errors,
        }

    def download_project(self, project_id: str, local_dir: Path, max_workers: int = 16) -> bool:
        """Tải 1 project từ Filebase về thư mục cục bộ (bao gồm input/video).

        Tách làm 2 nhóm:
          - File NHỎ (< HEAVY_FILE_THRESHOLD_BYTES, VD: keyframe .jpg): tải
            SONG SONG bằng ThreadPoolExecutor như cũ, thanh tiến độ đếm theo
            SỐ FILE — hợp lý vì có hàng ngàn file, mỗi file tải rất nhanh,
            nút thắt là ĐỘ TRỄ mạng chứ không phải băng thông.
          - File NẶNG (video gốc, checkpoint model...): tải TUẦN TỰ, SAU khi
            nhóm file nhỏ xong, mỗi file có dòng tiến độ RIÊNG tính theo BYTE
            THẬT (MB đã tải/tổng MB). Trước đây file nặng bị trộn chung vào
            thanh đếm file: 1 file 1.3GB cũng chỉ tính "+1" như 1 keyframe vài
            chục KB, nên khi tới file cuối, thanh gần như đứng im ở mốc
            (total-1)/total trong lúc file đó vẫn đang tải thật — nhìn như bị
            treo. Tải tuần tự (không xen với pool file nhỏ) để tránh 2 dòng
            tiến độ \\r ghi đè lẫn nhau trên cùng 1 dòng terminal."""
        if not self.bucket_ready:
            print(f"[filebase] Không tải được: bucket '{self.bucket_name}' chưa sẵn sàng.")
            return False
        remote_prefix = f"projects/{project_id}/"
        files = self._list_files_with_size(remote_prefix)
        if not files:
            print(f"[filebase] Không tìm thấy project '{project_id}'.")
            return False

        local_dir.mkdir(parents=True, exist_ok=True)
        small_files: list[tuple[str, Path]] = []
        heavy_files: list[tuple[str, Path, int]] = []
        for remote_key, size in files:
            if remote_key.endswith("/"):
                continue
            rel = remote_key[len(remote_prefix):]
            if not rel or rel.startswith("_"):
                continue
            local_file = local_dir / rel
            if size >= HEAVY_FILE_THRESHOLD_BYTES:
                heavy_files.append((remote_key, local_file, size))
            else:
                small_files.append((remote_key, local_file))

        total = len(small_files) + len(heavy_files)
        downloaded = 0
        failed: list[str] = []
        lock = threading.Lock()
        start = time.time()

        # --- Nhóm 1: file nhỏ, tải song song, tiến độ theo số file ---
        if small_files:
            print_progress_bar(0, len(small_files), prefix="[filebase] download",
                                suffix=f"0/{len(small_files)} file nhỏ")

            def _job(remote_key: str, local_file: Path) -> tuple[str, bool]:
                ok = self._download_file(remote_key, local_file)
                return remote_key, ok

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_job, rk, lf) for rk, lf in small_files]
                done_small = 0
                for future in as_completed(futures):
                    remote_key, ok = future.result()
                    with lock:
                        done_small += 1
                        downloaded += 1
                        if not ok:
                            failed.append(remote_key)
                        elapsed = time.time() - start
                        speed = done_small / elapsed if elapsed > 0 else 0
                        print_progress_bar(
                            done_small, len(small_files), prefix="[filebase] download",
                            suffix=f"{done_small}/{len(small_files)} file nhỏ, {speed:.1f} file/s",
                        )

        # --- Nhóm 2: file nặng, tải tuần tự, tiến độ theo byte thật ---
        for i, (remote_key, local_file, size) in enumerate(heavy_files, start=1):
            label = f"download nặng ({i}/{len(heavy_files)}) {local_file.name}"
            print(f"[filebase] Bắt đầu tải file nặng: {local_file.name} ({_format_size(size)})")
            progress_cb = _ByteProgress(label, size)
            ok = self._download_file(remote_key, local_file, callback=progress_cb)
            with lock:
                downloaded += 1
                if not ok:
                    failed.append(remote_key)

        elapsed = time.time() - start
        print(f"[filebase] Đã tải {downloaded - len(failed)}/{total} file về {local_dir} "
              f"trong {elapsed:.1f}s ({len(failed)} lỗi)" + (f": {failed[:5]}..." if failed else "."))
        return len(failed) == 0 or (downloaded - len(failed)) > 0

    def list_remote_projects(self) -> list[dict[str, Any]]:
        """Liệt kê tất cả project đang lưu trên Filebase."""
        if not self.bucket_ready:
            return []
        projects = {}
        try:
            files = self._list_files("projects/")
            for key in files:
                parts = key.split("/")
                if len(parts) >= 2:
                    project_id = parts[1]
                    if project_id not in projects:
                        projects[project_id] = {
                            "project_id": project_id,
                            "file_count": 0,
                        }
                    projects[project_id]["file_count"] += 1
        except Exception as e:
            print(f"[filebase] Lỗi khi liệt kê project: {e}")
        return list(projects.values())

    def delete_project(self, project_id: str) -> bool:
        """Xoá toàn bộ 1 project trên Filebase."""
        if not self.bucket_ready:
            print(f"[filebase] Không xoá được: bucket '{self.bucket_name}' chưa sẵn sàng.")
            return False
        remote_prefix = f"projects/{project_id}/"
        files = self._list_files(remote_prefix)
        deleted = 0
        for key in files:
            if self._delete_file(key):
                deleted += 1
        print(f"[filebase] Đã xoá {deleted} file của project '{project_id}'")
        return deleted > 0

    def sync_checkpoint(self, project_dir: Path, project_id: str, stage: str) -> bool:
        """Upload riêng 1 file checkpoint (dùng cho sync nhanh 1 stage)."""
        if not self.bucket_ready:
            return False
        ckpt_file = project_dir / "checkpoints" / f"{stage}.json"
        if ckpt_file.exists():
            remote_key = f"projects/{project_id}/checkpoints/{stage}.json"
            return self._upload_file(ckpt_file, remote_key)
        return False


def get_filebase_storage_from_config(cfg) -> FilebaseStorage | None:
    """Tạo instance FilebaseStorage từ section [filebase] trong config.toml.

    Đọc thêm region_name/addressing_style (mặc định giữ hành vi cũ của
    Filebase nếu không có trong config, để không phá cấu hình cũ) — cho phép
    dùng bất kỳ dịch vụ tương thích S3 nào (Tigris, Filebase, ...) chỉ bằng
    cách đổi giá trị trong config.toml, không cần sửa code."""
    access_key = cfg.get("filebase.access_key", "")
    secret_key = cfg.get("filebase.secret_key", "")
    bucket = cfg.get("filebase.bucket_name", "ai-director-video")
    endpoint = cfg.get("filebase.endpoint_url", "https://s3.filebase.com")
    region_name = cfg.get("filebase.region_name", "us-east-1")
    addressing_style = cfg.get("filebase.addressing_style", "path")

    if not access_key or access_key.startswith("PASTE_"):
        print("[filebase] Chưa cấu hình access_key — tắt tính năng sync cloud.")
        return None

    try:
        storage = FilebaseStorage(
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket,
            endpoint_url=endpoint,
            region_name=region_name,
            addressing_style=addressing_style,
        )
        if storage.bucket_ready:
            print(f"[filebase] Bucket '{bucket}' sẵn sàng ({endpoint}).")
        else:
            print(f"[filebase] CẢNH BÁO: bucket '{bucket}' KHÔNG dùng được — "
                  f"tính năng cloud sync sẽ bị bỏ qua cho tới khi bucket được sửa "
                  f"(xem lỗi ensure_bucket ở trên).")
        return storage
    except ImportError:
        print("[filebase] Chưa cài boto3. Chạy: pip install boto3")
        return None
    except Exception as e:
        print(f"[filebase] Khởi tạo thất bại: {e}")
        return None
