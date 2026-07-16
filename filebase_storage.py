"""
filebase_storage.py — Tích hợp lưu trữ Filebase (tương thích S3).

Dùng boto3 để giao tiếp với Filebase (object storage tương thích S3) nhằm
sao lưu/khôi phục dữ liệu project lên/từ cloud.

Filebase endpoint: https://s3.filebase.com
Giao thức: tương thích S3 API
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class FilebaseStorage:
    """Quản lý upload/download dữ liệu project lên/từ Filebase qua S3 API."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket_name: str = "ai-director-video",
        endpoint_url: str = "https://s3.filebase.com",
        auto_create_bucket: bool = True,
    ):
        if not HAS_BOTO3:
            raise ImportError("Cần cài boto3 để dùng Filebase storage. Chạy: pip install boto3")

        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url

        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            # Filebase yêu cầu region_name="us-east-1" bắt buộc để tính chữ ký
            # request (SigV4) đúng cách — không phụ thuộc region mặc định của
            # máy/AWS profile người dùng (nếu có), tránh lỗi ký sai âm thầm.
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

        # bucket_ready = bucket đã xác nhận tồn tại (hoặc vừa được tự tạo) và
        # dùng được. Mọi hàm upload/list/download bên dưới sẽ kiểm tra cờ này
        # trước, để báo lỗi 1 lần rõ ràng thay vì thất bại lặp lại từng file.
        self.bucket_ready = False
        if auto_create_bucket:
            self.bucket_ready = self.ensure_bucket()

    def ensure_bucket(self) -> bool:
        """Kiểm tra bucket đã tồn tại chưa; nếu chưa, tự tạo trên Filebase.

        Trả về True nếu bucket sẵn sàng dùng được (đã có sẵn hoặc vừa tạo
        thành công), False nếu không dùng được (vd tên bucket đã bị người
        khác dùng — tên bucket trên Filebase là DUY NHẤT TOÀN CỤC, giống S3).
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

    def _download_file(self, remote_key: str, local_path: Path) -> bool:
        """Tải 1 file từ Filebase về."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                self.bucket_name,
                remote_key,
                str(local_path),
            )
            return True
        except Exception as e:
            print(f"[filebase] Tải thất bại cho {remote_key}: {e}")
            return False

    def _list_files(self, prefix: str) -> list[str]:
        """Liệt kê tất cả file dưới 1 prefix (thư mục) trên Filebase."""
        try:
            files = []
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    files.append(obj["Key"])
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

    def _remote_size(self, remote_key: str) -> int | None:
        """Lấy kích thước (byte) của 1 object trên Filebase, None nếu chưa tồn tại."""
        try:
            resp = self.client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return resp.get("ContentLength")
        except Exception:
            return None

    def upload_project(self, project_dir: Path, project_id: str) -> dict[str, Any]:
        """Upload toàn bộ thư mục project (bao gồm input/ chứa video gốc,
        checkpoints/, output/) lên Filebase.

        Video gốc giờ nằm trong project_dir/input/ (xem project_manager.py:
        create_project()) nên rglob("*") ở đây tự động cuốn nó theo — không
        cần logic upload riêng cho video. Vì video có thể rất lớn và hiếm khi
        đổi giữa các lần sync, ta so kích thước với bản đã có trên cloud
        (head_object) và bỏ qua nếu trùng, để tránh upload lại hàng GB không
        cần thiết mỗi lần đồng bộ.
        """
        if not self.bucket_ready:
            print(f"[filebase] Bỏ qua đồng bộ: bucket '{self.bucket_name}' chưa sẵn sàng "
                  f"(xem lỗi ensure_bucket ở trên).")
            return {"uploaded": 0, "skipped": 0, "errors": 0, "error_files": [],
                    "aborted": True}

        remote_prefix = f"projects/{project_id}/"
        uploaded = []
        skipped = []
        errors = []

        for local_file in project_dir.rglob("*"):
            if local_file.is_dir():
                continue
            if local_file.name.startswith("_") and local_file.name.endswith(".json"):
                # Bỏ qua file meta, sẽ upload riêng ở dưới
                continue
            rel = local_file.relative_to(project_dir)
            remote_key = f"{remote_prefix}{rel.as_posix()}"

            local_size = local_file.stat().st_size
            remote_size = self._remote_size(remote_key)
            if remote_size is not None and remote_size == local_size:
                skipped.append(remote_key)
                continue

            ok = self._upload_file(local_file, remote_key)
            if ok:
                uploaded.append(remote_key)
            else:
                errors.append(remote_key)

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

        return {
            "uploaded": len(uploaded),
            "skipped": len(skipped),
            "errors": len(errors),
            "error_files": errors,
        }

    def download_project(self, project_id: str, local_dir: Path) -> bool:
        """Tải 1 project từ Filebase về thư mục cục bộ (bao gồm input/video)."""
        if not self.bucket_ready:
            print(f"[filebase] Không tải được: bucket '{self.bucket_name}' chưa sẵn sàng.")
            return False
        remote_prefix = f"projects/{project_id}/"
        files = self._list_files(remote_prefix)
        if not files:
            print(f"[filebase] Không tìm thấy project '{project_id}'.")
            return False

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for remote_key in files:
            if remote_key.endswith("/"):
                continue
            rel = remote_key[len(remote_prefix):]
            if not rel or rel.startswith("_"):
                continue
            local_file = local_dir / rel
            ok = self._download_file(remote_key, local_file)
            if ok:
                downloaded += 1
            else:
                print(f"[filebase] Tải thất bại: {rel}")

        print(f"[filebase] Đã tải {downloaded} file về {local_dir}")
        return True

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
    """Tạo instance FilebaseStorage từ section [filebase] trong config.toml."""
    access_key = cfg.get("filebase.access_key", "")
    secret_key = cfg.get("filebase.secret_key", "")
    bucket = cfg.get("filebase.bucket_name", "ai-director-video")
    endpoint = cfg.get("filebase.endpoint_url", "https://s3.filebase.com")

    if not access_key or access_key.startswith("PASTE_"):
        print("[filebase] Chưa cấu hình access_key — tắt tính năng sync Filebase.")
        return None

    try:
        storage = FilebaseStorage(
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket,
            endpoint_url=endpoint,
        )
        if storage.bucket_ready:
            print(f"[filebase] Bucket '{bucket}' sẵn sàng.")
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
