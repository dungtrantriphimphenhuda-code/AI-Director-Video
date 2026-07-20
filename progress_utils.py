"""
progress_utils.py — tiện ích hiển thị tiến trình dùng chung cho toàn bộ pipeline.

Mục tiêu: người dùng luôn thấy được chương trình đang chạy (không bị "treo"
im lặng), đặc biệt ở những bước tốn thời gian mà không có log riêng theo
từng phần (load model, gọi API LLM, chạy ffmpeg...).

Hai công cụ chính:
  - print_progress_bar(...)  : thanh tiến trình dạng % cho các vòng lặp có
    tổng số bước biết trước (VD: "3/12 scene").
  - Heartbeat(...) / run_subprocess(...) : in "vẫn đang chạy... Ns" mỗi vài
    giây trong lúc một lệnh/khối code chạy MÀ KHÔNG có tiến trình rõ ràng
    theo bước (VD: 1 lệnh ffmpeg, 1 lần gọi API, load model nặng).

Không phụ thuộc thư viện ngoài (không cần tqdm) để không phá vỡ requirements.txt.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from typing import Any

# Kết nối "mềm" tới progress_server.STATE (dashboard web xem tiến trình qua
# cloudflared trên CI): import kiểu try/except để module này KHÔNG bắt buộc
# phải có progress_server.py mới chạy được (vd môi trường cũ chưa cập nhật
# file mới) — nếu import lỗi, mọi update bên dưới tự bỏ qua, không ảnh hưởng
# gì tới pipeline chính.
try:
    from progress_server import STATE as _PROGRESS_STATE
except Exception:
    _PROGRESS_STATE = None  # type: ignore[assignment]


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    bar_len: int = 30,
) -> None:
    """
    In (hoặc cập nhật) một thanh tiến trình dạng phần trăm trên cùng 1 dòng
    (dùng \\r để ghi đè, không làm log bị trôi dài).

    Gọi trong vòng lặp với current = 1..total. Khi current >= total, tự động
    xuống dòng để log tiếp theo không bị đè lên thanh tiến trình.
    """
    if total <= 0:
        return
    fraction = min(max(current / total, 0.0), 1.0)
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    line = f"\r{prefix} [{bar}] {fraction * 100:5.1f}% ({current}/{total}) {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


class Heartbeat:
    """
    Context manager chạy nền một thread in "[label] vẫn đang chạy... Ns" mỗi
    `interval` giây, để báo hiệu tiến trình CHƯA bị treo dù không có log
    riêng (VD: đang chờ 1 lệnh ffmpeg hoặc 1 API call trả lời).

    Dùng:
        with Heartbeat("preprocess:extract_audio"):
            subprocess.run(cmd, capture_output=True, check=True)
    """

    def __init__(self, label: str, interval: float = 5.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._start
            sys.stdout.write(f"\r[{self.label}] vẫn đang chạy... {elapsed:5.0f}s   ")
            sys.stdout.flush()

    def __enter__(self) -> "Heartbeat":
        self._start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self._start
        status = "lỗi" if exc_type else "xong"
        sys.stdout.write(f"\r[{self.label}] {status} sau {elapsed:.1f}s" + " " * 15 + "\n")
        sys.stdout.flush()
        return False  # không nuốt exception


def run_subprocess(
    cmd: list[str],
    label: str,
    heartbeat_interval: float = 5.0,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """
    Wrapper quanh subprocess.run(): hiển thị heartbeat trong lúc lệnh đang
    chạy. Hữu ích cho ffmpeg/ffprobe — vốn không in gì ra khi capture_output=True
    nên trước đây người dùng không biết lệnh có bị treo hay không.
    """
    with Heartbeat(label, heartbeat_interval):
        return subprocess.run(cmd, **kwargs)


def run_ffmpeg_with_progress(
    cmd: list[str],
    label: str,
    total_duration: float | None = None,
    heartbeat_interval: float = 5.0,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Chạy 1 lệnh ffmpeg (cmd[0] == "ffmpeg") và hiển thị tiến độ:

      - Nếu biết `total_duration` (giây, VD: thời lượng clip/scene sắp xử lý):
        chèn `-progress pipe:1 -nostats` vào lệnh, đọc `out_time_ms` từ stdout
        để in thanh % tiến độ THẬT (giống print_progress_bar), không phải log giả.
      - Nếu không biết `total_duration` (VD: lệnh không liên quan tới thời lượng
        media, hoặc caller không truyền): dùng Heartbeat làm dự phòng — chỉ báo
        "vẫn đang chạy... Ns" để biết chưa bị treo.

    stderr gốc của ffmpeg luôn được giữ lại (không bị nuốt) để in ra khi lệnh lỗi.
    """
    if not total_duration or total_duration <= 0:
        with Heartbeat(label, heartbeat_interval):
            return subprocess.run(cmd, capture_output=True, check=check)

    # Chèn cờ progress ngay sau tên lệnh "ffmpeg" (không ảnh hưởng các cờ khác).
    progress_cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]

    proc = subprocess.Popen(
        progress_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    time_re = re.compile(r"out_time_ms=(\d+)")
    last_pct = -1
    assert proc.stdout is not None
    assert proc.stderr is not None

    # Đọc stderr ở 1 thread riêng ĐỒNG THỜI với việc đọc stdout ở vòng lặp
    # chính bên dưới. Trước đây stderr chỉ được đọc SAU KHI vòng lặp đọc
    # stdout kết thúc — nếu ffmpeg in đủ nhiều cảnh báo ra stderr để làm đầy
    # buffer pipe của hệ điều hành trong lúc tiến trình chính đang bận đọc
    # stdout, cả 2 bên có thể treo lẫn nhau (ffmpeg chờ ghi, Python chờ đọc).
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_chunks.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    for line in proc.stdout:
        m = time_re.search(line)
        if not m:
            continue
        current_sec = min(int(m.group(1)) / 1_000_000, total_duration)
        pct = int(current_sec / total_duration * 100)
        if pct != last_pct:
            print_progress_bar(
                current_sec, total_duration, prefix=f"[{label}]",
                suffix=f"{current_sec:5.1f}s/{total_duration:.1f}s",
            )
            last_pct = pct
    stderr_thread.join()
    stderr_output = "".join(stderr_chunks)
    returncode = proc.wait()
    if last_pct < 100:
        print_progress_bar(total_duration, total_duration, prefix=f"[{label}]", suffix="hoàn tất")

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, progress_cmd, output=None, stderr=stderr_output)
    return subprocess.CompletedProcess(progress_cmd, returncode, stdout=None, stderr=stderr_output)


class StepTracker:
    """
    Theo dõi tiến trình tổng thể của pipeline (các stage lớn trong run.py),
    in ra "[Bước i/N] tên_stage" kèm thời gian đã trôi qua của từng bước,
    để người dùng biết đang ở đâu trong toàn bộ quá trình chạy.
    """

    def __init__(self, stage_names: list[str]):
        self.stage_names = stage_names
        self.total = len(stage_names)
        self._stage_start = 0.0
        self._pipeline_start = time.time()
        if _PROGRESS_STATE is not None:
            try:
                _PROGRESS_STATE.set_stages(stage_names)
            except Exception:
                pass

    def start(self, stage: str) -> None:
        idx = self.stage_names.index(stage) + 1
        self._stage_start = time.time()
        print("=" * 70)
        print_progress_bar(idx - 1, self.total, prefix="[pipeline]", suffix=f"chuẩn bị: {stage}")
        print(f"[Bước {idx}/{self.total}] Bắt đầu: {stage}")
        print("=" * 70)
        if _PROGRESS_STATE is not None:
            try:
                _PROGRESS_STATE.stage_started(stage, idx - 1, self.total)
            except Exception:
                pass

    def finish(self, stage: str, skipped: bool = False) -> None:
        idx = self.stage_names.index(stage) + 1
        elapsed = time.time() - self._stage_start
        tag = "(đã bỏ qua, dùng checkpoint)" if skipped else f"(mất {elapsed:.1f}s)"
        print_progress_bar(idx, self.total, prefix="[pipeline]", suffix=f"hoàn tất: {stage} {tag}")
        total_elapsed = time.time() - self._pipeline_start
        print(f"[Bước {idx}/{self.total}] Xong: {stage} {tag} | tổng thời gian đã chạy: {total_elapsed / 60:.1f} phút\n")
        if _PROGRESS_STATE is not None:
            try:
                _PROGRESS_STATE.stage_finished(stage, idx, self.total, elapsed, skipped=skipped)
            except Exception:
                pass
