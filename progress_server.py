"""
progress_server.py — server HTTP nhẹ (chỉ dùng thư viện chuẩn, không thêm
dependency) để xem tiến trình pipeline qua trình duyệt, kèm dự đoán thời
gian hoàn thành (ETA).

Thiết kế cho GitHub Actions: máy ảo CI không có domain/IP public, nên tự
mở cổng HTTP nội bộ (mặc định 8787) rồi để bước riêng trong workflow dùng
`cloudflared tunnel --url http://localhost:8787` (Cloudflare Quick Tunnel —
KHÔNG cần tài khoản/API token) để lấy 1 link `https://xxxx.trycloudflare.com`
công khai tạm thời, xem được ngay trên điện thoại/máy khác trong lúc job
đang chạy.

Cách hoạt động:
  - `STATE` là 1 instance `ProgressState` dùng chung (singleton) cho cả
    process. `progress_utils.StepTracker` tự cập nhật vào đây mỗi khi 1
    stage bắt đầu/kết thúc (xem progress_utils.py) — KHÔNG cần sửa gì ở
    run.py/ci_run_latest_project.py để việc theo dõi từng stage hoạt động.
  - `start_server(port)` mở 1 `ThreadingHTTPServer` chạy nền (daemon thread),
    trả về server để có thể `.shutdown()` khi pipeline chạy xong.
  - Trang `/` là HTML tự làm mới (fetch `/status.json` mỗi 3s bằng JS thuần,
    không cần thư viện ngoài).

ETA: dự đoán bằng thời lượng trung bình của các stage ĐÃ CHẠY THẬT (bỏ qua
  stage skip vì có checkpoint, do gần như tức thời và không phản ánh chi phí
  thật) nhân với số stage còn lại. Ở stage đầu tiên (chưa có stage nào chạy
  xong để lấy trung bình), dùng bảng trọng số ước lượng tĩnh (đo từ vài lần
  chạy thật, xem `_DEFAULT_STAGE_WEIGHTS`) để vẫn hiện được ETA thô ngay từ
  đầu thay vì để trống — chỉ mang tính tham khảo, không chính xác tuyệt đối.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Trọng số tương đối THÔ giữa các stage (đo kinh nghiệm: vision + tts + render
# thường tốn thời gian nhất). Chỉ dùng để ước tính ETA trước khi có stage nào
# chạy xong THẬT trong lần chạy hiện tại; sau đó ETA chuyển sang dùng số đo
# thật của chính lần chạy này (chính xác hơn nhiều vì phụ thuộc độ dài video).
_DEFAULT_STAGE_WEIGHTS: dict[str, float] = {
    "preprocess": 0.05,
    "asr": 0.10,
    "vision": 0.25,
    "semantic_graph": 0.03,
    "reference": 0.02,
    "script": 0.15,
    "tts": 0.20,
    "render": 0.20,
}


class ProgressState:
    """Trạng thái tiến trình dùng chung, an toàn giữa nhiều thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.project_id: str = ""
        self.stage_names: list[str] = []
        self.total: int = 0
        self.current_index: int = 0
        self.current_stage: str = ""
        self.pipeline_start: float = time.time()
        self._stage_start_ts: float = time.time()
        self.stage_durations: dict[str, float] = {}   # stage đã XONG (không tính skip) -> giây
        self.skipped_stages: set[str] = set()
        self.status: str = "starting"   # starting | running | done | error
        self.error_message: str | None = None
        self.finished_at: float | None = None

    # -- hooks gọi từ progress_utils.StepTracker --------------------------
    def set_stages(self, stage_names: list[str], project_id: str = "") -> None:
        with self._lock:
            self.stage_names = list(stage_names)
            self.total = len(stage_names)
            self.project_id = project_id or self.project_id
            self.status = "running"

    def stage_started(self, stage: str, idx: int, total: int) -> None:
        with self._lock:
            self.current_stage = stage
            self.current_index = idx
            self.total = total
            self._stage_start_ts = time.time()
            self.status = "running"

    def stage_finished(self, stage: str, idx: int, total: int, elapsed: float, skipped: bool = False) -> None:
        with self._lock:
            self.current_index = idx
            self.total = total
            if skipped:
                self.skipped_stages.add(stage)
            else:
                self.stage_durations[stage] = elapsed

    def mark_done(self) -> None:
        with self._lock:
            self.status = "done"
            self.finished_at = time.time()

    def mark_error(self, message: str) -> None:
        with self._lock:
            self.status = "error"
            self.error_message = message
            self.finished_at = time.time()

    # -- tính toán để hiển thị ---------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed_total = now - self.pipeline_start
            done_real = [s for s in self.stage_durations if s not in self.skipped_stages or s in self.stage_durations]
            real_durations = list(self.stage_durations.values())

            remaining_stages = max(self.total - self.current_index, 0)
            eta_seconds: float | None
            eta_basis: str
            if real_durations:
                avg = sum(real_durations) / len(real_durations)
                eta_seconds = avg * remaining_stages
                eta_basis = "đo thật từ lần chạy này"
            elif self.stage_names:
                # Chưa có stage thật nào xong -> dùng bảng trọng số tĩnh, quy
                # ra tỉ lệ % còn lại rồi ước lượng thô so với thời gian đã
                # trôi qua của stage hiện tại (chỉ mang tính tham khảo).
                total_weight = sum(_DEFAULT_STAGE_WEIGHTS.get(s, 1 / max(len(self.stage_names), 1))
                                    for s in self.stage_names) or 1.0
                remaining_names = self.stage_names[self.current_index:] if self.current_index < len(self.stage_names) else []
                remaining_weight = sum(_DEFAULT_STAGE_WEIGHTS.get(s, 1 / max(len(self.stage_names), 1))
                                        for s in remaining_names)
                current_weight = _DEFAULT_STAGE_WEIGHTS.get(self.current_stage, 1 / max(len(self.stage_names), 1))
                cur_elapsed = now - self._stage_start_ts
                # Suy ra "thời gian cho 1 đơn vị trọng số" từ stage đang chạy dở.
                per_weight_sec = (cur_elapsed / current_weight) if current_weight > 0 else None
                eta_seconds = (per_weight_sec * remaining_weight) if per_weight_sec else None
                eta_basis = "ước lượng thô (chưa có stage nào xong để đo thật)"
            else:
                eta_seconds = None
                eta_basis = "chưa có dữ liệu"

            return {
                "project_id": self.project_id,
                "status": self.status,
                "current_stage": self.current_stage,
                "current_index": self.current_index,
                "total_stages": self.total,
                "stage_names": self.stage_names,
                "elapsed_seconds": round(elapsed_total, 1),
                "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                "eta_basis": eta_basis,
                "stage_durations": {k: round(v, 1) for k, v in self.stage_durations.items()},
                "skipped_stages": sorted(self.skipped_stages),
                "error_message": self.error_message,
                "updated_at": now,
            }


STATE = ProgressState()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "đang ước tính..."
    seconds = max(seconds, 0)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Director Video — tiến trình pipeline</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1115; color:#e6e6e6;
         max-width:720px; margin:2rem auto; padding:0 1rem; }
  h1 { font-size:1.2rem; color:#8ab4ff; }
  .bar-outer { background:#22252c; border-radius:8px; height:22px; overflow:hidden; margin:0.5rem 0 1rem; }
  .bar-inner { background:linear-gradient(90deg,#4f8cff,#8ab4ff); height:100%; width:0%;
               transition:width 0.6s ease; text-align:right; }
  .stage-list { list-style:none; padding:0; }
  .stage-list li { padding:0.35rem 0.6rem; margin:0.15rem 0; border-radius:6px; background:#1a1c22; }
  .stage-list li.done { color:#7be08a; }
  .stage-list li.skip { color:#8ab4ff; }
  .stage-list li.current { background:#2a2f3d; color:#ffd479; font-weight:600; }
  .stat { display:flex; justify-content:space-between; padding:0.2rem 0; color:#a9adba; }
  .status-ok { color:#7be08a; } .status-err { color:#ff6b6b; }
  code { color:#8ab4ff; }
</style>
</head>
<body>
  <h1>AI Director Video — tiến trình pipeline (CI)</h1>
  <div class="stat"><span>Project</span><span id="project">-</span></div>
  <div class="stat"><span>Trạng thái</span><span id="status">-</span></div>
  <div class="bar-outer"><div class="bar-inner" id="bar"></div></div>
  <div class="stat"><span>Đã chạy</span><span id="elapsed">-</span></div>
  <div class="stat"><span>Dự kiến còn lại (ETA)</span><span id="eta">-</span></div>
  <ul class="stage-list" id="stages"></ul>
  <p style="color:#666;font-size:0.8rem">Trang tự làm mới mỗi 3 giây. Link này chỉ tồn tại trong lúc job GitHub
  Actions đang chạy (Cloudflare Quick Tunnel).</p>
<script>
async function refresh() {
  try {
    const r = await fetch('/status.json', {cache: 'no-store'});
    const d = await r.json();
    document.getElementById('project').textContent = d.project_id || '(chưa rõ)';
    const statusEl = document.getElementById('status');
    statusEl.textContent = d.status;
    statusEl.className = d.status === 'error' ? 'status-err' : (d.status === 'done' ? 'status-ok' : '');
    const pct = d.total_stages ? Math.round(100 * d.current_index / d.total_stages) : 0;
    const bar = document.getElementById('bar');
    bar.style.width = pct + '%';
    bar.textContent = pct + '%';
    document.getElementById('elapsed').textContent = fmt(d.elapsed_seconds);
    document.getElementById('eta').textContent = d.eta_seconds === null ? d.eta_basis : (fmt(d.eta_seconds) + ' (' + d.eta_basis + ')');
    const ul = document.getElementById('stages');
    ul.innerHTML = '';
    (d.stage_names || []).forEach((name, i) => {
      const li = document.createElement('li');
      let label = name;
      if (d.stage_durations[name] !== undefined) { li.className = 'done'; label += ' — xong (' + fmt(d.stage_durations[name]) + ')'; }
      if ((d.skipped_stages || []).includes(name)) { li.className = 'skip'; label += ' — bỏ qua (đã có checkpoint)'; }
      if (name === d.current_stage && d.status === 'running') { li.className = 'current'; label += ' — đang chạy...'; }
      li.textContent = label;
      ul.appendChild(li);
    });
  } catch (e) { /* server có thể chưa sẵn sàng, thử lại ở lần sau */ }
}
function fmt(sec) {
  if (sec === null || sec === undefined) return '-';
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
  if (h) return h + 'h' + String(m).padStart(2,'0') + 'm';
  if (m) return m + 'm' + String(s).padStart(2,'0') + 's';
  return s + 's';
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # im lặng, khỏi rác log CI
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/status.json"):
            body = json.dumps(STATE.snapshot(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # mọi path khác -> trang HTML chính
        body = _PAGE_TEMPLATE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(port: int = 8787) -> ThreadingHTTPServer:
    """Mở server tiến trình chạy nền (daemon thread). Gọi `.shutdown()` trên
    giá trị trả về khi pipeline chạy xong để đóng gọn gàng."""
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[progress] Server tiến trình đang chạy tại http://localhost:{port} "
          f"(dùng cloudflared để lấy link công khai tạm thời).")
    return server
