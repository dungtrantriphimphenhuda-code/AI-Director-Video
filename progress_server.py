"""
progress_server.py — server HTTP nhẹ (chỉ dùng thư viện chuẩn, không thêm
dependency) để xem tiến trình pipeline qua trình duyệt, kèm dự đoán thời
gian hoàn thành (ETA) và log trực tiếp (live log).

Thiết kế cho GitHub Actions: máy ảo CI không có domain/IP public, nên tự
mở cổng HTTP nội bộ (mặc định 8787) rồi để bước riêng trong workflow dùng
`cloudflared tunnel --url http://localhost:8787` (Cloudflare Quick Tunnel —
KHÔNG cần tài khoản/API token) để lấy 1 link `https://xxxx.trycloudflare.com`
công khai tạm thời, xem được ngay trên điện thoại/máy khác trong lúc job
đang chạy.

Cách hoạt động (KHÔNG đổi logic pipeline, chỉ thêm lớp quan sát bên ngoài):
  - `STATE` là 1 instance `ProgressState` dùng chung (singleton) cho cả
    process. `progress_utils.StepTracker` tự cập nhật vào đây mỗi khi 1
    stage bắt đầu/kết thúc (xem progress_utils.py) — KHÔNG cần sửa gì ở
    run.py/ci_run_latest_project.py để việc theo dõi từng stage hoạt động.
  - `LOG` là 1 ring-buffer log dùng chung: `start_server()` tự "tee"
    sys.stdout/sys.stderr (vẫn in ra console CI như cũ, KHÔNG mất log gốc)
    đồng thời chép từng dòng vào ring-buffer này. Vì vậy mọi print() có sẵn
    trong toàn bộ pipeline (asr/vision/render/...) tự động xuất hiện trên
    dashboard mà không cần sửa 1 dòng code nào ở nơi khác.
  - Trang `/` là 1 dashboard tĩnh (không cần thư viện ngoài), tự làm mới:
      * `/status.json`  — vẫn như cũ, fetch mỗi ~2.5s (nhẹ, JSON nhỏ).
      * `/log.json?since=N` — CHỈ trả về các dòng log MỚI có seq > N (kèm
        `latest_seq` để lần fetch sau tiếp tục từ đó). Nhờ vậy payload luôn
        nhỏ, không phải gửi lại toàn bộ log tích luỹ mỗi lần (nguyên nhân
        gây lag/giật ở bản cũ khi log dài ra theo thời gian).
  - `start_server(port)` mở 1 `ThreadingHTTPServer` chạy nền (daemon thread),
    trả về server để có thể `.shutdown()` khi pipeline chạy xong.

ETA: dự đoán bằng thời lượng trung bình của các stage ĐÃ CHẠY THẬT (bỏ qua
  stage skip vì có checkpoint, do gần như tức thời và không phản ánh chi phí
  thật) nhân với số stage còn lại. Ở stage đầu tiên (chưa có stage nào chạy
  xong để lấy trung bình), dùng bảng trọng số ước lượng tĩnh (đo từ vài lần
  chạy thật, xem `_DEFAULT_STAGE_WEIGHTS`) để vẫn hiện được ETA thô ngay từ
  đầu thay vì để trống — chỉ mang tính tham khảo, không chính xác tuyệt đối.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, parse_qs

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


class LogBuffer:
    """
    Ring-buffer log dùng chung, an toàn giữa nhiều thread.

    Mỗi dòng có 1 số thứ tự (seq) tăng dần. Client chỉ cần nhớ seq lớn nhất
    đã nhận rồi hỏi "cho tôi các dòng có seq > since" (`snapshot_since`) —
    KHÔNG cần gửi lại toàn bộ log mỗi lần, nên payload luôn nhỏ & mượt dù
    pipeline chạy hàng giờ liền (khác bản cũ: không có log, hoặc nếu lấy
    nguyên khối log CI sẽ ngày càng nặng và giật/lag).

    Cũng theo dõi riêng 1 "dòng đang gõ dở" (`partial`) để phản ánh các dòng
    dùng \\r để tự ghi đè (progress bar, heartbeat...) mà KHÔNG làm phình
    to ring-buffer — mỗi lần \\r chỉ cập nhật tại chỗ, không tạo dòng mới.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self._lines: list[tuple[int, float, str]] = []  # (seq, ts, text)
        self._seq = 0
        self._partial = ""

    def commit_line(self, text: str, overwrite: bool = False) -> None:
        text = text.rstrip("\n")
        with self._lock:
            self._partial = ""
            if text == "":
                return
            if overwrite and self._lines:
                self._seq += 1
                _, ts, _ = self._lines[-1]
                self._lines[-1] = (self._seq, ts, text)
            else:
                self._seq += 1
                self._lines.append((self._seq, time.time(), text))
                if len(self._lines) > self._maxlen:
                    del self._lines[: len(self._lines) - self._maxlen]

    def set_partial(self, text: str) -> None:
        with self._lock:
            self._partial = text

    def snapshot_since(self, since_seq: int, max_lines: int = 300) -> dict[str, Any]:
        with self._lock:
            new_lines = [(s, ts, tx) for (s, ts, tx) in self._lines if s > since_seq]
            if len(new_lines) > max_lines:
                new_lines = new_lines[-max_lines:]
            latest_seq = self._lines[-1][0] if self._lines else since_seq
            return {
                "lines": [{"seq": s, "ts": ts, "text": tx} for (s, ts, tx) in new_lines],
                "latest_seq": latest_seq,
                "partial": self._partial,
            }


STATE = ProgressState()
LOG = LogBuffer()


class _TeeStream:
    """Bọc ngoài 1 stream (stdout/stderr) thật: vẫn ghi ra console CI như
    bình thường (KHÔNG mất log gốc trong GitHub Actions), đồng thời chép
    song song từng dòng vào `LOG` để dashboard hiển thị live."""

    def __init__(self, real_stream: Any, log_buffer: LogBuffer) -> None:
        self._real = real_stream
        self._log = log_buffer
        self._partial = ""

    def write(self, s: str) -> int:
        try:
            self._real.write(s)
        except Exception:
            pass
        try:
            self._feed(s)
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()

    def _feed(self, s: str) -> None:
        for ch in s:
            if ch == "\n":
                self._log.commit_line(self._partial, overwrite=False)
                self._partial = ""
            elif ch == "\r":
                self._log.commit_line(self._partial, overwrite=True)
                self._partial = ""
            else:
                self._partial += ch
        # Dòng đang gõ dở (chưa \n/\r) vẫn hiện ngay trên dashboard dạng
        # "preview" mờ, để không phải chờ heartbeat/progress bar in xong.
        self._log.set_partial(self._partial)


_tee_installed = False
_tee_lock = threading.Lock()


def install_log_tee() -> None:
    """Gắn LogBuffer vào sys.stdout/sys.stderr (idempotent — gọi nhiều lần
    vẫn an toàn, chỉ gắn 1 lần duy nhất cho cả process)."""
    global _tee_installed
    with _tee_lock:
        if _tee_installed:
            return
        sys.stdout = _TeeStream(sys.stdout, LOG)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, LOG)  # type: ignore[assignment]
        _tee_installed = True


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
<title>AI Director Video — Pipeline Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700;900&family=Roboto+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0c10; --panel:#12151c; --panel-2:#161a23; --border:#232733;
    --text:#e7e9ee; --muted:#8b93a7; --accent:#6d8dff; --accent-2:#22d3ee;
    --ok:#3ddc84; --warn:#ffb454; --err:#ff5d6c;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{
    margin:0; font-family:'Roboto',-apple-system,Segoe UI,sans-serif;
    background:
      radial-gradient(900px 500px at 12% -10%, rgba(109,141,255,.16), transparent 60%),
      radial-gradient(700px 500px at 110% 10%, rgba(34,211,238,.12), transparent 60%),
      var(--bg);
    color:var(--text); min-height:100%; padding:2.2rem 1.1rem 3rem;
  }
  .wrap{max-width:880px; margin:0 auto;}
  .top{display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.4rem; flex-wrap:wrap;}
  .brand{display:flex; align-items:center; gap:.7rem;}
  .brand .dot-logo{
    width:34px; height:34px; border-radius:10px; flex:none;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    box-shadow:0 0 0 1px rgba(255,255,255,.08), 0 6px 18px -6px rgba(109,141,255,.6);
    position:relative; overflow:hidden;
  }
  .brand .dot-logo::after{
    content:""; position:absolute; inset:0;
    background:linear-gradient(120deg, transparent 30%, rgba(255,255,255,.55) 50%, transparent 70%);
    background-size:220% 100%; animation:sheen 3.2s ease-in-out infinite;
  }
  @keyframes sheen{0%{background-position:160% 0;}100%{background-position:-60% 0;}}
  .brand h1{font-size:1.05rem; font-weight:700; margin:0; letter-spacing:.2px;}
  .brand span.sub{display:block; font-size:.72rem; color:var(--muted); font-weight:500; margin-top:1px;}

  .badge{
    display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .75rem;
    border-radius:999px; background:var(--panel); border:1px solid var(--border);
    font-size:.78rem; font-weight:600; color:var(--muted); white-space:nowrap;
  }
  .badge .ping{width:8px; height:8px; border-radius:50%; background:var(--muted); position:relative;}
  .badge.running .ping{background:var(--accent);}
  .badge.running .ping::after{
    content:""; position:absolute; inset:-4px; border-radius:50%; border:2px solid var(--accent);
    animation:ping 1.4s cubic-bezier(0,0,.2,1) infinite;
  }
  .badge.done{color:var(--ok);} .badge.done .ping{background:var(--ok);}
  .badge.error{color:var(--err);} .badge.error .ping{background:var(--err);}
  @keyframes ping{0%{transform:scale(1); opacity:.9;}75%,100%{transform:scale(2.4); opacity:0;}}

  .card{
    background:linear-gradient(180deg, var(--panel-2), var(--panel));
    border:1px solid var(--border); border-radius:16px; padding:1.2rem 1.3rem;
    margin-bottom:1rem; box-shadow:0 12px 30px -18px rgba(0,0,0,.6);
  }
  .stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.9rem; margin-bottom:1rem;}
  .stat{background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:.7rem .85rem;}
  .stat .k{font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:600;}
  .stat .v{font-size:1.02rem; font-weight:700; margin-top:.2rem; word-break:break-word;}

  .bar-outer{background:#1a1e28; border:1px solid var(--border); border-radius:999px; height:16px; overflow:hidden; position:relative;}
  .bar-inner{
    height:100%; width:0%; border-radius:999px; position:relative;
    background:linear-gradient(90deg,var(--accent),var(--accent-2));
    transition:width .7s cubic-bezier(.2,.7,.2,1);
  }
  .bar-inner::after{
    content:""; position:absolute; inset:0;
    background:repeating-linear-gradient(45deg, rgba(255,255,255,.18) 0 10px, transparent 10px 20px);
    background-size:28px 28px; animation:stripes 1.1s linear infinite; opacity:.55;
  }
  @keyframes stripes{to{background-position:28px 0;}}
  .bar-label{display:flex; justify-content:space-between; font-size:.75rem; color:var(--muted); margin-top:.4rem;}
  .bar-label b{color:var(--text);}

  .card h2{font-size:.85rem; margin:0 0 .8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; font-weight:700;}
  .stage-list{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.4rem;}
  .stage-list li{
    display:flex; align-items:center; gap:.65rem; padding:.55rem .7rem; border-radius:10px;
    background:var(--panel); border:1px solid transparent; font-size:.86rem; color:var(--muted);
    transition:background .3s ease, border-color .3s ease;
  }
  .stage-list li .icon{width:18px; height:18px; flex:none; display:flex; align-items:center; justify-content:center;}
  .stage-list li.done{color:var(--ok);}
  .stage-list li.done .icon svg{stroke:var(--ok);}
  .stage-list li.skip{color:var(--accent-2);}
  .stage-list li.current{background:rgba(109,141,255,.1); border-color:rgba(109,141,255,.35); color:var(--text); font-weight:600;}
  .spinner{
    width:14px; height:14px; border-radius:50%; border:2px solid rgba(109,141,255,.25);
    border-top-color:var(--accent); animation:spin .8s linear infinite;
  }
  @keyframes spin{to{transform:rotate(360deg);}}
  .stage-name{flex:1;}
  .stage-meta{font-size:.72rem; color:var(--muted); font-weight:500;}

  .logcard{padding:0; overflow:hidden;}
  .logcard .loghead{
    display:flex; align-items:center; justify-content:space-between; gap:.6rem;
    padding:.8rem 1rem; border-bottom:1px solid var(--border); background:var(--panel-2);
  }
  .logcard .loghead .left{display:flex; align-items:center; gap:.55rem; font-size:.78rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.06em;}
  .livedot{width:7px; height:7px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 0 rgba(61,220,132,.6); animation:pulse 1.6s infinite;}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(61,220,132,.55);}70%{box-shadow:0 0 0 7px rgba(61,220,132,0);}100%{box-shadow:0 0 0 0 rgba(61,220,132,0);}}
  #logbox{
    height:340px; overflow-y:auto; padding:.8rem 1rem 1rem; font-family:'Roboto Mono',ui-monospace,monospace;
    font-size:.78rem; line-height:1.55; background:#0b0d13;
  }
  #logbox .line{white-space:pre-wrap; word-break:break-word; opacity:0; animation:fadein .25s ease forwards; color:#c7cbd6;}
  #logbox .line.partial{color:#6b7280; font-style:italic;}
  #logbox .line.hl-err{color:var(--err);}
  #logbox .line.hl-ok{color:var(--ok);}
  #logbox .line.hl-step{color:var(--accent-2); font-weight:600;}
  @keyframes fadein{from{opacity:0; transform:translateY(2px);}to{opacity:1; transform:translateY(0);}}
  #jump{
    position:sticky; bottom:.6rem; margin-left:1rem; display:none; align-items:center; gap:.4rem;
    background:var(--accent); color:#fff; border:none; padding:.35rem .7rem; border-radius:999px;
    font-size:.72rem; font-weight:700; cursor:pointer; box-shadow:0 6px 16px -6px rgba(109,141,255,.7);
  }
  footer{color:#5b6273; font-size:.72rem; text-align:center; margin-top:1.2rem;}
  footer code{color:var(--accent-2);}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="dot-logo"></div>
      <div>
        <h1>AI Director Video</h1>
        <span class="sub">Pipeline Dashboard · CI</span>
      </div>
    </div>
    <div class="badge" id="statusBadge"><span class="ping"></span><span id="statusText">đang kết nối...</span></div>
  </div>

  <div class="card">
    <div class="stats">
      <div class="stat"><div class="k">Project</div><div class="v" id="project">—</div></div>
      <div class="stat"><div class="k">Bước hiện tại</div><div class="v" id="currentStage">—</div></div>
      <div class="stat"><div class="k">Đã chạy</div><div class="v" id="elapsed">—</div></div>
      <div class="stat"><div class="k">Dự kiến còn lại</div><div class="v" id="eta">—</div></div>
    </div>
    <div class="bar-outer"><div class="bar-inner" id="bar"></div></div>
    <div class="bar-label"><span id="barPct">0%</span><span id="etaBasis" style="text-align:right"></span></div>
  </div>

  <div class="card">
    <h2>Các bước pipeline</h2>
    <ul class="stage-list" id="stages"></ul>
  </div>

  <div class="card logcard">
    <div class="loghead">
      <div class="left"><span class="livedot"></span> Live log</div>
      <div class="left" id="logCount">0 dòng</div>
    </div>
    <div id="logbox"></div>
    <button id="jump" onclick="stickBottom=true;scrollLog();">↓ Cuộn xuống mới nhất</button>
  </div>

  <footer>Trang tự làm mới liên tục qua <code>/status.json</code> và <code>/log.json</code>. Link này chỉ tồn tại trong lúc job GitHub Actions đang chạy (Cloudflare Quick Tunnel).</footer>
</div>

<script>
const ICON_DONE = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_SKIP = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M5 4l10 8-10 8V4z" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/><path d="M17 4v16" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>';
const ICON_PENDING = '<svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="none" stroke="#3a3f4d" stroke-width="2"/></svg>';

let lastSeq = 0;
let stickBottom = true;
const logbox = document.getElementById('logbox');
const jumpBtn = document.getElementById('jump');
logbox.addEventListener('scroll', () => {
  const nearBottom = logbox.scrollHeight - logbox.scrollTop - logbox.clientHeight < 40;
  stickBottom = nearBottom;
  jumpBtn.style.display = nearBottom ? 'none' : 'inline-flex';
});

function classifyLine(text){
  const t = text.toLowerCase();
  if (t.includes('lỗi') || t.includes('error') || t.includes('##[error]')) return 'hl-err';
  if (t.includes('xong') || t.includes('hoàn tất') || t.includes('done')) return 'hl-ok';
  if (t.startsWith('[bước') || t.startsWith('===')) return 'hl-step';
  return '';
}

function scrollLog(){
  if (stickBottom) logbox.scrollTop = logbox.scrollHeight;
}

let partialEl = null;
async function refreshLog(){
  try{
    const r = await fetch('/log.json?since=' + lastSeq, {cache:'no-store'});
    const d = await r.json();
    if (d.lines && d.lines.length){
      const frag = document.createDocumentFragment();
      d.lines.forEach(l => {
        const div = document.createElement('div');
        div.className = 'line ' + classifyLine(l.text);
        div.textContent = l.text;
        frag.appendChild(div);
      });
      if (partialEl) { partialEl.remove(); partialEl = null; }
      logbox.appendChild(frag);
      // giữ tối đa ~400 dòng trên DOM (thay dòng cũ, tránh phình to & lag)
      while (logbox.children.length > 400) logbox.removeChild(logbox.firstChild);
      lastSeq = d.latest_seq;
      document.getElementById('logCount').textContent = lastSeq + ' dòng';
    }
    if (d.partial){
      if (!partialEl){ partialEl = document.createElement('div'); partialEl.className='line partial'; logbox.appendChild(partialEl); }
      partialEl.textContent = d.partial;
    } else if (partialEl){
      partialEl.remove(); partialEl = null;
    }
    scrollLog();
  } catch(e) { /* server có thể chưa sẵn sàng, thử lại ở lần sau */ }
}

async function refreshStatus() {
  try {
    const r = await fetch('/status.json', {cache: 'no-store'});
    const d = await r.json();
    document.getElementById('project').textContent = d.project_id || '(chưa rõ)';
    document.getElementById('currentStage').textContent = d.current_stage || '—';

    const badge = document.getElementById('statusBadge');
    const label = {starting:'đang khởi động', running:'đang chạy', done:'hoàn tất', error:'lỗi'}[d.status] || d.status;
    document.getElementById('statusText').textContent = label;
    badge.className = 'badge ' + (d.status || '');

    const pct = d.total_stages ? Math.round(100 * d.current_index / d.total_stages) : 0;
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('barPct').textContent = pct + '%';
    document.getElementById('elapsed').textContent = fmt(d.elapsed_seconds);
    document.getElementById('eta').textContent = d.eta_seconds === null ? '—' : fmt(d.eta_seconds);
    document.getElementById('etaBasis').textContent = d.eta_seconds === null ? d.eta_basis : ('(' + d.eta_basis + ')');

    const ul = document.getElementById('stages');
    ul.innerHTML = '';
    (d.stage_names || []).forEach((name) => {
      const li = document.createElement('li');
      const icon = document.createElement('span'); icon.className = 'icon';
      const nameEl = document.createElement('span'); nameEl.className = 'stage-name'; nameEl.textContent = name;
      const meta = document.createElement('span'); meta.className = 'stage-meta';
      if (d.stage_durations[name] !== undefined) {
        li.className = 'done'; icon.innerHTML = ICON_DONE; meta.textContent = 'xong · ' + fmt(d.stage_durations[name]);
      } else if ((d.skipped_stages || []).includes(name)) {
        li.className = 'skip'; icon.innerHTML = ICON_SKIP; meta.textContent = 'bỏ qua (checkpoint)';
      } else if (name === d.current_stage && d.status === 'running') {
        li.className = 'current'; icon.innerHTML = '<span class="spinner"></span>'; meta.textContent = 'đang chạy...';
      } else {
        icon.innerHTML = ICON_PENDING; meta.textContent = 'chờ';
      }
      li.appendChild(icon); li.appendChild(nameEl); li.appendChild(meta);
      ul.appendChild(li);
    });
  } catch (e) { /* server có thể chưa sẵn sàng, thử lại ở lần sau */ }
}

function fmt(sec) {
  if (sec === null || sec === undefined) return '—';
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
  if (h) return h + 'h' + String(m).padStart(2,'0') + 'm';
  if (m) return m + 'm' + String(s).padStart(2,'0') + 's';
  return s + 's';
}

refreshStatus(); refreshLog();
setInterval(refreshStatus, 2500);
setInterval(refreshLog, 1200);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # im lặng, khỏi rác log CI
        pass

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path.startswith("/status.json"):
            self._send_json(STATE.snapshot())
            return
        if parsed.path.startswith("/log.json"):
            qs = parse_qs(parsed.query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except (ValueError, IndexError):
                since = 0
            self._send_json(LOG.snapshot_since(since))
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
    giá trị trả về khi pipeline chạy xong để đóng gọn gàng.

    Cũng tự gắn log-tee vào sys.stdout/sys.stderr (idempotent) để mọi
    print() có sẵn trong pipeline tự động chảy vào dashboard live log,
    không cần sửa gì ở nơi khác.
    """
    install_log_tee()
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[progress] Server tiến trình đang chạy tại http://localhost:{port} "
          f"(dùng cloudflared để lấy link công khai tạm thời).")
    return server
