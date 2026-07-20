"""
config.py — bộ đọc cấu hình trung tâm.

Toàn bộ project đọc cấu hình từ `config.toml` thông qua module này.
Không còn bất kỳ chỗ nào dùng os.getenv() hoặc python-dotenv.

Dùng `tomllib` (built-in từ Python 3.11+), fallback về thư viện `tomli`
cho Python < 3.11 (bắt buộc trong requirements.txt).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    import tomli as tomllib  # type: ignore[import-not-found]


class Config:
    """
    Wrapper mỏng quanh dict đã parse từ config.toml.

    Cho phép truy cập kiểu `cfg.get("api.cerebras_api_key")` (dot-path)
    hoặc `cfg["api"]["cerebras_api_key"]` (dict thường).
    """

    def __init__(self, data: dict[str, Any], config_path: Path):
        self._data = data
        self.config_path = config_path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Đọc giá trị theo đường dẫn dạng 'section.key', trả về default nếu thiếu."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Trả về toàn bộ một section (vd: cfg.section('processing'))."""
        return self._data.get(name, {})

    def set(self, dotted_key: str, value: Any) -> None:
        """Ghi đè 1 giá trị trong bộ nhớ (KHÔNG ghi xuống config.toml trên đĩa).
        Dùng khi cần cập nhật cấu hình lúc chạy (vd: người dùng nhập lại đường
        dẫn video vì đường dẫn trong config.toml không tồn tại)."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, dotted_key: str, default: str | None = None) -> Path:
        """
        Đọc một giá trị đường dẫn từ config và chuẩn hoá thành Path tuyệt đối,
        tương đối theo thư mục chứa config.toml (không phải theo cwd hiện tại).
        """
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"Missing required path config: {dotted_key}")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self.config_path.parent / p).resolve()
        return p

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def _is_running_in_github_actions() -> bool:
    """
    True khi và chỉ khi đang chạy trong 1 job của GitHub Actions.

    GitHub tự set biến môi trường GITHUB_ACTIONS="true" cho MỌI job, không
    cần workflow khai báo gì thêm. Biến này KHÔNG tồn tại trên máy cá nhân,
    Colab, hay VPS Linux tự chạy tay/cron — nên dùng nó làm điều kiện là an
    toàn 100%: override bên dưới chỉ có tác dụng trên GitHub Actions, không
    bao giờ vô tình kích hoạt ở nơi khác.
    """
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


# Runner miễn phí của GitHub Actions (kể cả sau khi nâng lên 4 vCPU/16GB RAM
# cuối 2023) vẫn CHỈ đảm bảo ~14GB disk trống (GitHub xác nhận chính thức:
# https://github.com/actions/runner-images/discussions/9329). Các backend
# "local" (tải model AI về chạy ngay trên máy) dễ dàng vượt quá con số này
# khi cộng dồn: Qwen3-4B-Instruct-2507 (~8GB), Qwen3-VL-4B (~8GB),
# funasr (paraformer-zh + fsmn-vad + ct-punc, vài GB), moondream2 (~4GB) —
# đó là lý do job hay bị "The operation was canceled" ngay sau đoạn
# tải+load model: hết dung lượng đĩa (hoặc RAM) khiến runner bị kill giữa
# chừng, không phải ai đó bấm huỷ workflow.
#
# fallback_key: tên key trong [api] mà backend nhẹ thay thế cần có giá trị
# không rỗng thì mới an toàn để tự động chuyển sang (nếu không có, coi như
# secret đó chưa được cấu hình trên repo -> KHÔNG override, để nguyên giá
# trị người dùng chọn và chỉ in cảnh báo, tránh đổi sang 1 backend chắc
# chắn sẽ lỗi vì thiếu key).
_CI_LIGHTWEIGHT_OVERRIDES = [
    # (dotted_key gốc, {giá trị "nặng" -> (giá trị nhẹ thay thế, dotted_key api-key cần có)})
    ("api.script_backend", {
        "local": ("cerebras", "api.cerebras_api_key"),
    }),
    ("processing.vision_backend", {
        "local": ("mistral", "api.mistral_api_key"),
        "moondream": ("mistral", "api.mistral_api_key"),
    }),
    ("processing.asr_backend", {
        # funasr tự tải riêng 3 model (ASR + VAD + dấu câu) -> luôn ép về
        # faster-whisper trên CI, không cần api-key nào nên fallback_key
        # để None (luôn coi là an toàn để override).
        "funasr": ("whisper", None),
    }),
]


def _apply_ci_safe_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """
    Nếu đang chạy trong GitHub Actions, tự động thay các backend "local"
    (tải + chạy model AI ngay trên máy, tốn nhiều disk/RAM) bằng backend
    API tương đương, miễn là secret cần thiết đã có sẵn trong config đã
    render. Không đụng gì tới config nếu KHÔNG chạy trong GitHub Actions —
    chạy trên máy thật/Colab/VPS Linux tự quản lý vẫn dùng đúng backend
    người dùng đã chọn trong config.toml, không thay đổi hành vi.
    """
    if not _is_running_in_github_actions():
        return data

    # [ci] force_lightweight_backends = false: người dùng đã tự xác nhận
    # runner GitHub Actions (vd 4 vCPU / 16GB RAM / đủ disk) chịu được các
    # backend "local" (tải + chạy model AI ngay trên máy) và MUỐN ưu tiên
    # chạy local thay vì tự động đổi sang API — tắt HẲN override bên dưới.
    # Mặc định (không có mục [ci] hoặc để true) vẫn giữ hành vi AN TOÀN cũ.
    # Rủi ro khi tắt: nếu ước tính sai, job có thể bị runner huỷ giữa chừng
    # vì hết disk/RAM — nhờ checkpoint gần real-time, lần chạy sau sẽ tự
    # tiếp tục đúng chỗ dừng, không mất toàn bộ tiến độ.
    ci_section = data.get("ci", {}) if isinstance(data.get("ci"), dict) else {}
    if not ci_section.get("force_lightweight_backends", True):
        print("[ci-config] 'ci.force_lightweight_backends = false' — GIỮ NGUYÊN "
              "backend 'local' trên GitHub Actions theo lựa chọn của người dùng "
              "(không tự đổi sang API). Nếu job bị huỷ giữa chừng vì hết disk/"
              "RAM, chạy lại workflow (tự resume từ checkpoint) hoặc bật lại "
              "true để quay về chế độ an toàn.")
        return data

    def _get(dotted: str) -> Any:
        node: Any = data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def _set(dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    for dotted_key, mapping in _CI_LIGHTWEIGHT_OVERRIDES:
        current = _get(dotted_key)
        if current not in mapping:
            continue
        replacement, required_key = mapping[current]
        if required_key is not None and not _get(required_key):
            print(
                f"[ci-config] CẢNH BÁO: '{dotted_key} = \"{current}\"' rất nặng "
                f"cho GitHub Actions runner (dễ hết disk/RAM -> job bị huỷ giữa "
                f"chừng), nhưng thiếu '{required_key}' nên KHÔNG thể tự chuyển "
                f"sang '{replacement}'. Thêm secret tương ứng nếu muốn CI chạy "
                f"nhẹ và ổn định hơn."
            )
            continue
        _set(dotted_key, replacement)
        print(f"[ci-config] Phát hiện chạy trong GitHub Actions: tự đổi "
              f"'{dotted_key}' từ \"{current}\" sang \"{replacement}\" để tránh "
              f"hết disk/RAM khi tải model local (không ảnh hưởng máy thật/Colab).")

    return data


def load_config(path: str | os.PathLike = "config.toml") -> Config:
    """
    Load và parse config.toml. Ném lỗi rõ ràng nếu file không tồn tại
    hoặc thiếu section bắt buộc.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {config_path}\n"
            f"Hãy copy config.toml.example thành config.toml rồi điền key (xem README mục Cài đặt)."
        )

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    required_sections = ["api", "tts", "processing", "paths"]
    missing = [s for s in required_sections if s not in data]
    if missing:
        raise ValueError(
            f"config.toml thiếu (các) section bắt buộc: {missing}. "
            f"Cần có đủ [api], [tts], [processing], [paths]."
        )

    data = _apply_ci_safe_overrides(data)

    return Config(data, config_path)


# Instance dùng chung, lazy-load lần đầu gọi get_config().
_default_config: Config | None = None


def get_config(path: str | os.PathLike = "config.toml") -> Config:
    """Trả về Config đã cache (singleton nhẹ) để tránh đọc file lặp lại."""
    global _default_config
    if _default_config is None:
        _default_config = load_config(path)
    return _default_config


def reset_config_cache() -> None:
    """Dùng trong test/Colab khi cần load lại config sau khi config.toml bị sửa thủ công."""
    global _default_config
    _default_config = None
