"""
viterbox_env.py — Tạo & quản lý venv RIÊNG cho backend TTS "viterbox".

Xem giải thích đầy đủ trong viterbox_worker.py: package `viterbox` ghim
version numpy/transformers xung đột trực tiếp với requirements.txt chính của
project, nên phải cài trong 1 virtualenv tách biệt, không đụng tới
site-packages của tiến trình Python đang chạy pipeline. Module này chỉ được
gọi khi tts.engine == "viterbox" (xem tts.py) — không ảnh hưởng gì tới các
backend TTS khác hay tới môi trường Python chính.

Hoạt động giống hệt trên máy thật / Colab / VPS Linux / GitHub Actions: chỉ
cần `python3 -m venv` chạy được (luôn có sẵn trên mọi bản Python 3 chuẩn).

LƯU Ý CHO GITHUB ACTIONS: venv này KHÔNG có GPU trên runner miễn phí (không
có CUDA) -> tự động rơi về "cpu", chậm hơn nhiều so với chạy trên máy/Colab
có GPU. Việc cài đặt (tải torch + package viterbox + model AI ~3-4GB) cũng sẽ
phải làm LẠI TỪ ĐẦU mỗi lần chạy CI vì runner bị xoá sau mỗi job -- không
phải lỗi của cơ chế cache trong module này, mà là giới hạn vốn có của runner
dùng 1 lần.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_VENV_DIRNAME = ".viterbox_venv"
_VITERBOX_GIT_URL = "git+https://github.com/iamdinhthuan/viterbox-tts.git"
_MARKER_NAME = ".install_ok"

# Package `viterbox` (upstream: iamdinhthuan/viterbox-tts) dùng các module này
# ở runtime (qua s3tokenizer / s3gen / conformer...) nhưng KHÔNG khai báo
# chúng trong pyproject.toml của chính nó -> pip không tự cài, gây lỗi
# "No module named 'omegaconf'" (và có thể thiếu thêm module tương tự tuỳ
# version). Ta cài bù các dependency còn thiếu này ngay sau khi cài viterbox,
# bất kể chạy trên máy nào (không riêng gì GitHub Actions).
_KNOWN_MISSING_DEPS = [
    "omegaconf",
]

# Map "tên module khi import" -> "tên package khi pip install", cho các
# trường hợp 2 tên khác nhau (phòng khi auto-heal ở dưới gặp phải).
_MODULE_TO_PIP_NAME = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
}


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_bin_dir(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def _pip_install(py: Path, *packages: str) -> None:
    subprocess.run([str(py), "-m", "pip", "install", *packages], check=True)


def _pip_install_no_isolation(py: Path, venv_dir: Path, *packages: str) -> None:
    env = os.environ.copy()
    bin_dir = str(_venv_bin_dir(venv_dir))
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    subprocess.run(
        [str(py), "-m", "pip", "install", "--no-build-isolation", *packages],
        check=True, env=env,
    )


def _self_test_import(py: Path) -> tuple[bool, str]:
    """Thử `import viterbox` trong venv, trả về (thành_công, tên_module_thiếu_nếu_có)."""
    result = subprocess.run(
        [str(py), "-c", "import viterbox"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return True, ""
    match = re.search(r"No module named '([^']+)'", result.stderr)
    missing = match.group(1).split(".")[0] if match else ""
    return False, missing


def ensure_viterbox_env(project_root: Path) -> Path:
    """
    Tạo venv riêng (nếu chưa có) + cài package `viterbox`, trả về đường dẫn
    python interpreter của venv đó.

    Idempotent: nếu đã cài xong lần trước (marker file tồn tại), bỏ qua luôn
    không cài lại -- tránh tải lại torch/model mỗi lần chạy pipeline trên
    cùng 1 máy (quan trọng khi chạy lặp lại trên máy thật/self-hosted runner;
    trên GitHub Actions hosted runner thì venv bị xoá cùng máy ảo sau mỗi
    job nên vẫn phải cài lại mỗi lần dù có marker hay không).

    Sau khi cài xong, LUÔN chạy self-test `import viterbox` trong venv. Nếu
    thiếu module (như 'omegaconf' — bug ở gói viterbox gốc, không khai báo
    dependency đầy đủ), tự động pip install bù và thử lại, tối đa vài lần.
    Nhờ vậy hàm này tự "chữa lành" trên bất kỳ thiết bị nào (máy thật, Colab,
    VPS, GitHub Actions...), không cần sửa tay mỗi lần môi trường khác nhau.
    """
    venv_dir = project_root / _VENV_DIRNAME
    marker = venv_dir / _MARKER_NAME
    py = _venv_python(venv_dir)

    if marker.exists() and py.exists():
        ok, missing = _self_test_import(py)
        if ok:
            print(f"[viterbox-env] venv riêng đã sẵn sàng tại {venv_dir}")
            return py
        print(f"[viterbox-env] venv cũ bị thiếu module '{missing}', cài bù rồi kiểm tra lại...")
        marker.unlink(missing_ok=True)

    if not py.exists():
        print(f"[viterbox-env] Tạo venv riêng cho Viterbox tại {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    print("[viterbox-env] Cài package 'viterbox' (+ dependencies riêng: "
          "torch/transformers/numpy phiên bản do viterbox tự ghim, KHÔNG liên "
          "quan tới version trong requirements.txt chính) — lần đầu có thể mất "
          "vài phút vì phải tải torch + model AI (~3-4GB)...")

    # LƯU Ý: viterbox ghim numpy<1.26.0 (vd 1.25.2), bản này KHÔNG có wheel
    # dựng sẵn cho Python 3.12+ nên pip phải build từ source. Cần tránh 2 lỗi:
    #   (1) pip >= 26.x không tự động cấp 'setuptools' cho build isolation
    #       của các gói sdist -> "Cannot import 'setuptools.build_meta'".
    #   (2) setuptools >= 83.x dùng pkgutil.ImpImporter (đã xoá ở Python 3.12)
    #       -> AttributeError trong build isolation.
    # Giải pháp: ghim pip về bản ổn định, pin setuptools bản cũ tương thích,
    # và dùng --no-build-isolation để build bằng setuptools trong venv thay vì
    # để pip tự kéo bản mới nhất (dễ vỡ) vào môi trường build tạm.
    #
    # QUAN TRỌNG: Pre-install numpy <1.26 (build source với setuptools cũ) và
    # pandas >=2.1.1 (wheel cho cp312) NGAY TỪ ĐẦU, trước khi install viterbox.
    # Lý do: pandas 2.1.0 tồn tại dạng source dist dùng meson build backend;
    # khi pip resolve dependency viterbox với --no-build-isolation, nó chạy
    # metadata preparation của pandas-2.1.0.tar.gz trong môi trường hiện tại
    # (không isolation) và gây lỗi "meson executable not found" dù đã cài meson.
    # Pre-install pandas >=2.1.1 (có wheel sẵn, không cần build) giúp pip
    # không bao giờ đụng tới pandas-2.1.0.tar.gz. Tương tự, pre-install numpy
    # tránh việc pip phải build numpy từ source trong lúc install viterbox.
    _pip_install(py, "pip==24.3.1")
    _pip_install(py, "setuptools==68.2.2", "wheel")
    # QUAN TRỌNG (fix lỗi ResolutionImpossible trên Python 3.12+):
    # Không được yêu cầu numpy<1.26 và pandas>=2.1.1 trong CÙNG 1 lệnh pip,
    # vì MỌI bản pandas>=2.1.1 đều khai báo phụ thuộc numpy>=1.26.0 khi
    # python_version>=3.12 -> pip resolver luôn báo xung đột không thể giải
    # (xem traceback CI thực tế). Mục đích ban đầu của việc pre-install pandas
    # ở đây chỉ là để tránh pip sau này (lúc cài viterbox) phải build
    # pandas-2.1.0.tar.gz (sdist dùng meson). Ta vẫn đạt mục đích đó bằng cách
    # cài pandas RIÊNG với --no-deps, để pip không kiểm tra/khớp lại yêu cầu
    # numpy của pandas (numpy<1.26 đã cài ở venv vẫn hoạt động bình thường lúc
    # runtime dù metadata của pandas "muốn" bản mới hơn).
    _pip_install_no_isolation(py, venv_dir,
        "numpy<1.26", "meson-python", "meson", "ninja")
    _pip_install(py, "pandas>=2.1.1", "--no-deps")
    _pip_install_no_isolation(py, venv_dir, _VITERBOX_GIT_URL)

    # Cài bù các dependency mà gói viterbox gốc quên khai báo (xem
    # _KNOWN_MISSING_DEPS ở trên) — làm ngay để đỡ phải chờ self-test loop.
    print(f"[viterbox-env] Cài bù dependency còn thiếu ở gói viterbox gốc: {_KNOWN_MISSING_DEPS}...")
    _pip_install(py, *_KNOWN_MISSING_DEPS)

    # Self-test + auto-heal: nếu vẫn còn thiếu module khác (chưa biết trước),
    # tự phát hiện qua thông báo lỗi và cài bù, lặp lại tối đa 5 lần.
    for attempt in range(1, 6):
        ok, missing = _self_test_import(py)
        if ok:
            break
        if not missing:
            raise RuntimeError(
                "[viterbox-env] `import viterbox` lỗi nhưng không xác định "
                "được tên module bị thiếu để tự cài bù — cần kiểm tra thủ công."
            )
        pip_name = _MODULE_TO_PIP_NAME.get(missing, missing)
        print(f"[viterbox-env] (lần {attempt}/5) Thiếu module '{missing}', "
              f"tự cài bù '{pip_name}'...")
        _pip_install(py, pip_name)
    else:
        ok, missing = _self_test_import(py)
        if not ok:
            raise RuntimeError(
                f"[viterbox-env] Vẫn lỗi `import viterbox` sau nhiều lần tự "
                f"cài bù (module thiếu cuối cùng: '{missing}'). Cần kiểm tra "
                f"thủ công, có thể do lỗi khác ngoài thiếu dependency."
            )

    marker.write_text("ok", encoding="utf-8")
    print("[viterbox-env] Cài xong.")
    return py
