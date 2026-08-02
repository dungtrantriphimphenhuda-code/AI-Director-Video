#!/usr/bin/env python3
"""
render_ci_config.py — Dựng config.toml (dùng cho CI) từ config.ci.toml.template
bằng cách thay các placeholder dạng ${TEN_BIEN} bằng giá trị lấy từ biến môi
trường (workflow GitHub Actions bơm các biến này từ Secrets trước khi gọi
script).

Chạy: python .github/scripts/render_ci_config.py
Input : config.ci.toml.template (đã có sẵn trong repo, an toàn để commit)
Output: config.toml (KHÔNG commit — chỉ tồn tại trong máy ảo của lần chạy đó)

Khác với dùng thẳng `envsubst`: script này kiểm tra và báo lỗi rõ ràng nếu
thiếu secret BẮT BUỘC cho cloud sync, thay vì âm thầm ghi chuỗi rỗng vào
config.toml rồi để lỗi hiện ra mù mờ ở tận bước cloud storage lúc pipeline
đã chạy được một lúc.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

TEMPLATE = Path("config.ci.toml.template")
OUTPUT = Path("config.toml")

# Áp dụng khi secret không được set (để trống) — Tigris mặc định dùng các
# giá trị này, nên không bắt buộc người dùng phải tạo secret cho chúng.
DEFAULTS = {
    "CLOUD_REGION_NAME": "auto",
    "CLOUD_ADDRESSING_STYLE": "virtual",
}

# Secret được phép để trống — tính năng liên quan chỉ đơn giản bị tắt/bỏ qua.
OPTIONAL = {"HF_TOKEN"}

# Thiếu bất kỳ secret nào trong nhóm này -> dừng ngay, vì cloud sync là kênh
# DUY NHẤT giúp project "sống sót" qua các lần chạy (runner bị xoá sạch sau
# mỗi lần chạy xong) — chạy tiếp mà không có cloud coi như vô nghĩa.
REQUIRED_FOR_CLOUD = {
    "CLOUD_ACCESS_KEY",
    "CLOUD_SECRET_KEY",
    "CLOUD_BUCKET_NAME",
    "CLOUD_ENDPOINT_URL",
}

# Danh sách biến HỢP LỆ duy nhất được phép thay — cố định, KHÔNG dò tự động
# bằng regex trên toàn bộ file. Lý do: config.ci.toml.template có một dòng
# comment ở đầu file minh hoạ cú pháp bằng "${TEN_BIEN}" — nếu quét mù mọi
# "${...}" trong file (kể cả trong comment), chuỗi ví dụ đó sẽ bị hiểu nhầm
# thành một biến thật cần thay, gây cảnh báo/lỗi sai. Chỉ thay đúng các biến
# trong danh sách này, giữ nguyên mọi "${...}" khác (kể cả trong comment).
KNOWN_VARS = {
    "CEREBRAS_API_KEY",
    "HF_TOKEN",
    "MISTRAL_API_KEY",
    "CLOUD_ACCESS_KEY",
    "CLOUD_SECRET_KEY",
    "CLOUD_BUCKET_NAME",
    "CLOUD_ENDPOINT_URL",
    "CLOUD_REGION_NAME",
    "CLOUD_ADDRESSING_STYLE",
}

PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def main() -> int:
    if not TEMPLATE.exists():
        print(f"[ci-config] LỖI: không tìm thấy {TEMPLATE} trong repo.", file=sys.stderr)
        return 1

    text = TEMPLATE.read_text(encoding="utf-8")
    found = set(PLACEHOLDER_RE.findall(text))
    # Chỉ xử lý các placeholder nằm trong KNOWN_VARS; bất kỳ "${...}" nào
    # khác (vd. trong comment minh hoạ) bị bỏ qua, không thay, không báo lỗi.
    names = sorted(found & KNOWN_VARS)
    unknown = found - KNOWN_VARS
    if unknown:
        print(f"[ci-config] Bỏ qua placeholder không thuộc danh sách biết trước "
              f"(có thể chỉ là ví dụ trong comment): {sorted(unknown)}")

    values: dict[str, str] = {}
    missing_required = []
    for name in names:
        val = os.environ.get(name, "") or DEFAULTS.get(name, "")
        if not val and name in REQUIRED_FOR_CLOUD:
            missing_required.append(name)
        values[name] = val

    if missing_required:
        print(
            "[ci-config] LỖI: thiếu GitHub Secrets bắt buộc cho cloud sync: "
            + ", ".join(missing_required)
            + ".\n[ci-config] Vào repo -> Settings -> Secrets and variables -> "
              "Actions -> New repository secret để thêm.\n"
              "[ci-config] Xem hướng dẫn trong README, mục "
              "'Chạy tự động bằng GitHub Actions'.",
            file=sys.stderr,
        )
        return 1

    for name in names:
        if not values[name] and name not in OPTIONAL:
            print(f"[ci-config] Lưu ý: secret '{name}' đang để trống — "
                  f"tính năng liên quan (nếu có dùng) sẽ bị bỏ qua/lỗi khi chạy tới.")

    def repl(m: re.Match) -> str:
        var = m.group(1)
        return values[var] if var in values else m.group(0)  # giữ nguyên nếu không thuộc KNOWN_VARS

    rendered = PLACEHOLDER_RE.sub(repl, text)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"[ci-config] Đã tạo {OUTPUT} từ {TEMPLATE} ({len(names)} biến đã thay).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
