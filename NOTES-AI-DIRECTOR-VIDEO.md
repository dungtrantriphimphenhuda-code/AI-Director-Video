# Ghi chú sửa lỗi & nâng cấp — AI-Director-Video

## Vòng 1 — Lỗi gốc: video 25.3 phút thay vì mục tiêu 3 phút

Đo được từ `tts_report.json` thật (127 câu, engine `viterbox`):
tổng 3761 từ / 498.0s audio thật = **~7.55 từ/giây**, nhưng
`processing.words_per_sec` trong config đang để **2.5** (mặc định repo) —
sai lệch ~3 lần.

Hệ quả kép:
1. `calc_output_duration()` ước tính mỗi câu cần đọc lâu gấp ~3 lần thời
   lượng TTS thật → `build_storyboard()` gán cho mỗi clip 1 "slot" thời gian
   dài gấp 3 lần cần thiết.
2. `render.py` giãn tốc độ clip gốc (setpts/atempo) ra để LẤP ĐẦY slot sai
   đó → clip gốc bị kéo chậm không cần thiết + có khoảng lặng dài sau khi
   TTS đọc xong nhưng slot chưa hết.
3. Không có bước nào kiểm tra TỔNG thời lượng sau khi gộp nhiều batch
   narration (mỗi batch tự viết theo tỉ lệ scene riêng, không ai canh tổng)
   → nội dung tự phình to.

**Fix vòng 1:** sửa `words_per_sec = 7.5` trong config + thêm hàm
`_enforce_target_duration()` cắt cứng tổng về đúng `target_duration_sec`
(giữ câu đầu/cuối, cắt câu match_score thấp nhất trước).

## Vòng 2 — Không khoá cứng độ dài, làm "thông minh" theo từng phim

Người dùng phản hồi: ép cứng về 1 con số target_duration_sec chung cho mọi
video là gò bó — phim nào cũng có lượng nội dung đáng kể để kể khác nhau.

Tham khảo bộ skill `video-recap-skills` (dự án Claude Skills khác của cùng
tác giả) áp dụng cho trường hợp tương tự, thấy 2 ý tưởng đáng lấy:

### 1. `speech_safety_margin` — hệ số dự phòng khi ước tính

`video-recap-skills` đo `speech_rate` thực tế của giọng TTS (giống
`words_per_sec` ở đây) NHƯNG luôn nhân thêm `speech_safety_margin` (mặc
định 0.85) trước khi dùng để ước tính thời lượng — vì TTS thực tế dao động
±10-20% quanh giá trị đo trung vị tuỳ câu (ngữ điệu, dấu câu, độ dài). Nếu
dùng đúng số đo trung vị (như fix vòng 1), khoảng một nửa số câu sẽ đọc
CHẬM hơn ước tính → lại tràn slot, chỉ là lỗi nhỏ hơn thay vì biến mất hẳn.

→ Đã thêm `processing.speech_safety_margin = 0.85` và tham số
`safety_margin` vào `calc_output_duration()`.

### 2. "Coverage is diagnostic, not a creative quota"

Đây là nguyên tắc lõi của `video-recap-skills`: không có bước nào ép TỔNG
thời lượng narration về đúng 1 con số cố định. Thay vào đó:
- Model viết narration theo "beat map" nội dung — timing chỉ là gợi ý mềm
  để tránh viết vụn từng câu, prompt luôn ghi rõ *"never pad to hit this
  number"*.
- Sau khi viết xong, một bước lint riêng (`narration_lint.py`) đo
  **coverage** (bao nhiêu % thời lượng có lời bình) và chỉ **cảnh báo** nếu
  quá thưa (<0.5, "under_narrated") hay quá dày đặc (>0.85, "không có chỗ
  cho tiếng gốc thở") — không tự động sửa gì, để Agent/người dùng quyết định.

→ Áp dụng tương tự vào AI-Director-Video:
- **`processing.target_duration_mode = "auto"`** (mặc định mới): không còn
  bước cắt cứng nào sau khi gộp batch. `target_duration_sec` chỉ còn dùng để
  chia ngân sách gợi ý cho từng batch khi viết (không đổi). Sau khi gộp,
  `_log_duration_estimate()` chỉ IN ra tổng ước tính so với gợi ý, không sửa
  gì. Đặt `target_duration_mode = "fixed"` nếu vẫn muốn hành vi ép cứng cũ
  (vd ràng buộc thời lượng nền tảng đăng video).
- **`pacing_report.json`** (mới, sinh ra ở cuối stage `script`): chẩn đoán
  TỪNG CLIP riêng lẻ thay vì chỉ nhìn tổng — vì 1 video có thể có tổng thời
  lượng hợp lý nhưng vẫn có vài clip cục bộ bị "gấp" (cảnh gốc quá dài so
  với câu lời bình ngắn → tua nhanh) hoặc "lê thê" (cảnh gốc quá ngắn so với
  câu lời bình dài → kéo chậm) mà ép tổng không phát hiện được. Dùng đúng
  công thức `speed = src_span / out_dur` mà `render.py` áp dụng thật, so với
  `max_speed_ratio` (mặc định 4.0) và `min_speed_ratio` (mới, mặc định 0.5).
  Chỉ báo cáo, không tự sửa nội dung — nhưng `render.py` giờ THỰC SỰ kẹp tốc
  độ trong khoảng này (trước đây `max_speed_ratio` chỉ là 1 dòng log, không
  hề chặn tốc độ tua thật), bù phần thời lượng thiếu bằng đông cứng khung
  hình cuối (`tpad`) khi cần kéo chậm nhưng bị kẹp lại.

### Kết quả mô phỏng trên đúng 127 câu narration thật (little-brother)

| | Tổng thời lượng | Số câu giữ lại |
|---|---|---|
| Trước khi sửa (bug gốc) | 25.3 phút | 127/127 |
| Fix vòng 1 (words_per_sec đúng + cắt cứng về 180s) | 3.45 phút | 51/127 |
| Fix vòng 2 (auto mode + safety margin, không cắt) | 10.0 phút | 127/127 (giữ nguyên nội dung, độ dài do chính nội dung quyết định) |

`pacing_report.json` mô phỏng trên cùng dữ liệu: 17/127 clip sẽ bị chẩn đoán
"rushed" (cảnh gốc dài, lời bình ngắn), 22/127 "dragging" (cảnh gốc ngắn,
lời bình dài) — đây chính là những điểm cụ thể nên xem lại câu văn hoặc lựa
chọn scene, thay vì đoán mò xem "khoảng lặng/gấp gáp" nằm ở đâu.

## Việc chưa làm (ngoài phạm vi lần sửa này)

- `pacing_report.json` chỉ chẩn đoán + `render.py` chỉ kẹp tốc độ ở mức
  ffmpeg — chưa có bước tự động VIẾT LẠI câu narration bị flag "dragging"
  cho ngắn lại hoặc "rushed" cho dài ra (cần gọi lại LLM, ngoài phạm vi vá
  lỗi lần này). Nếu muốn, có thể thêm 1 bước re-generate có chọn lọc chỉ cho
  các clip bị flag, dùng `story_so_far` xung quanh để giữ mạch truyện.
- Ý tưởng "content-led beat map" đầy đủ của `video-recap-skills` (LLM tự
  gán từng đoạn cho narration/tiếng gốc/im lặng dựa trên ai đang "sở hữu"
  khoảnh khắc đó — `audio_owner`) là một kiến trúc khác hẳn (giữ nguyên
  timeline gốc, chỉ chèn lời bình vào chỗ trống) so với AI-Director-Video
  (dựng timeline MỚI từ các đoạn scene được chọn). Portable một phần
  (coverage diagnostic, safety margin) nhưng port toàn bộ kiến trúc
  audio_owner cần đổi cách `build_storyboard()` hoạt động — việc lớn hơn
  nhiều, chưa làm ở đây.
