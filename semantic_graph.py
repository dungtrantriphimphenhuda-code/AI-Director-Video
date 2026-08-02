"""
semantic_graph.py — hợp nhất ASR timeline + vision analysis thành
Video Semantic Graph (semantic_blocks.json), theo schema mô tả trong
ref-semantic-graph.md.

Đây là input chính cho script_writer (LLM viết kịch bản) — LLM không nhận
video thô, chỉ nhận block ngữ nghĩa đã có: khung thời gian, tóm tắt hình ảnh,
thoại, nhân vật, cảm xúc, v.v.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def build_semantic_blocks(
    scenes: list[dict[str, Any]],
    asr_timeline: list[dict[str, Any]],
    vision_analysis: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Với mỗi scene: gắn các đoạn ASR chồng lấp thời gian + kết quả vision tương ứng.
    Trả về danh sách block ngữ nghĩa dùng cho story director (LLM).
    """
    vision_by_scene = {v["scene_id"]: v for v in vision_analysis}

    blocks = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        start, end = scene["start"], scene["end"]

        dialogues = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in asr_timeline
            if _overlaps(start, end, seg["start"], seg["end"])
        ]

        vision = vision_by_scene.get(scene_id, {})

        block = {
            "scene_id": scene_id,
            "start": start,
            "end": end,
            "dialogues": dialogues,
            "visual_summary": vision.get("visual_summary", ""),
            "characters": vision.get("characters", []),
            "location": vision.get("location", ""),
            "actions": vision.get("actions", []),
            "emotion": vision.get("emotion", ""),
            "shot_type": vision.get("shot_type", ""),
            "visual_intensity": vision.get("visual_intensity", 0.0),
            "tags": vision.get("tags", []),
            "dialogue_density": len(dialogues),
            # draft_narration: CHỈ tồn tại (không rỗng) khi vision_backend =
            # "gemma4_video" — Gemma 4 xem trực tiếp video + phụ đề nên viết
            # được 1 bản nháp lời bình cấp scene ngay lúc phân tích. Các
            # backend khác không có field này trong vision_analysis.json ->
            # .get(..., "") trả về rỗng, KHÔNG đổi hành vi cũ. script_writer.py
            # dùng field này làm nguyên liệu tham khảo, không phải lời bình
            # cuối cùng (script_writer vẫn chịu trách nhiệm mạch lạc toàn phim).
            "draft_narration": vision.get("draft_narration", ""),
        }
        blocks.append(block)

    return blocks


def run_semantic_graph(
    cfg,
    preprocess_result: dict[str, Any],
    asr_timeline: list[dict[str, Any]],
    vision_analysis: list[dict[str, Any]],
    checkpoint_mgr=None,
) -> list[dict[str, Any]]:
    """Entry point cho việc hợp nhất — không phải stage checkpoint riêng,
    thường được gọi ngay sau vision, nhưng vẫn hỗ trợ checkpoint riêng nếu cần."""
    output_dir = cfg.resolve_path("paths.output_dir")
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    blocks = build_semantic_blocks(preprocess_result["scenes"], asr_timeline, vision_analysis)

    with open(pipeline_dir / "semantic_blocks.json", "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    print(f"[semantic_graph] Xong: {len(blocks)} block ngữ nghĩa.")

    if checkpoint_mgr is not None:
        checkpoint_mgr.save("semantic_graph", blocks)

    return blocks
