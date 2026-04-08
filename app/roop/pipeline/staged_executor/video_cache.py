import json
import math
from pathlib import Path

import cv2
import numpy as np

from roop.media.ffmpeg_writer import FFMPEG_VideoWriter
from roop.media.video_io import open_video_capture


def normalize_cache_image(image):
    image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"cache image must be HxWx3, got {image.shape}")
    return image


class VideoStageCache:
    """
    Stores stage cache images in a grid-packed MP4 plus a JSON index.

    Layout:
        <cache>.mp4
        <cache>.idx.bin

    The video contains grid-packed cache crops, while the index records the
    frame number and cell position for each cache key so partial reads can seek
    only the needed frames.
    """

    def __init__(
        self,
        codec="libx264",
        crf=4,
        preset="veryfast",
        max_frame_extent=2048,
        fps=30,
        output_pix_fmt="yuv444p",
    ):
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.max_frame_extent = max(64, int(max_frame_extent))
        self.fps = max(1, int(fps))
        self.output_pix_fmt = output_pix_fmt

    def _resolve_paths(self, cache_path):
        base_path = Path(cache_path)
        if str(base_path).lower().endswith(".idx.bin"):
            index_path = base_path
            video_path = base_path.with_suffix("").with_suffix(".mp4")
        elif base_path.suffix.lower() == ".mp4":
            video_path = base_path
            index_path = base_path.with_suffix(".idx.bin")
        else:
            video_path = base_path.with_suffix(".mp4")
            index_path = base_path.with_suffix(".idx.bin")
        return video_path, index_path

    def _read_index(self, index_path):
        if not index_path.exists():
            return {}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _choose_grid(self, tile_h, tile_w, item_count):
        max_cols = max(1, self.max_frame_extent // max(tile_w, 1))
        max_rows = max(1, self.max_frame_extent // max(tile_h, 1))
        max_cells = max_cols * max_rows
        cells_per_frame = max(1, min(item_count, max_cells))
        cols = max(1, min(max_cols, math.ceil(math.sqrt(cells_per_frame))))
        rows = max(1, min(max_rows, math.ceil(cells_per_frame / cols)))
        frame_w = cols * tile_w
        frame_h = rows * tile_h
        if frame_w % 2 != 0:
            frame_w += 1
        if frame_h % 2 != 0:
            frame_h += 1
        return {
            "cols": cols,
            "rows": rows,
            "cells_per_frame": cols * rows,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    def _build_index(self, cache_map):
        items = [(cache_key, normalize_cache_image(cache_map[cache_key])) for cache_key in sorted(cache_map)]
        if not items:
            return {
                "version": 2,
                "tile_shape": [0, 0],
                "grid_shape": [0, 0],
                "frame_shape": [0, 0],
                "count": 0,
                "items": {},
            }, []

        tile_h = max(image.shape[0] for _, image in items)
        tile_w = max(image.shape[1] for _, image in items)
        grid = self._choose_grid(tile_h, tile_w, len(items))
        index = {
            "version": 2,
            "tile_shape": [tile_h, tile_w],
            "grid_shape": [grid["rows"], grid["cols"]],
            "frame_shape": [grid["frame_h"], grid["frame_w"]],
            "count": len(items),
            "items": {},
        }

        packed_frames = []
        frame_canvas = None
        for item_index, (cache_key, image) in enumerate(items):
            frame_idx = item_index // grid["cells_per_frame"]
            cell_idx = item_index % grid["cells_per_frame"]
            row = cell_idx // grid["cols"]
            col = cell_idx % grid["cols"]
            if cell_idx == 0:
                if frame_canvas is not None:
                    packed_frames.append(frame_canvas)
                frame_canvas = np.zeros((grid["frame_h"], grid["frame_w"], 3), dtype=np.uint8)
            top = row * tile_h
            left = col * tile_w
            h, w = image.shape[:2]
            frame_canvas[top:top + h, left:left + w] = image
            index["items"][cache_key] = {
                "frame_idx": frame_idx,
                "row": row,
                "col": col,
                "shape": [h, w, 3],
            }
        if frame_canvas is not None:
            packed_frames.append(frame_canvas)
        return index, packed_frames

    def _write_video(self, video_path, frames):
        if not frames:
            video_path.unlink(missing_ok=True)
            return
        writer = FFMPEG_VideoWriter(
            str(video_path),
            (frames[0].shape[1], frames[0].shape[0]),
            self.fps,
            codec=self.codec,
            crf=self.crf,
            quality_args=["-crf", str(self.crf)],
            ffmpeg_params=["-preset", self.preset, "-g", "1", "-bf", "0"],
            output_pix_fmt=self.output_pix_fmt,
            video_filter=None,
        )
        try:
            for frame in frames:
                writer.write_frame(frame)
        finally:
            writer.close()

    def _read_video_frame(self, capture, frame_idx):
        if frame_idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"failed to decode cache frame {frame_idx}")
        return frame

    def _extract_image(self, frame, metadata, tile_h, tile_w):
        top = int(metadata["row"]) * tile_h
        left = int(metadata["col"]) * tile_w
        height, width = metadata["shape"][:2]
        image = frame[top:top + height, left:left + width]
        return normalize_cache_image(image)

    def write(self, cache_path, cache_map):
        video_path, index_path = self._resolve_paths(cache_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        index, packed_frames = self._build_index(cache_map)
        if index["count"] == 0:
            index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            video_path.unlink(missing_ok=True)
            return video_path

        self._write_video(video_path, packed_frames)
        index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        return video_path

    def read(self, cache_path):
        video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        items = index.get("items", {})
        if not items:
            return {}
        return self.read_keys(cache_path, list(items))

    def list_keys(self, cache_path):
        _video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        return sorted(index.get("items", {}))

    def count(self, cache_path):
        _video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        count = index.get("count")
        if isinstance(count, int):
            return max(0, count)
        return len(index.get("items", {}))

    def read_keys(self, cache_path, keys):
        video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        items = index.get("items", {})
        if not items or not keys:
            return {}

        tile_h, tile_w = index.get("tile_shape", [0, 0])
        if tile_h <= 0 or tile_w <= 0:
            return {}

        requested = {cache_key: items[cache_key] for cache_key in keys if cache_key in items}
        if not requested:
            return {}

        grouped = {}
        for cache_key, metadata in requested.items():
            grouped.setdefault(int(metadata["frame_idx"]), []).append((cache_key, metadata))

        selected = {}
        capture = open_video_capture(str(video_path))
        try:
            for frame_idx in sorted(grouped):
                frame = self._read_video_frame(capture, frame_idx)
                for cache_key, metadata in grouped[frame_idx]:
                    selected[cache_key] = self._extract_image(frame, metadata, tile_h, tile_w)
        finally:
            capture.release()
        return selected


__all__ = ["VideoStageCache", "normalize_cache_image"]
