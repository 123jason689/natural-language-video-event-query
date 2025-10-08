from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional

import cv2
import torch


@dataclass
class FrameBatch:
    """Container for a batch of frames sampled from a video."""

    frames: torch.Tensor
    timestamps_ms: torch.Tensor
    frame_indices: torch.Tensor
    fps: float
    height: int
    width: int
    device: torch.device
    file_path: str
    batch_id: int

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "FrameBatch":
        """Return a new FrameBatch on the requested device/dtype."""
        tensor = self.frames.to(device=device, dtype=dtype or self.frames.dtype, non_blocking=True)
        return FrameBatch(
            frames=tensor,
            timestamps_ms=self.timestamps_ms.to(device=device, non_blocking=True),
            frame_indices=self.frame_indices.to(device=device, non_blocking=True),
            fps=self.fps,
            height=self.height,
            width=self.width,
            device=tensor.device,
            file_path=self.file_path,
            batch_id=self.batch_id,
        )


class VidTensor(Iterator[FrameBatch]):
    """Iterable wrapper around cv2.VideoCapture that streams frames in batches."""

    def __init__(
        self,
        vid_path: str,
        load_device: torch.device,
        *,
        batch_size: int = 16,
        target_fps: Optional[float] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        vid = cv2.VideoCapture(vid_path)
        if not vid.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {vid_path}")

        self.video_capture = vid
        self._file_path = vid_path
        self.device = load_device

        self._native_fps = float(vid.get(cv2.CAP_PROP_FPS) or 0.0)
        self._total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

        self.batch_size = int(batch_size)
        self._batch_dtype = torch.float32 if load_device.type == "cpu" else torch.float16

        self.fps = self._native_fps if (target_fps is None or target_fps <= 0 or self._native_fps <= 0) else min(
            float(target_fps), self._native_fps
        )
        self.total_frame = self._estimate_total_frames()
        self.height = self._height
        self.width = self._width

        self._interval_ms = 1000.0 / self.fps if self.fps > 0 else None
        self._next_keep_ms = 0.0
        self._frame_idx = 0
        self._kept_frames = 0
        self._batch_counter = 0
        self._closed = False
        self._last_batch: Optional[FrameBatch] = None

    def __iter__(self) -> "VidTensor":
        self.reset()
        return self

    def __next__(self) -> FrameBatch:
        if self._closed:
            raise StopIteration

        frames: List[torch.Tensor] = []
        timestamps: List[float] = []
        kept_indices: List[int] = []

        while len(frames) < self.batch_size:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            timestamp_ms = float(self.video_capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            keep = self._should_keep(timestamp_ms)
            if keep:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame_rgb))
                timestamps.append(timestamp_ms)
                kept_indices.append(self._frame_idx)
                self._kept_frames += 1

            self._frame_idx += 1

        if not frames:
            self.close()
            raise StopIteration

        thwc = torch.stack(frames, dim=0)
        batch = self._build_frame_batch(thwc, timestamps, kept_indices)
        self._last_batch = batch
        self._batch_counter += 1
        return batch

    def __len__(self) -> int:
        expected = self._estimate_total_frames()
        return math.ceil(expected / float(self.batch_size)) if expected > 0 else 0

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def last_batch(self) -> Optional[FrameBatch]:
        return self._last_batch

    def reset(self) -> None:
        if self.video_capture is None:
            vid = cv2.VideoCapture(self._file_path)
            if not vid.isOpened():
                raise FileNotFoundError(f"Unable to reopen video file: {self._file_path}")
            self.video_capture = vid
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._next_keep_ms = 0.0
        self._frame_idx = 0
        self._kept_frames = 0
        self._batch_counter = 0
        self._closed = False
        self._last_batch = None

    def close(self) -> None:
        if not self._closed:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self._closed = True

    def _estimate_total_frames(self) -> int:
        if self._native_fps <= 0 or self.fps <= 0:
            return self._total_frame
        ratio = min(1.0, self.fps / self._native_fps)
        return max(1, int(math.ceil(self._total_frame * ratio))) if self._total_frame > 0 else 0

    def _should_keep(self, timestamp_ms: float) -> bool:
        if self._interval_ms is None:
            return True
        if timestamp_ms + 1e-6 < self._next_keep_ms:
            return False
        self._next_keep_ms += self._interval_ms
        return True

    def _build_frame_batch(
        self,
        thwc: torch.Tensor,
        timestamps: List[float],
        kept_indices: List[int],
    ) -> FrameBatch:
        assert thwc.ndim == 4, "Expected stacked frames with shape (T, H, W, C)"
        tensor = self.permute_THWC_to_TCHW(thwc).contiguous()
        tensor = tensor.to(self.device, dtype=self._batch_dtype, non_blocking=self.device.type != "cpu")
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float64)
        indices_tensor = torch.tensor(kept_indices, dtype=torch.long)
        return FrameBatch(
            frames=tensor,
            timestamps_ms=timestamps_tensor,
            frame_indices=indices_tensor,
            fps=self.fps if self.fps > 0 else self._native_fps,
            height=self.height,
            width=self.width,
            device=tensor.device,
            file_path=self._file_path,
            batch_id=self._batch_counter,
        )

    @staticmethod
    def permute_THWC_to_TCHW(img_tensor: torch.Tensor) -> torch.Tensor:
        if img_tensor.shape[1] != 3:
            return img_tensor.permute(0, 3, 1, 2)
        return img_tensor

    def __del__(self) -> None:  # pragma: no cover - defensive
        self.close()