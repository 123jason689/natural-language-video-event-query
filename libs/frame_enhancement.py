import kornia as K
import kornia.filters as KF
import kornia.enhance as KE
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torch
import os
import numpy as np
import cv2
import time
from pathlib import Path
from typing import List, Union, Tuple, Optional
from .typings_ import FrameBatch

class AutoEnhance:
    """
    Flexible auto-enhancement:
      - CPU backend: per-frame, OpenCV-heavy (low RAM)
      - GPU backend (CUDA/MPS): Kornia/Torch in chunked batches with AMP

    Output: float32 (CPU) or chosen dtype (GPU), range [0,1], same (T,3,H,W).
    """

    def __init__(
        self,
        ema_alpha: float = 0.6,
        use_gpu: bool = True,
        gpu_dtype: torch.dtype = torch.float16,   # good default for CUDA
        max_gpu_pixels_per_batch: int = 1280*720*4,  # cap VRAM via adaptive chunking
        clahe_tile: int = 8,
        clahe_clip_default: float = 2.0,
        diag_short_side: int = 256,
        unsharp_sigma: float = 1.5,
        enable_median: bool = True,
    ):
        self.ema_alpha = float(ema_alpha)
        self.use_gpu = bool(use_gpu)
        self.gpu_dtype = gpu_dtype
        self.max_gpu_pixels_per_batch = int(max_gpu_pixels_per_batch)
        self.clahe_tile = int(clahe_tile)
        self.clahe_clip_default = float(clahe_clip_default)
        self.diag_short_side = int(diag_short_side)
        self.unsharp_sigma = float(unsharp_sigma)
        self.enable_median = bool(enable_median)

        # CPU resources
        self._clahe_cv = cv2.createCLAHE(clipLimit=self.clahe_clip_default,
                                         tileGridSize=(self.clahe_tile, self.clahe_tile))
        self._gamma_val = None
        self._gamma_lut = None

        # EMA state (scalars / small vectors)
        self._ema_Ymean = None
        self._ema_Ystd  = None
        self._ema_sharp = None
        self._ema_noise = None
        self._ema_wbgain = None  # np.float32[3] (B,G,R)

    def __call__(self, batch) -> "FrameBatch":
        x = batch.frames  # (T,3,H,W)
        device = x.device
        want_gpu = (device.type != "cpu") and self.use_gpu

        if want_gpu:
            batch.frames = self._enhance_gpu(batch.frames)
        else:
            batch.frames = self._enhance_cpu(batch.frames)

        print("Frames Enhanced")
        return batch
    
    @torch.inference_mode()
    def _downsample_for_diag(self, bgr_u8: np.ndarray, short_side: int = 256) -> np.ndarray:
        h, w = bgr_u8.shape[:2]
        scale = short_side / max(h, w)
        if scale < 1.0:
            return cv2.resize(bgr_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return bgr_u8

    def _gamma_lut_u8(self, gamma: float) -> np.ndarray:
        inv = max(gamma, 1e-3)
        return (np.power(np.arange(256, dtype=np.float32)/255.0, 1.0/inv)*255.0).clip(0,255).astype(np.uint8)

    # ---------------- CPU BACKEND (OpenCV per-frame) ---------------- #
    @torch.inference_mode()
    def _enhance_cpu(self, video_tchw: torch.Tensor) -> torch.Tensor:
        assert video_tchw.ndim == 4 and video_tchw.shape[1] == 3
        T, C, H, W = video_tchw.shape
        device = video_tchw.device

        out = torch.empty((T, C, H, W), dtype=torch.float32, device=device)

        def ema(prev, cur):
            if prev is None: return cur
            return self.ema_alpha * cur + (1.0 - self.ema_alpha) * prev

        for i in range(T):
            f = video_tchw[i]
            if f.dtype != torch.uint8:
                if f.is_floating_point():
                    f = (f.clamp(0,1) * 255.0).to(torch.uint8)
                else:
                    f = f.clamp(0,255).to(torch.uint8)

            rgb = f.permute(1,2,0).contiguous().cpu().numpy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Diagnostics
            ds = self._downsample_for_diag(bgr, self.diag_short_side)
            gray = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=3).var()
            blur = cv2.GaussianBlur(ds, (3,3), 1.0)
            noise = cv2.absdiff(ds, blur).mean()

            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[...,0].astype(np.float32)/255.0
            Ymean = float(Y.mean())
            Ystd  = float(Y.std() + 1e-8)

            self._ema_Ymean = ema(self._ema_Ymean, Ymean)
            self._ema_Ystd  = ema(self._ema_Ystd,  Ystd)
            self._ema_sharp = ema(self._ema_sharp, sharp)
            self._ema_noise = ema(self._ema_noise, noise)

            means = bgr.reshape(-1,3).mean(axis=0) + 1e-6  # B,G,R
            g = means[1]
            wbgain = np.array([g/means[0], 1.0, g/means[2]], dtype=np.float32)
            self._ema_wbgain = wbgain if self._ema_wbgain is None else (
                self.ema_alpha*wbgain + (1.0-self.ema_alpha)*self._ema_wbgain
            )

            # Params
            if self._ema_Ymean < 0.22: gamma = 0.70
            elif self._ema_Ymean < 0.35: gamma = 0.80
            elif self._ema_Ymean < 0.55: gamma = 0.90
            else: gamma = 1.00

            if self._ema_Ystd < 0.05: clahe_clip = 3.5
            elif self._ema_Ystd < 0.08: clahe_clip = 2.5
            elif self._ema_Ystd < 0.12: clahe_clip = 1.8
            else: clahe_clip = 1.2
            self._clahe_cv.setClipLimit(clahe_clip)

            if self.enable_median:
                if self._ema_noise > 0.06*255: med = 5
                elif self._ema_noise > 0.035*255: med = 3
                else: med = 0
                if med in (3,5): bgr = cv2.medianBlur(bgr, med)

            # WB
            bgr = np.clip(bgr.astype(np.float32) * self._ema_wbgain[None,None,:], 0, 255).astype(np.uint8)

            # Gamma LUT
            if self._gamma_val != gamma:
                self._gamma_lut = self._gamma_lut_u8(gamma)
                self._gamma_val = gamma
            bgr = cv2.LUT(bgr, self._gamma_lut)

            # CLAHE on Y
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[...,0]
            y[:] = self._clahe_cv.apply(y)
            bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            # Unsharp
            blurred = cv2.GaussianBlur(bgr, (0,0), sigmaX=self.unsharp_sigma, sigmaY=self.unsharp_sigma)
            # usm amount from sharp/noise
            base = 0.9 if self._ema_sharp < 0.002 else (0.7 if self._ema_sharp < 0.006 else 0.5)
            noise_pen = min(max(self._ema_noise/(0.08*255), 0.0), 1.0) * 0.4
            usm_amount = float(np.clip(base - noise_pen, 0.3, 1.0))
            bgr = cv2.addWeighted(bgr, 1.0 + usm_amount, blurred, -usm_amount, 0)

            # back to torch float32 [0,1]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_f = torch.from_numpy(rgb).to(dtype=torch.float32, device=device).permute(2,0,1) / 255.0
            out[i].copy_(rgb_f)

        return out

    # ---------------- GPU BACKEND (Torch/Kornia, chunked) ---------------- #
    @torch.inference_mode()
    def _enhance_gpu(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Chunked GPU pipeline to limit VRAM:
          - Keeps uint8 until normalization stage if input is uint8
          - AMP to reduce VRAM/boost throughput
          - CLAHE via Kornia per-frame but batched
        """
        assert video_tchw.ndim == 4 and video_tchw.shape[1] == 3
        device = video_tchw.device
        T, C, H, W = video_tchw.shape

        # Decide chunk size from pixel budget
        pixels_per_frame = H * W
        max_frames = max(1, self.max_gpu_pixels_per_batch // max(1, pixels_per_frame))
        Bchunk = min(max_frames, 32)  # soft cap

        # Output dtype (float16/float32) on GPU in [0,1]
        out_dtype = self.gpu_dtype if device.type != "cpu" else torch.float32
        out = torch.empty((T, C, H, W), dtype=out_dtype, device=device)

        # Tiny EMA states (torch tensors to keep on device)
        ema_Ymean = None
        ema_Ystd  = None
        ema_sharp = None
        ema_noise = None
        ema_wbgain = None  # (3,)

        def ema(prev, cur, alpha: float):
            if prev is None: return cur
            return alpha * cur + (1.0 - alpha) * prev

        # Autocast for mixed precision on CUDA/MPS
        use_amp = (device.type == "cuda")

        for s in range(0, T, Bchunk):
            e = min(s + Bchunk, T)
            xb = video_tchw[s:e]  # (B,3,H,W) possibly uint8

            # ---- keep uint8 until needed ----
            if xb.dtype == torch.uint8:
                x_float = xb.to(out_dtype) / 255.0
            else:
                x_float = xb.to(out_dtype)
                if float(x_float.max()) > 1.5:
                    x_float = (x_float / 255.0)

            with torch.autocast(device_type='cuda', dtype=self.gpu_dtype, enabled=use_amp):
                # Diagnostics on downsampled frames to save VRAM
                # Use bilinear downscale to diag_short_side
                short = self.diag_short_side
                scale = short / max(H, W)
                if scale < 1.0:
                    h2, w2 = int(H*scale), int(W*scale)
                    x_small = F.interpolate(x_float, size=(h2, w2), mode="bilinear", align_corners=False)
                else:
                    x_small = x_float

                ycb_small = K.color.rgb_to_ycbcr(x_small)
                Y_small = ycb_small[:, :1]
                Ymean = Y_small.mean(dim=(2,3))  # (B,1)
                Ystd  = Y_small.std(dim=(2,3)) + 1e-8

                # sharpness via Laplacian var
                lap = KF.laplacian(Y_small, kernel_size=3)
                sharp = (lap**2).mean(dim=(2,3))  # (B,1)

                # noise proxy
                blur = KF.gaussian_blur2d(x_small, (3,3), (1.0,1.0))
                noise = (x_small - blur).abs().mean(dim=(1,2,3), keepdim=True)  # (B,1)

                # gray-world gains from full-res (cheap means)
                means = x_float.mean(dim=(2,3))  # (B,3) RGB
                # convert to BGR-like logic: use G as pivot (channel 1)
                mG = means[:,1:2]
                wb_gain = (mG / (means + 1e-6)).clamp(0.6, 1.6)  # (B,3)

                # EMA per-frame within chunk
                B = x_float.shape[0]
                Ymean_s = torch.empty_like(Ymean)
                Ystd_s  = torch.empty_like(Ystd)
                sharp_s = torch.empty_like(sharp)
                noise_s = torch.empty_like(noise)
                wbg_s   = torch.empty_like(wb_gain)

                for i in range(B):
                    Ym  = Ymean[i:i+1]  # (1,1)
                    Ys  = Ystd[i:i+1]
                    Sh  = sharp[i:i+1]
                    Nl  = noise[i:i+1]
                    Wbg = wb_gain[i:i+1]  # (1,3)

                    ema_Ymean = ema(ema_Ymean, Ym, self.ema_alpha)
                    ema_Ystd  = ema(ema_Ystd,  Ys, self.ema_alpha)
                    ema_sharp = ema(ema_sharp, Sh, self.ema_alpha)
                    ema_noise = ema(ema_noise, Nl, self.ema_alpha)
                    ema_wbgain = ema(ema_wbgain, Wbg, self.ema_alpha)

                    Ymean_s[i:i+1] = ema_Ymean
                    Ystd_s[i:i+1]  = ema_Ystd
                    sharp_s[i:i+1] = ema_sharp
                    noise_s[i:i+1] = ema_noise
                    wbg_s[i:i+1]   = ema_wbgain

                # Params from EMA
                gamma = torch.where(
                    Ymean_s < 0.22, torch.full_like(Ymean_s, 0.70),
                    torch.where(
                        Ymean_s < 0.35, torch.full_like(Ymean_s, 0.80),
                        torch.where(Ymean_s < 0.55, torch.full_like(Ymean_s, 0.90),
                                    torch.full_like(Ymean_s, 1.00))
                    )
                )  # (B,1)

                clahe_clip = torch.where(
                    Ystd_s < 0.05, torch.full_like(Ystd_s, 3.5),
                    torch.where(
                        Ystd_s < 0.08, torch.full_like(Ystd_s, 2.5),
                        torch.where(Ystd_s < 0.12, torch.full_like(Ystd_s, 1.8),
                                    torch.full_like(Ystd_s, 1.2))
                    )
                )  # (B,1)

                # Median selection
                if self.enable_median:
                    ksize = torch.where(
                        noise_s > 0.06, torch.full_like(noise_s, 5.0),
                        torch.where(noise_s > 0.035, torch.full_like(noise_s, 3.0),
                                    torch.full_like(noise_s, 0.0))
                    ).squeeze(1)  # (B,)
                else:
                    ksize = torch.zeros((x_float.shape[0],), device=device, dtype=x_float.dtype)

                # Unsharp amount
                base = torch.where(sharp_s < 0.002, torch.full_like(sharp_s, 0.9),
                                   torch.where(sharp_s < 0.006, torch.full_like(sharp_s, 0.7),
                                               torch.full_like(sharp_s, 0.5)))
                noise_pen = (noise_s.clamp(0, 0.08) / 0.08) * 0.4
                usm_amount = (base - noise_pen).clamp(0.3, 1.0).to(x_float.dtype)  # (B,1)

                # --- apply ---
                xb = x_float

                # 1) median per index (avoid big masks)
                if self.enable_median:
                    idx3 = (ksize == 3).nonzero(as_tuple=False).squeeze(1)
                    idx5 = (ksize == 5).nonzero(as_tuple=False).squeeze(1)
                    if idx3.numel():
                        xb[idx3] = KF.median_blur(xb[idx3], (3,3))
                    if idx5.numel():
                        xb[idx5] = KF.median_blur(xb[idx5], (5,5))

                # 2) gray-world WB
                gains = wbg_s.view(-1,3,1,1).to(xb.dtype)
                xb = (xb * gains).clamp_(0,1)

                # 3) gamma
                gamma_b = gamma.to(xb.dtype).view(-1,1,1,1)
                xb = torch.pow(xb.clamp(0,1), gamma_b)

                # 4) CLAHE per frame (batched loop to keep VRAM bounded)
                # Kornia's CLAHE expects (B,1,H,W); do Y channel round-trip
                ycb = K.color.rgb_to_ycbcr(xb)
                Y = ycb[:, :1]
                out_frames = []
                for i in range(xb.shape[0]):
                    clip = float(clahe_clip[i,0].item())
                    Yi = KE.equalize_clahe(Y[i:i+1], clip_limit=clip, grid_size=(self.clahe_tile, self.clahe_tile))
                    x_rgb = K.color.ycbcr_to_rgb(torch.cat([Yi, ycb[i:i+1,1:]], dim=1))
                    out_frames.append(x_rgb)
                xb = torch.cat(out_frames, dim=0)

                # 5) unsharp + lerp
                sharp_fixed = KF.unsharp_mask(
                    xb, kernel_size=(5,5), sigma=(self.unsharp_sigma,self.unsharp_sigma), border_type='reflect'
                )
                a = usm_amount.to(xb.dtype).view(-1,1,1,1)
                xb = torch.lerp(xb, sharp_fixed, a).clamp_(0,1)

            out[s:e].copy_(xb.to(out_dtype))

            # free temps between chunks
            del x_float, x_small, ycb_small, Y_small, lap, blur, means, wb_gain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return out
    
class ToDtypeDevice:
    def __init__(self, dtype, device:torch.device):
        assert dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64), "dtype not supported, please choose torch.float16 for GPU and torch.float32 for CPU."
        self.device = device
        self.dtype = dtype
    
    def __call__(self, batch: FrameBatch) -> FrameBatch:
        if batch.frames.device.type == "cpu" and self.device.type != "cpu":
            batch.frames = batch.frames.pin_memory()

        non_blocking = (batch.frames.is_pinned() and self.device.type == "cuda")

        batch.frames = batch.frames.to(self.device, dtype=self.dtype, non_blocking=non_blocking)
        batch.device = batch.frames.device
        print("Transferred")
        return batch
    
# helpers / altered from source code
class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, batch: FrameBatch) -> FrameBatch:
        batch.frames = self.resize(batch.frames)
        batch.height, batch.width = batch.frames.shape[2], batch.frames.shape[3]
        print("Resized the frames")
        return batch
    
    def resize(self, frames:torch.Tensor):
        # size can be min_size (scalar) or (w, h) tuple

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None)->Union[Tuple, List]:
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        # tensor format (T, C, H, W)
        size = get_size((frames.shape[2], frames.shape[3]), self.size, self.max_size)
        size = list(size)
        rescaled_frames = VF.resize(frames, size, antialias=True)

        return rescaled_frames
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, batch: FrameBatch) -> FrameBatch:
        batch.frames = VF.normalize(batch.frames, mean=self.mean, std=self.std, inplace=True)
        print("Frames Normalized")
        return batch

class SavedToHistory:
    def __init__(self, save_dir:str, file_name:Optional[str]=None):
        path = os.fspath(save_dir)
        if not path:
            raise ValueError("save_dir must be a non-empty path")

        # If path exists ensure it's a directory, otherwise try to create it
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise NotADirectoryError(f"Path exists and is not a directory: {path}")
        else:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                raise OSError(f"Unable to create directory '{path}': {e}") from e

        self.save_dir = path
        self.file_name = file_name

    def __call__(self, batch: FrameBatch) -> FrameBatch:
        file_stub = self.file_name or f"batch_{batch.batch_id:04d}"
        print(self.tensor_to_video(batch.frames, batch.fps, file_stub))
        print("Saved to history")
        return batch

    def tensor_to_video(self, video_tchw: torch.Tensor, fps: float, file_name: str | None = None) -> str:
        """
        Save a (T, C=3, H, W) RGB tensor as an .mp4 video using OpenCV.

        Returns:
            str: Absolute path to the saved video file.
        """
        # ---- validate inputs ----
        assert isinstance(video_tchw, torch.Tensor), "video_tchw must be a torch.Tensor"
        assert video_tchw.ndim == 4, "Expected tensor with shape (T, C, H, W)"
        T, C, H, W = video_tchw.shape
        assert C == 3, "Expected 3 channels (RGB)"
        assert T > 0, "Video has zero frames"
        assert fps and fps > 0, "fps must be > 0"

        # ---- convert to uint8 RGB on CPU ----
        if video_tchw.is_floating_point():
            x = (video_tchw.clamp(0, 1) * 255.0).to(torch.uint8)
        elif video_tchw.dtype == torch.uint8:
            x = video_tchw
        else:
            # Fallback: clamp numeric types to [0,255] and cast
            x = video_tchw.clamp(0, 255).to(torch.uint8)

        # (T, H, W, C) RGB, contiguous for OpenCV
        video_np = x.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        t, h, w, c = video_np.shape

        # ---- prepare output path ----
        out_dir = Path(self.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not file_name:
            file_name = f"video_{time.strftime('%Y%m%d_%H%M%S')}"

        base = out_dir / file_name
        out_file = base.with_suffix(".mp4")

        # avoid overwrite by suffixing _00, _01, ...
        index = 0
        while out_file.exists():
            out_file = Path(f"{base}_{index:02d}.mp4")
            index += 1

        # ---- open writer ----
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (int(w), int(h)))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {out_file}")

        # ---- write frames (convert RGB -> BGR) ----
        for frame in video_np:
            # ensure contiguous; cv2.cvtColor needs it
            frame = np.ascontiguousarray(frame)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        print(f"Saved {out_file} ({t} frames at {fps} FPS, size {w}x{h})")
        return str(out_file.resolve())


class FPSDownsample:
    def __init__(self, target_fps:float):
        self.target_fps = target_fps

    def __call__(self, batch: FrameBatch) -> FrameBatch:
        if self.target_fps <= 0 or batch.fps <= 0 or self.target_fps >= batch.fps:
            return batch

        interval_ms = 1000.0 / float(self.target_fps)
        keep_mask = torch.zeros(batch.frames.shape[0], dtype=torch.bool)
        next_keep_ms = 0.0
        kept = 0
        for idx, ts in enumerate(batch.timestamps_ms.tolist()):
            if ts + 1e-6 >= next_keep_ms:
                keep_mask[idx] = True
                kept += 1
                next_keep_ms += interval_ms

        if kept == 0:
            return batch

        batch.frames = batch.frames[keep_mask]
        batch.timestamps_ms = batch.timestamps_ms[keep_mask]
        batch.frame_indices = batch.frame_indices[keep_mask]
        batch.fps = self.target_fps
        print("Scaled down FPS")
        return batch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch: FrameBatch) -> FrameBatch:
        for t in self.transforms:
            batch = t(batch)
        print("Finish transforming every frames in the batch")
        return batch

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
