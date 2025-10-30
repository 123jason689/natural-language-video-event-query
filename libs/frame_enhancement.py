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
from typing import List, Union, Tuple, Optional, Literal, Dict
from .typings_ import FrameBatch

Op = Literal["median", "white_balance", "gamma", "clahe", "unsharp"]

class AutoEnhanceSelective:
    """
    Selective video enhancement for tensors shaped (T,3,H,W), RGB, in [0,1] or uint8.
    Only applies operations listed in `ops`, in that order.

    Modes:
      - auto_tune=True: parameters derived from lightweight diagnostics (EMA).
      - auto_tune=False: uses fixed params you provide.

    Ops:
      - "median": median filter (3x3 or 5x5 when auto; fixed ksize otherwise)
      - "white_balance": gray-world gains (auto) or fixed per-channel gains
      - "gamma": gamma correction (auto from Ymean; fixed otherwise)
      - "clahe": CLAHE on Y channel (auto from Ystd; fixed otherwise)
      - "unsharp": unsharp mask + lerp (auto amount from sharp+noise; fixed otherwise)

    GPU path is chunked to respect `max_gpu_pixels_per_batch` and uses AMP on CUDA.
    Output dtype: float32 on CPU, `gpu_dtype` on GPU; range [0,1].
    """

    def __init__(
        self,
        ops: List[Op],
        *,
        auto_tune: bool = True,
        ema_alpha: float = 0.85,
        use_gpu: bool = True,
        gpu_dtype: torch.dtype = torch.float16,
        max_gpu_pixels_per_batch: int = 1280*720*4,
        clahe_tile: int = 8,
        # Fixed params used when auto_tune=False
        fixed_params: Optional[Dict[str, float]] = None,
        # Diagnostics scale
        diag_short_side: int = 256,
        # Unsharp kernel sigma
        unsharp_sigma: float = 1.5,
    ):
        self.ops = list(ops)
        self.auto_tune = bool(auto_tune)
        self.ema_alpha = float(ema_alpha)
        self.use_gpu = bool(use_gpu)
        self.gpu_dtype = gpu_dtype
        self.max_gpu_pixels_per_batch = int(max_gpu_pixels_per_batch)
        self.clahe_tile = int(clahe_tile)
        self.diag_short_side = int(diag_short_side)
        self.unsharp_sigma = float(unsharp_sigma)

        self.fixed = {
            # median
            "median_ksize": 0,                # 0 disables
            # white balance gains (r,g,b)
            "wb_r": 1.0, "wb_g": 1.0, "wb_b": 1.0,
            # gamma
            "gamma": 1.0,
            # clahe
            "clahe_clip": 2.0,
            # unsharp
            "unsharp_amount": 0.5,
        }
        if fixed_params:
            self.fixed.update({k: float(v) for k, v in fixed_params.items()})

        self._wb_prev_bgr = None
        self._wb_prev_rgb_t = None

        # CPU helpers
        self._clahe_cv = cv2.createCLAHE(clipLimit=self.fixed["clahe_clip"],
                                         tileGridSize=(self.clahe_tile, self.clahe_tile))
        self._gamma_val = None
        self._gamma_lut = None

        # EMA state (kept tiny)
        self._ema_Ymean = None
        self._ema_Ystd  = None
        self._ema_sharp = None
        self._ema_noise = None
        self._ema_wbgain = None  # np.float32[3] in BGR-order for CPU, tensor[3] RGB for GPU

    def __call__(self, batch):
        x = batch.frames  # (T,3,H,W)
        if x.device.type != "cpu" and self.use_gpu:
            batch.frames = self._enhance_gpu_selective(x)
        else:
            batch.frames = self._enhance_cpu_selective(x)
        return batch

    # ---------------- Internals: small utilities ---------------- #
    @staticmethod
    def _to_u8_frame(f: torch.Tensor) -> torch.Tensor:
        if f.dtype == torch.uint8:
            return f
        if f.is_floating_point():
            return (f.clamp(0,1) * 255.0).to(torch.uint8)
        return f.clamp(0,255).to(torch.uint8)

    @staticmethod
    def _scale_for_diag(h: int, w: int, short_side: int) -> float:
        return min(1.0, short_side / max(h, w))

    @staticmethod
    def _lut_gamma_u8(gamma: float) -> np.ndarray:
        inv = max(gamma, 1e-3)
        return (np.power(np.arange(256, dtype=np.float32)/255.0, 1.0/inv)*255.0).clip(0,255).astype(np.uint8)

    def _ema(self, prev, cur):
        if prev is None: return cur
        return self.ema_alpha * cur + (1.0 - self.ema_alpha) * prev

    # ---------------- CPU path ---------------- #
    @torch.inference_mode()
    def _enhance_cpu_selective(self, video_tchw: torch.Tensor) -> torch.Tensor:
        assert video_tchw.ndim == 4 and video_tchw.shape[1] == 3
        T, C, H, W = video_tchw.shape
        device = video_tchw.device
        out = torch.empty((T, C, H, W), dtype=torch.float32, device=device)

        need_noise  = ("median" in self.ops) or ("unsharp" in self.ops)
        need_sharp  = ("unsharp" in self.ops)
        need_Ymean  = ("gamma" in self.ops) and self.auto_tune
        need_Ystd   = ("clahe" in self.ops) and self.auto_tune
        need_wb     = ("white_balance" in self.ops)

        for i in range(T):
            f = self._to_u8_frame(video_tchw[i])
            rgb = f.permute(1,2,0).contiguous().cpu().numpy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Diagnostics only if needed
            if any([need_noise, need_sharp, need_Ymean, need_Ystd, need_wb]):
                scale = self._scale_for_diag(*bgr.shape[:2], self.diag_short_side)
                ds = cv2.resize(bgr, (int(bgr.shape[1]*scale), int(bgr.shape[0]*scale)),
                                interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

                if need_sharp or need_noise:
                    gray = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
                if need_sharp:
                    sharp = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=3).var()
                    self._ema_sharp = self._ema(self._ema_sharp, float(sharp))
                if need_noise:
                    blur = cv2.GaussianBlur(ds, (3,3), 1.0)
                    noise = cv2.absdiff(ds, blur).mean()
                    self._ema_noise = self._ema(self._ema_noise, float(noise))

                if need_Ymean or need_Ystd:
                    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                    Y = ycrcb[...,0].astype(np.float32)/255.0
                    if need_Ymean:
                        self._ema_Ymean = self._ema(self._ema_Ymean, float(Y.mean()))
                    if need_Ystd:
                        self._ema_Ystd  = self._ema(self._ema_Ystd,  float(Y.std() + 1e-8))

                if need_wb:
                    means = bgr.reshape(-1,3).mean(axis=0) + 1e-6  # B,G,R
                    g = means[1]
                    gw = np.array([g/means[0], 1.0, g/means[2]], dtype=np.float32)
                    self._ema_wbgain = gw if self._ema_wbgain is None else self._ema(self._ema_wbgain, gw)

            # Apply only requested ops, in order
            for op in self.ops:
                if op == "median":
                    k = 0
                    if self.auto_tune:
                        if self._ema_noise is not None and self._ema_noise > 0.06*255: k = 5
                        elif self._ema_noise is not None and self._ema_noise > 0.035*255: k = 3
                    else:
                        k = int(self.fixed["median_ksize"])
                    if k in (3,5):
                        bgr = cv2.medianBlur(bgr, k)

                elif op == "white_balance":
                    # 1) choose target gains in BGR order
                    if self.auto_tune and self._ema_wbgain is not None:
                        gains = self._ema_wbgain.astype(np.float32)  # B,G,R from diagnostics
                    else:
                        gains = np.array(
                            [self.fixed["wb_b"], self.fixed["wb_g"], self.fixed["wb_r"]],
                            dtype=np.float32
                        )

                    # 2) absolute clamp to keep colors sane under colored lights
                    gains = np.clip(gains, 0.7, 1.3)

                    # 3) slew-rate limit vs previous frame to prevent hue flipping
                    prev = getattr(self, "_wb_prev_bgr", None)
                    if prev is not None:
                        max_step = 0.05  # allow at most ±5% change per frame
                        lo = prev * (1.0 - max_step)
                        hi = prev * (1.0 + max_step)
                        gains = np.clip(gains, lo, hi)

                    # 4) apply and stash for next time
                    bgr = np.clip(bgr.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
                    self._wb_prev_bgr = gains

                elif op == "gamma":
                    if self.auto_tune and self._ema_Ymean is not None:
                        Ym = self._ema_Ymean
                        if Ym < 0.22: gamma = 0.70
                        elif Ym < 0.35: gamma = 0.80
                        elif Ym < 0.55: gamma = 0.90
                        else: gamma = 1.00
                    else:
                        gamma = float(self.fixed["gamma"])
                    if self._gamma_val != gamma:
                        self._gamma_lut = self._lut_gamma_u8(gamma)
                        self._gamma_val = gamma
                    bgr = cv2.LUT(bgr, self._gamma_lut)

                elif op == "clahe":
                    if self.auto_tune and self._ema_Ystd is not None:
                        Ys = self._ema_Ystd
                        if Ys < 0.05: clip = 3.5
                        elif Ys < 0.08: clip = 2.5
                        elif Ys < 0.12: clip = 1.8
                        else: clip = 1.2
                    else:
                        clip = float(self.fixed["clahe_clip"])
                    self._clahe_cv.setClipLimit(clip)
                    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
                    y = ycrcb[...,0]
                    y[:] = self._clahe_cv.apply(y)
                    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

                elif op == "unsharp":
                    # form sharpened version once, then lerp by amount
                    blurred = cv2.GaussianBlur(bgr, (0,0), sigmaX=self.unsharp_sigma, sigmaY=self.unsharp_sigma)
                    if self.auto_tune:
                        base = 0.9 if (self._ema_sharp or 0) < 0.002 else (0.7 if (self._ema_sharp or 0) < 0.006 else 0.5)
                        noise_pen = min(max((self._ema_noise or 0)/(0.08*255), 0.0), 1.0) * 0.4
                        a = float(np.clip(base - noise_pen, 0.0, 1.0))
                    else:
                        a = float(self.fixed["unsharp_amount"])
                    bgr = cv2.addWeighted(bgr, 1.0 + a, blurred, -a, 0)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_f = torch.from_numpy(rgb).to(dtype=torch.float32, device=device).permute(2,0,1) / 255.0
            out[i].copy_(rgb_f)

        return out

    # ---------------- GPU path ---------------- #
    @torch.inference_mode()
    def _enhance_gpu_selective(self, video_tchw: torch.Tensor) -> torch.Tensor:
        assert video_tchw.ndim == 4 and video_tchw.shape[1] == 3
        device = video_tchw.device
        T, C, H, W = video_tchw.shape

        pixels_per_frame = H * W
        max_frames = max(1, self.max_gpu_pixels_per_batch // max(1, pixels_per_frame))
        Bchunk = min(max_frames, 32)

        out_dtype = self.gpu_dtype if device.type != "cpu" else torch.float32
        out = torch.empty((T, C, H, W), dtype=out_dtype, device=device)

        need_noise  = ("median" in self.ops) or ("unsharp" in self.ops)
        need_sharp  = ("unsharp" in self.ops)
        need_Ymean  = ("gamma" in self.ops) and self.auto_tune
        need_Ystd   = ("clahe" in self.ops) and self.auto_tune
        need_wb     = ("white_balance" in self.ops)

        use_amp = (device.type == "cuda")

        # tiny EMA tensors on device
        ema_Ymean = None
        ema_Ystd  = None
        ema_sharp = None
        ema_noise = None
        ema_wbg   = None  # (3,)

        def ema_t(prev, cur):
            if prev is None: return cur
            return self.ema_alpha * cur + (1.0 - self.ema_alpha) * prev

        for s in range(0, T, Bchunk):
            e = min(s + Bchunk, T)
            xb0 = video_tchw[s:e]

            if xb0.dtype == torch.uint8:
                xb = xb0.to(out_dtype) / 255.0
            else:
                xb = xb0.to(out_dtype)
                if float(xb.max()) > 1.5:
                    xb = xb / 255.0

            with torch.autocast(device_type='cuda', dtype=self.gpu_dtype, enabled=use_amp):
                # Diagnostics only if needed
                if any([need_noise, need_sharp, need_Ymean, need_Ystd, need_wb]):
                    short = self.diag_short_side
                    scale = min(1.0, short / max(H, W))
                    x_small = F.interpolate(xb, size=(int(H*scale), int(W*scale)), mode="bilinear", align_corners=False) if scale < 1.0 else xb

                    if need_Ymean or need_Ystd:
                        ycb_s = K.color.rgb_to_ycbcr(x_small)
                        Y_s = ycb_s[:, :1]
                        Ym = Y_s.mean(dim=(2,3))  # (B,1)
                        Ys = Y_s.std(dim=(2,3)) + 1e-8

                    if need_sharp:
                        lap = KF.laplacian((x_small[:, :1] if x_small.shape[1] == 1 else K.color.rgb_to_grayscale(x_small)), kernel_size=3)
                        Sh = (lap**2).mean(dim=(2,3))  # (B,1)
                    if need_noise:
                        blur = KF.gaussian_blur2d(x_small, (3,3), (1.0,1.0))
                        Nl = (x_small - blur).abs().mean(dim=(1,2,3), keepdim=True)  # (B,1)

                    if need_wb:
                        means = xb.mean(dim=(2,3))  # (B,3) RGB
                        g = means[:,1:2]
                        Wbg = (g / (means + 1e-6)).clamp(0.6, 1.6)  # (B,3) RGB

                    # fold into EMA per item
                    B = xb.shape[0]
                    if need_Ymean:  # overwrite with most recent EMA per frame
                        for i in range(B):
                            ema_Ymean = ema_t(ema_Ymean, Ym[i:i+1])
                    if need_Ystd:
                        for i in range(B):
                            ema_Ystd  = ema_t(ema_Ystd,  Ys[i:i+1])
                    if need_sharp:
                        for i in range(B):
                            ema_sharp = ema_t(ema_sharp, Sh[i:i+1])
                    if need_noise:
                        for i in range(B):
                            ema_noise = ema_t(ema_noise, Nl[i:i+1])
                    if need_wb:
                        for i in range(B):
                            ema_wbg   = ema_t(ema_wbg,   Wbg[i:i+1])

                # Apply only requested ops, in order
                x = xb
                for op in self.ops:
                    if op == "median":
                        if self.auto_tune and ema_noise is not None:
                            k = 5 if float(ema_noise.item()) > 0.06 else (3 if float(ema_noise.item()) > 0.035 else 0)
                        else:
                            k = int(self.fixed["median_ksize"])
                        if k in (3,5):
                            x = KF.median_blur(x, (k,k))

                    elif op == "white_balance":
                        # 1) pick target gains (RGB order on GPU)
                        if self.auto_tune and ema_wbg is not None:
                            gains = ema_wbg.view(1, 3, 1, 1).to(x.dtype)
                        else:
                            gains = torch.tensor(
                                [self.fixed["wb_r"], self.fixed["wb_g"], self.fixed["wb_b"]],
                                dtype=x.dtype, device=x.device
                            ).view(1, 3, 1, 1)

                        # 2) clamp absolute range to avoid color overcorrection
                        gains = gains.clamp(0.7, 1.3)

                        # 3) slew-rate limit vs previous applied gains
                        prev = getattr(self, "_wb_prev_rgb_t", None)
                        if prev is not None:
                            max_step = 0.05  # allow ±5% change per frame
                            lo = prev * (1.0 - max_step)
                            hi = prev * (1.0 + max_step)
                            gains = torch.max(torch.min(gains, hi), lo)

                        # 4) apply and store for next batch
                        x = (x * gains).clamp_(0, 1)
                        self._wb_prev_rgb_t = gains.detach()

                    elif op == "gamma":
                        if self.auto_tune and ema_Ymean is not None:
                            Ym = float(ema_Ymean.item())
                            gamma = 0.70 if Ym < 0.22 else (0.80 if Ym < 0.35 else (0.90 if Ym < 0.55 else 1.00))
                        else:
                            gamma = float(self.fixed["gamma"])
                        x = torch.pow(x.clamp(0,1), torch.tensor(gamma, dtype=x.dtype, device=x.device))

                    elif op == "clahe":
                        if self.auto_tune and ema_Ystd is not None:
                            Ys = float(ema_Ystd.item())
                            clip = 3.5 if Ys < 0.05 else (2.5 if Ys < 0.08 else (1.8 if Ys < 0.12 else 1.2))
                        else:
                            clip = float(self.fixed["clahe_clip"])
                        ycb = K.color.rgb_to_ycbcr(x)
                        Y = ycb[:, :1]
                        Yeq = KE.equalize_clahe(Y, clip_limit=clip, grid_size=(self.clahe_tile, self.clahe_tile))
                        x = K.color.ycbcr_to_rgb(torch.cat([Yeq, ycb[:,1:]], dim=1))

                    elif op == "unsharp":
                        sharpened = KF.unsharp_mask(x, kernel_size=(5,5),
                                                    sigma=(self.unsharp_sigma,self.unsharp_sigma),
                                                    border_type='reflect')
                        if self.auto_tune:
                            base = 0.9 if (float(ema_sharp.item()) if ema_sharp is not None else 0) < 0.002 \
                                   else (0.7 if (float(ema_sharp.item()) if ema_sharp is not None else 0) < 0.006 else 0.5)
                            noise_pen = min(max((float(ema_noise.item()) if ema_noise is not None else 0)/0.08, 0.0), 1.0) * 0.4
                            a = max(0.0, min(1.0, base - noise_pen))
                        else:
                            a = float(self.fixed["unsharp_amount"])
                        x = torch.lerp(x, sharpened, torch.tensor(a, dtype=x.dtype, device=x.device)).clamp_(0,1)

                out[s:e].copy_(x.to(out_dtype))

            del xb0, xb
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
