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
from typing import List, Union, Tuple
from .typings_ import VidTensor

class AutoEnhance:

    def __init__(self, ema_alpha = 0.6, clahe_chunk = 16):
        self.ema_alpha = ema_alpha
        self.clahe_chunk = clahe_chunk

    def __call__(self, video_tchw: VidTensor):
        video_tchw.vid_tensor = self.adaptive_enhance_for_detection_TCHW(video_tchw.vid_tensor, self.ema_alpha, self.clahe_chunk)
        return video_tchw

    @torch.inference_mode()
    def adaptive_enhance_for_detection_TCHW(
        self,
        video_tchw: torch.Tensor,
        ema_alpha: float,
        chunk:int,             # process everything in this temporal chunk
        work_dtype=None,             # choose dtype explicitly to control memory
    ) -> torch.Tensor:
        r"""
        Apply adaptive enhancement to a video tensor (T, C=3, H, W) in RGB.

        The pipeline estimates per-frame diagnostics (luminance/contrast,
        sharpness via variance of Laplacian, noise via HF residual energy,
        gray-world gains), smooths them along time with an EMA, derives
        frame-wise parameters (gamma, CLAHE clip limit, denoise kernel size,
        unsharp amount), then applies:
          1) Optional median denoise (ksize ∈ {0, 3, 5})
          2) Gray-world white balance
          3) Gamma adjustment (gamma < 1 brightens shadows)
          4) CLAHE on luminance in YCbCr (chunked)
          5) Unsharp mask with adaptive blending (chunked)

        Parameters
        ----------
        video_tchw : torch.Tensor
            Input video tensor of shape (T, 3, H, W). Either:
            - `uint8` in [0, 255], or
            - floating point in [0, 1].
            Must be RGB order. Processed on `video_tchw.device`.
        ema_alpha : float
            Temporal EMA smoothing factor applied to diagnostics/params (0–1).
            Larger values reduce flicker but react more slowly to changes.
        clahe_chunk : int
            Number of frames per processing chunk for CLAHE and unsharp passes
            to balance memory and speed.

        Returns
        -------
        torch.Tensor
            Enhanced video tensor of shape (T, 3, H, W), dtype `float32`,
            range [0, 1], on the same device as input.

        Raises
        ------
        AssertionError
            If the input is not 4D (T, C, H, W) or C != 3.

        Notes
        -----
        - The operation is out-of-place with respect to the input tensor object:
          a *new* float tensor is created and returned. (No in-place ops on
          `video_tchw` itself.)
        - Median filtering, gamma, CLAHE, and unsharp use Kornia functional APIs.
        - CLAHE clip limit is selected per-frame but applied per-chunk for
          efficiency; for exact per-frame limits, loop per frame (slower).
        """
        x_in = vidt
        assert x_in.ndim == 4, "Expected (T, C, H, W)"
        T, C, H, W = x_in.shape
        device = x_in.device

        # --- choose working dtype ---
        if work_dtype is None:
            if device.type == "cuda":
                work_dtype = torch.float16  # usually fine for Kornia on GPU; change to bfloat16/float32 if needed
            else:
                work_dtype = torch.float32

        # --- preallocate output (we write final frames here) ---
        out = torch.empty((T, C, H, W), device=device, dtype=work_dtype)

        # --- EMA state (initialized lazily on first frame) ---
        Y_mean_s_prev = None     # (1,)
        Y_std_s_prev  = None     # (1,)
        sharp_s_prev  = None     # (1,)
        noise_s_prev  = None     # (1,)
        wb_gain_s_prev = None    # (C,)

        # helper: one-step EMA update: y_t = alpha * v_t + (1 - alpha) * y_{t-1}
        def ema_step(v_t, y_prev):
            if y_prev is None:
                return v_t
            return ema_alpha * v_t + (1.0 - ema_alpha) * y_prev

        # --- process in temporal chunks end-to-end ---
        for s in range(0, T, chunk):
            e = min(s + chunk, T)

            # ---- load/convert into output slice to avoid extra buffers ----
            # We'll operate in-place on out[s:e].
            if x_in.dtype == torch.uint8:
                # Convert & scale in one pass
                out[s:e].copy_(x_in[s:e].to(dtype=work_dtype))
                out[s:e].div_(255.0)
            else:
                # If already float on same device and same dtype -> cheap view copy
                out[s:e].copy_(x_in[s:e].to(dtype=work_dtype))

            xb = out[s:e]  # (B, C, H, W), B = e - s

            # ---- compute diagnostics per-frame in this chunk ----
            # Luminance & contrast
            ycbcr = K.color.rgb_to_ycbcr(xb)      # alloc (B,3,H,W)
            Y = ycbcr[:, :1]                      # view (B,1,H,W)
            Y_mean = Y.mean(dim=(2, 3))           # (B,1)
            Y_std  = Y.std(dim=(2, 3)) + 1e-8     # (B,1)

            # Sharpness proxy (variance of Laplacian)
            lap = KF.laplacian(Y, kernel_size=3)  # (B,1,H,W)
            sharp = lap.pow(2).mean(dim=(2, 3))   # (B,1)

            # Gray-world gains (color cast)
            means = xb.mean(dim=(2, 3))           # (B,C)
            m_avg = means.mean(dim=1, keepdim=True)     # (B,1)
            wb_gain = (m_avg / (means + 1e-6)).clamp(0.6, 1.6)  # (B,C)

            # Noise proxy (high-frequency energy)
            blur = KF.gaussian_blur2d(xb, (3, 3), (1.0, 1.0))
            hf = xb - blur
            noise_level = hf.abs().mean(dim=(1, 2, 3), keepdim=True)  # (B,1)

            # ---- EMA per frame (streaming across the chunk) ----
            # Prepare holders for this chunk's smoothed stats (we reuse tensors to avoid many small allocs)
            Y_mean_s  = torch.empty_like(Y_mean)
            Y_std_s   = torch.empty_like(Y_std)
            sharp_s   = torch.empty_like(sharp)
            noise_s   = torch.empty_like(noise_level)

            wb_gain_s = torch.empty_like(wb_gain)

            for i in range(e - s):
                # squeeze to scalars/vectors for EMA, then re-expand
                Ym  = Y_mean[i, 0]
                Ys  = Y_std[i, 0]
                Sh  = sharp[i, 0]
                Nl  = noise_level[i, 0, 0, 0]
                Wbg = wb_gain[i]  # (C,)

                Y_mean_s_prev = ema_step(Ym, Y_mean_s_prev)
                Y_std_s_prev  = ema_step(Ys, Y_std_s_prev)
                sharp_s_prev  = ema_step(Sh, sharp_s_prev)
                noise_s_prev  = ema_step(Nl, noise_s_prev)
                wb_gain_s_prev = ema_step(Wbg, wb_gain_s_prev)

                Y_mean_s[i, 0] = Y_mean_s_prev
                Y_std_s[i, 0]  = Y_std_s_prev
                sharp_s[i, 0]  = sharp_s_prev
                noise_s[i, 0, 0, 0] = noise_s_prev
                wb_gain_s[i] = wb_gain_s_prev

            # ---- derive adaptive params for this chunk ----
            # Gamma (<1 brightens)
            gamma = torch.where(
                Y_mean_s < 0.22, torch.full_like(Y_mean_s, 0.70),
                torch.where(
                    Y_mean_s < 0.35, torch.full_like(Y_mean_s, 0.80),
                    torch.where(
                        Y_mean_s < 0.55, torch.full_like(Y_mean_s, 0.90),
                        torch.full_like(Y_mean_s, 1.00),
                    ),
                ),
            )  # (B,1)

            # CLAHE clip limit
            clahe = torch.where(
                Y_std_s < 0.05, torch.full_like(Y_std_s, 3.5),
                torch.where(
                    Y_std_s < 0.08, torch.full_like(Y_std_s, 2.5),
                    torch.where(
                        Y_std_s < 0.12, torch.full_like(Y_std_s, 1.8),
                        torch.full_like(Y_std_s, 1.2),
                    ),
                ),
            )  # (B,1)

            # Denoise kernel size (0,3,5)
            ksize = torch.where(
                noise_s > 0.06, torch.full_like(noise_s, 5.0),
                torch.where(noise_s > 0.035, torch.full_like(noise_s, 3.0),
                            torch.full_like(noise_s, 0.0))
            ).squeeze(-1).squeeze(-1).squeeze(-1)  # -> (B,)


            # Unsharp amount
            base_sharp = torch.where(sharp_s < 0.002, 0.9,
                        torch.where(sharp_s < 0.006, 0.7, 0.5)).to(xb.dtype)  # (B,1)
            noise_penalty = (noise_s.clamp(0, 0.08) / 0.08) * 0.4
            usm_amount = (base_sharp - noise_penalty).clamp(0.3, 1.0)  # (B,1)

            # ---- apply enhancements (in-place where safe) ----

            # 1) Denoise (avoid boolean indexing gather; index lists)
            idx3 = (ksize == 3).nonzero(as_tuple=False).squeeze(1)
            idx5 = (ksize == 5).nonzero(as_tuple=False).squeeze(1)
            if idx3.numel():
                xb[idx3] = KF.median_blur(xb[idx3], (3, 3))
            if idx5.numel():
                xb[idx5] = KF.median_blur(xb[idx5], (5, 5))

            # 2) White balance (per-frame gains), in-place
            gains = wb_gain_s.view(-1, C, 1, 1).to(xb.dtype)
            xb.mul_(gains).clamp_(0, 1)

            # 3) Gamma, in-place
            xb.clamp_(0, 1)
            xb.pow_(gamma.view(-1, 1, 1, 1).to(xb.dtype))

            # 4) CLAHE on luminance; per-frame clip. Loop per frame to avoid a big concat.
            #    (Small loop over B keeps memory low.)
            for i in range(e - s):
                ycbcr_i = K.color.rgb_to_ycbcr(xb[i:i+1])      # (1,3,H,W)
                Y_i = ycbcr_i[:, :1]
                cl = float(clahe[i, 0].item())
                Y2_i = KE.equalize_clahe(Y_i, clip_limit=cl, grid_size=(8, 8))
                ycbcr_i = torch.cat([Y2_i, ycbcr_i[:, 1:]], dim=1)
                xb[i:i+1] = K.color.ycbcr_to_rgb(ycbcr_i).clamp_(0, 1)

            # 5) Unsharp mask; per-frame amount. Loop keeps memory bounded.
            for i in range(e - s):
                amt = float(usm_amount[i, 0].item())
                xb[i:i+1] = KE.unsharp_mask(
                    xb[i:i+1], kernel_size=(5, 5), sigma=(1.5, 1.5),
                    amount=amt, threshold=0.0
                ).clamp_(0, 1)

            # xb already writes into out[s:e] (no extra copies)

        # write back
        return out
    
class ToDtypeDevice:
    def __init__(self, dtype, device:torch.device):
        assert dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64), "dtype not supported, please choose torch.float16 for GPU and torch.float32 for CPU."
        self.device = device
        self.dtype = dtype
    
    def __call__(self, video_tchw:VidTensor):
        video_tchw.vid_tensor.pin_memory()
        video_tchw.vid_tensor = video_tchw.vid_tensor.to(self.device, dtype=self.dtype, non_blocking=True)
        video_tchw.device = self.device
        return video_tchw
    
# helpers / altered from source code
class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, frames:VidTensor):
        frames.vid_tensor = self.resize(frames.vid_tensor)
        frames.height, frames.width = frames.vid_tensor.shape[2], frames.vid_tensor.shape[3]
        return frames
    
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

    def __call__(self, frames:VidTensor):
        frames.vid_tensor = VF.normalize(frames.vid_tensor, mean=self.mean, std=self.std, inplace=True)
        return frames

class SavedToHistory:
    def __init__(self, save_dir:str, file_name:str):
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

    def __call__(self, video_tchw:VidTensor):
        print(self.tensor_to_video(video_tchw.vid_tensor, video_tchw.fps))
        return video_tchw

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

    def __call__(self, vid_tensor:VidTensor):
        self.fps_scale_down_to_tensor(vid_tensor, self.target_fps)
        assert isinstance(vid_tensor, VidTensor), "Not a VidTensor instance anymore"
        assert isinstance(vid_tensor.vid_tensor, torch.Tensor), "Not a VidTensor.vid_tensor instance anymore"
        print(vid_tensor.vid_tensor.ndim)
        return vid_tensor
    
    def fps_scale_down_to_tensor(self, vid_file: VidTensor, fps: float):
        if fps == 0:
            frame = vid_file.vid_tensor[0]
            
            return frame

        vid_fps = vid_file.fps or 0.0
        if vid_fps > 0 and fps > vid_fps:
            print("FPS larger than video fps. Defaulting to native video fps")
            fps = vid_fps

        interval_ms = 1000.0 / float(fps)
        next_keep_ms = 0.0
        frames_keep: torch.Tensor = torch.ones(int(vid_file.total_frame), dtype=torch.bool, device=vid_file.vid_tensor.device)
        dropped = 0
        frame_idx = 0

        while frame_idx < vid_file.total_frame:
            timestamp_ms = vid_file.vid_frame_msec_data[frame_idx]
            if timestamp_ms + 1e-6 >= next_keep_ms:
                next_keep_ms += interval_ms
            else:
                frames_keep[frame_idx] = False
                dropped += 1
            frame_idx += 1

        vid_file.total_frame = vid_file.total_frame - dropped
        vid_file.fps = fps
        vid_file.vid_tensor = vid_file.vid_tensor[frames_keep]
        vid_file.vid_frame_msec_data = vid_file.vid_frame_msec_data[frames_keep]

        assert len(vid_file.vid_tensor) == vid_file.total_frame, "Total frame and kept frame isn't synced"

        print(f"Shrink FPS from {vid_fps} FPS to {fps} retaining {len(vid_file.vid_tensor)} frames while dropping {dropped} frames")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid:VidTensor):
        for t in self.transforms:
            vid.vid_tensor = t(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# python main.py
# FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
# UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
# UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\TensorShape.cpp:4324.)
# final text_encoder_type: bert-base-uncased
# Shrink FPS from 30.0 FPS to 6 retaining 128 frames while dropping 510 frames
# Traceback (most recent call last):
#   File "C:\Users\jason\Docs\programming-project\AI\natural-language-video-event-query\main.py", line 13, in <module>
#     video_formated = load_frame_formated(video_formated) # format for GDino compatiblity
#                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\jason\miniconda3\envs\nlvideo\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\jason\Docs\programming-project\AI\natural-language-video-event-query\libs\preprocess.py", line 171, in load_frame_formated
#     frames_transformed = transform(vid)
#                          ^^^^^^^^^^^^^^
#   File "C:\Users\jason\Docs\programming-project\AI\natural-language-video-event-query\libs\frame_enhancement.py", line 403, in __call__
#     vid.vid_tensor = t(vid)
#                      ^^^^^^
#   File "C:\Users\jason\Docs\programming-project\AI\natural-language-video-event-query\libs\frame_enhancement.py", line 23, in __call__
#     video_tchw.vid_tensor = self.adaptive_enhance_for_detection_TCHW(video_tchw.vid_tensor, self.ema_alpha, self.clahe_chunk)
#                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\jason\miniconda3\envs\nlvideo\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\jason\Docs\programming-project\AI\natural-language-video-event-query\libs\frame_enhancement.py", line 82, in adaptive_enhance_for_detection_TCHW
#     assert video_tchw.ndim == 4, "Expected (T, C, H, W)"
#            ^^^^^^^^^^^^^^^
# AttributeError: 'VidTensor' object has no attribute 'ndim'