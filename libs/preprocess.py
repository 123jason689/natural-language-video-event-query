from typing import Iterator, List, overload, Union, Literal
import os
import kornia as K
import kornia.filters as KF
import kornia.enhance as KE
import groundingdino.datasets.transforms as T
import torchvision.transforms.functional as F
from groundingdino.util.box_ops import box_xyxy_to_cxcywh
from groundingdino.util.misc import interpolate
import torch
import math
import cv2
import numpy as np
from frame_enhancement import *

def fps_scale_down_to_np_arr(vid_file: cv2.VideoCapture, fps: int, output_type: Literal['list', 'iter'] = 'list' ) -> Union[List[np.typing.ArrayLike], Iterator[np.typing.ArrayLike]]:
    """
    CAUTION
    -------
    This function is deprecated. Prefer using the to_tensor version of this function

    Load and reduce the FPS by picking fragments the main video frames. 

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_frames, H, W, 3), dtype=uint8,
        containing all video frames in RGB order. Output in (B,H,W,C) format.
    """
    if not vid_file.isOpened():
        raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")

    if fps == 0:
        ret, frame = vid_file.read()
        if not ret:
            return [] if not output_type == 'iter' else iter(())
        arr = np.asarray(frame)
        if output_type == 'iter':
            def gen_one():
                yield arr
            return gen_one()
        return [arr]

    vid_fps = vid_file.get(cv2.CAP_PROP_FPS) or 0.0
    if vid_fps > 0 and fps > vid_fps:
        print("FPS larger than video fps. Defaulting to native video fps")
        fps = int(vid_fps)

    interval_ms = 1000.0 / float(fps)
    next_keep_ms = 0.0
    frames: List[np.typing.ArrayLike] = []
    dropped = 0
    frame_idx = 0

    def iterator():
        ### getting the first outerscope variable with nonlocal
        nonlocal next_keep_ms, dropped, frame_idx
        while True:
            ret, frame = vid_file.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
            timestamp_ms = vid_file.get(cv2.CAP_PROP_POS_MSEC) or (frame_idx / vid_fps * 1000.0 if vid_fps > 0 else frame_idx * interval_ms)
            if timestamp_ms + 1e-6 >= next_keep_ms:
                next_keep_ms += interval_ms
                yield np.asarray(frame_rgb)
            else:
                dropped += 1

    if output_type == 'iter':
        return iterator()

    # collect into list
    for f in iterator():
        frames.append(f)

    print(f"Shrink FPS from {vid_fps} FPS to {len(frames)} frames while dropping {dropped} frames")
    return frames

def fps_scale_down_to_tensor(vid_file: cv2.VideoCapture, fps: int)->torch.Tensor:
    if not vid_file.isOpened():
        raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")

    if fps == 0:
        ret, frame = vid_file.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame)
        
        return tensor

    vid_fps = vid_file.get(cv2.CAP_PROP_FPS) or 0.0
    if vid_fps > 0 and fps > vid_fps:
        print("FPS larger than video fps. Defaulting to native video fps")
        fps = int(vid_fps)

    interval_ms = 1000.0 / float(fps)
    next_keep_ms = 0.0
    vid_frame_count = int(vid_file.get(cv2.CAP_PROP_FRAME_COUNT))
    B = int(max(0, math.floor(((max(vid_frame_count - 1, 0) * float(fps)) / float(vid_fps))) + 1))
    H = int(vid_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vid_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    C = 3 # RGB
    frames: torch.Tensor = torch.empty((B,H,W,C), dtype=torch.uint8)
    dropped = 0

    for frame_idx in range(B):
        ret, frame = vid_file.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = vid_file.get(cv2.CAP_PROP_POS_MSEC) or (frame_idx / vid_fps * 1000.0 if vid_fps > 0 else frame_idx * interval_ms)
        if timestamp_ms + 1e-6 >= next_keep_ms:
            next_keep_ms += interval_ms
            frames[frame_idx] = torch.from_numpy(frame_rgb)
        else:
            dropped += 1

    vid_file.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Shrink FPS from {vid_fps} FPS to {len(frames)} frames while dropping {dropped} frames")
    return frames

def vid_to_np_arr(vid_file: cv2.VideoCapture) -> np.typing.NDArray[np.uint8]:
    """
    Load all frames from a video file.

    Parameters
    ----------
    path : cv2.VideoCapture
        A cv2.VideoCapture type object.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_frames, H, W, 3), dtype=uint8,
        containing all video frames in RGB order.
    """

    if not vid_file.isOpened():
        raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")
    
    frames = []

    while True:
        ret, frame = vid_file.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return np.stack(frames)

def vid_to_torch_tensor(vid_file: cv2.VideoCapture, return_dtype:torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load all frames from a video file to torch tensor format ready for .

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_frames, H, W, 3), dtype=uint8,
        containing all video frames in RGB order.
    """
    if not vid_file.isOpened():
        raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")

    T = int(vid_file.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(vid_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vid_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    C = 3 # RGB

    frames: torch.Tensor = torch.empty((T,H,W,C), dtype=torch.uint8)

    for frame_idx in range(T):
        ret, frame = vid_file.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[frame_idx] = torch.from_numpy(frame)

    permute_THWC_to_TCHW(frames).to(dtype=return_dtype)

    return frames

def permute_THWC_to_TCHW(img_tensor:torch.Tensor)->torch.Tensor:
    return img_tensor.permute(0,3,1,2)

## input (B,H,W,C)
@torch.inference_mode()
def load_frame_formated(frames:torch.Tensor, device)->torch.Tensor:
    if frames.shape[-1] < frames.shape[1]:
        frames = permute_THWC_to_TCHW(frames)
    transform = T.Compose(
        [
            To_Dtype_Device(torch.float32, device),
            Auto_Enhance(),
            Resize(800, max_size=1333),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    frames_transformed, _ = transform(frames, None)
    return frames_transformed

# helpers / altered from source code
class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, frames, target=None):
        return self.resize(frames, target), target
    
    def resize(self, frames):
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

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        # tensor format (T, C, H, W)
        size = get_size((frames.shape[2], frames.shape[3]), self.size, self.max_size)
        rescaled_frames = F.resize(frames, size, antialias=True)

        return rescaled_frames
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames, target=None):
        frames = F.normalize(frames, mean=self.mean, std=self.std)
        if target is None:
            return frames, None
        return frames, target

class SavedToHistory:
    def __init__(self, save_dir:str):
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

    def __call__(self, video_tchw:torch.tensor, target = None):
        ##
        ##
        ## MAKE SURE TO FIX THIS, THE FPS IS STATIC, NEEDS TO REFERENCE THE ORIGINAL VIDEO
        ## can be made into an object / class style before passing in
        ##
        ##
        self.tensor_to_video(video_tchw, 30)
        return video_tchw, target

    def tensor_to_video(self, video_tchw:torch.tensor, fps:int, file_name=None):

        assert video_tchw.ndim == 4, "Expected tensor with (T,C,H,W)"
        assert video_tchw.shape[1] == 3, "Expected Channel to be RGB"

        if video_tchw.is_floating_point():
            video_np = (video_tchw.clamp(0,1) * 255).byte().cpu().numpy()
        else:
            video_np = video_tchw.cpu().numpy()
        
        video_np = np.transpose(video_np, (0,2,3,1))
        t,h,w,c = video_np.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        out_path = self.save_dir if self.save_dir[-1] is "/" else self.save_dir+"/"
        out_file = f"{out_path}{file_name}.mp4"

        index = 0
        while os.path.isfile(out_file):
            out_file = f"{out_path}{file_name}_{index:02d}.mp4"
            index += 1
            

        writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

        for frame in video_np:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Saved {out_path} ({t} frames at {fps} FPS, size {w}x{h})")