from typing import Iterator, List, overload, Union, Literal
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

def vid_to_torch_tensor(vid_file: cv2.VideoCapture) -> torch.Tensor:
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

    B = int(vid_file.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(vid_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vid_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    C = 3 # RGB

    frames: torch.Tensor = torch.empty((B,H,W,C), dtype=torch.uint8)

    for frame_idx in range(B):
        ret, frame = vid_file.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[frame_idx] = torch.from_numpy(frame)

    return frames

def permute_THWC_to_TCHW(img_tensor:torch.Tensor)->torch.Tensor:
    return img_tensor.permute(0,3,1,2)

## input (B,H,W,C)
@torch.inference_mode()
def load_frame_formated(frames:torch.Tensor)->torch.Tensor:
    if frames.shape[-1] < frames.shape[1]:
        frames = permute_THWC_to_TCHW(frames)
    transform = T.Compose(
        [
            Resize_CV2(800, max_size=1333),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    frames_transformed, _ = transform(frames, None)
    return frames_transformed

# helpers / altered from source code
class Resize_CV2(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, frames, target=None):
        return self.resize(frames, target)
    
    def resize(self, frames, target):
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

        return rescaled_frames, None
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames, target=None):
        frames = F.normalize(frames, mean=self.mean, std=self.std)
        if target is None:
            return frames, None
        return frames, target
