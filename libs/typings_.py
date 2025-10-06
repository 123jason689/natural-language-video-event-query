from typing import List, Tuple

import torch
import cv2

class VidTensor:
    def __init__(self, vid_path:str, load_device:torch.device):
        """
        _attributes are the original attributes no changes
        """
        vid = cv2.VideoCapture(vid_path)

        self._file_path = vid_path
        self._fps = vid.get(cv2.CAP_PROP_FPS)
        self._total_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self._height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.device = load_device
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.total_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        (self.vid_tensor, self.vid_frame_msec_data) = self.bgr_vid_to_rgb_tensor(vid, torch.device("cpu"))
        vid.release()

    def bgr_vid_to_rgb_tensor(self, vid_file: cv2.VideoCapture, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read all frames from an opened cv2.VideoCapture, convert BGR->RGB and
        return a video tensor together with per-frame timestamps.

        Parameters
        ----------
        vid_file : cv2.VideoCapture
            An opened VideoCapture instance to read frames from.
        device : torch.device
            Target device for the returned video tensor. If `device.type == 'cpu'`
            the video tensor is returned as float32 on that device; otherwise
            it is returned as float16 on the given (non-CPU) device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple (video_tensor, timestamps) where:
            - video_tensor: torch.Tensor of shape (T, C, H, W) with C=3 (RGB).
              Dtype is float32 on CPU or float16 on non-CPU devices, and the
              tensor is moved to `device`.
            - timestamps: torch.Tensor of shape (T,) containing the frame
              timestamps in milliseconds (float, one value per frame).
        """
        if not vid_file.isOpened():
            raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")

        frames:List = []
        frames_msec:List = []

        while True:
            ret, frame = vid_file.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
            frames_msec.append(vid_file.get(cv2.CAP_PROP_POS_MSEC))
        
        thwc = torch.stack(frames, dim=0)

        assert thwc.shape[0] == len(frames_msec), "Frame's time stamp data not synchronized"

        if device.type == "cpu":
            return (self.permute_THWC_to_TCHW(thwc).contiguous().to(device, dtype=torch.float32), torch.tensor(frames_msec).to(device))
        else:
            return (self.permute_THWC_to_TCHW(thwc).contiguous().to(device, dtype=torch.float16), torch.tensor(frames_msec).to(device))
    
    def permute_THWC_to_TCHW(self, img_tensor:torch.Tensor)->torch.Tensor:
        if img_tensor.shape[1] != 3:
            return img_tensor.permute(0,3,1,2)
        else:
            return img_tensor