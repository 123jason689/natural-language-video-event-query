from typing import Iterator, List, overload, Union
import cv2
import numpy as np

def fps_scale_down(vid_file: cv2.VideoCapture, fps: int, as_itter: bool = False) -> Union[List[np.typing.ArrayLike], Iterator[np.typing.ArrayLike]]:
    if not vid_file.isOpened():
        raise ValueError("File is not opened, VideoCapture instance is empty. Make sure to load the video first")

    if fps == 0:
        ret, frame = vid_file.read()
        vid_file.release()
        if not ret:
            return [] if not as_itter else iter(())
        arr = np.asarray(frame)
        if as_itter:
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
            frame_idx += 1
            timestamp_ms = vid_file.get(cv2.CAP_PROP_POS_MSEC) or (frame_idx / vid_fps * 1000.0 if vid_fps > 0 else frame_idx * interval_ms)
            if timestamp_ms + 1e-6 >= next_keep_ms:
                next_keep_ms += interval_ms
                yield np.asarray(frame)
            else:
                dropped += 1
        vid_file.release()

    if as_itter:
        return iterator()

    # collect into list
    for f in iterator():
        frames.append(f)

    print(f"Shrink FPS from {vid_fps} FPS to {len(frames)} frames while dropping {dropped} frames")
    return frames
