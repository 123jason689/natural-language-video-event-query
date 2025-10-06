import cv2
from libs.preprocess import VidTensor, load_frame_formated
from libs.gdino_process import Model as GDINO
import libs.yoloe_process as yolo
import torch

DEVICE = "cpu"

model = GDINO("./models/dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./models/dino/GroundingDINO/weights/groundingdino_swint_ogc.pth", device=DEVICE)

video_formated = VidTensor("cam_footage.mp4", torch.device(DEVICE)) #change it your self

video_formated = load_frame_formated(video_formated) # format for GDino compatiblity

raw_input = model.predict_with_classes(video_formated, ["person white shirt", "chair"], 0.35, 0.25, True, "./history")

## Inference 
# output = predict(vid_formated, request)


