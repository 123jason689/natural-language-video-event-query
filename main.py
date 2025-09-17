import cv2
import libs.preprocess as pp
import libs.yoloe_process as yolo

vid = cv2.VideoCapture("cam_footage.mp4")
itter = pp.fps_scale_down(vid, 10, True)
model = yolo.Yolo("./models/yoloe/yolov8l_latest.pt")

model.set_prompt_classes(['person', 'table', 'tree', 'lamp'])

for frame in itter:
    model.predict_prompts([frame])




