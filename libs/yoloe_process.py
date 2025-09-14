from ultralytics import YOLOE
import preprocess
import numpy as np
import cv2
from typing import List, overload, Iterator

# Initialize a YOLOE model
model = YOLOE("./yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()

@overload
def predict_prompts(prompt_classes: List[str], source: List[np.typing.ArrayLike]):...

@overload
def predict_prompts(prompt_classes: List[str], source: Iterator[np.typing.ArrayLike]):...

def predict_prompts(prompt_classes: List[str], source) :
    global model
    model.set_classes(prompt_classes, model.get_text_pe(prompt_classes))

    for frame in source:
        result = model.predict(frame)
        ### continue tthisssssssssss