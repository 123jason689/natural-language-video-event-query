from ultralytics import YOLOE
import numpy as np
import cv2
from typing import List, overload, Iterator, Union
import os

class Yolo:
    def __init__(self, model_weight_path:str):
        try:
            if not os.path.isfile(model_weight_path):
                raise FileNotFoundError(f"The YOLOE model weights not found")
            else:
                self.model = YOLOE(model_weight_path)
        except:
            raise TimeoutError("Model weight not loaded, Something causes an error!")
        self.prompt_classes = list()

    def add_class(self, class_name: str):
        if class_name not in self.prompt_classes:
            self.prompt_classes.append(class_name)
            self.model.set_classes(self.prompt_classes, self.model.get_text_pe(self.prompt_classes))

    def remove_class(self, class_name: str):
        if class_name in self.prompt_classes:
            self.prompt_classes.remove(class_name)
            self.model.set_classes(self.prompt_classes, self.model.get_text_pe(self.prompt_classes))

    def get_prompt_classes(self) -> List[str]:
        return self.prompt_classes.copy()

    def set_prompt_classes(self, new_prompt_classes:List[str]):
        self.prompt_classes = new_prompt_classes
        self.model.set_classes(self.prompt_classes, self.model.get_text_pe(self.prompt_classes))

    def clear_prompt_classes(self):
        self.prompt_classes.clear()
        self.model.set_classes(self.prompt_classes, self.model.get_text_pe(self.prompt_classes))
        
    def predict_prompts(self, source:List[np.typing.ArrayLike]) :
        for frame in source:
            result = self.model.predict(frame, imgsz=1280)
            result[0].show()
            print(result[0].boxes)
            ### results extract boxes, and 