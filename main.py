import torch

from libs.gdino_process import (
	Model as GDINO,
	save_to_dir_anotated,
)
from libs.preprocess import load_frame_formated
from libs.typings_ import VidTensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_PATH = "cam_footage.mp4"
CLASSES = ["person white shirt", "chair"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
BATCH_SIZE = 16
TARGET_FPS = 6
SAVE_ANNOTATED = True
ANNOTATION_DIR = "./history"


def run_pipeline() -> None:
	model = GDINO(
		"./models/dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
		"./models/dino/GroundingDINO/weights/groundingdino_swint_ogc.pth",
		device=str(DEVICE),
	)

	video_stream = VidTensor(
		VIDEO_PATH,
		torch.device("cpu"),
		batch_size=BATCH_SIZE,
		target_fps=TARGET_FPS,
	)

	all_results = []

	for batch in video_stream:
		processed = load_frame_formated(batch, save_history_dir=None)
		batch_results = model.predict_with_classes(
			processed,
			CLASSES,
			BOX_THRESHOLD,
			TEXT_THRESHOLD,
		)
		all_results.extend(batch_results)

	video_stream.close()

	if SAVE_ANNOTATED:
		save_path = save_to_dir_anotated(video_stream.file_path, all_results, ANNOTATION_DIR)
		if save_path:
			print(f"Annotated video saved to {save_path}")


if __name__ == "__main__":
	run_pipeline()