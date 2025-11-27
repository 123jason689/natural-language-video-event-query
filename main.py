import torch

from libs.gdino_process import (
	Model as GDINO,
	save_to_dir_anotated,
)
from libs.preprocess import load_frame_formated
from libs.typings_ import VidTensor
import time
from libs.ocsort.ocsort import OCSort


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_PATH = "./sitting.mp4"
CLASSES = ["person black shoes sitting on a bench"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
BATCH_SIZE = 16
TARGET_FPS = 6
SAVE_ANNOTATED = True
ANNOTATION_DIR = "./history"


def run_pipeline() -> None:
	print('Loading the model, make sure empty memory still available')
	
	model = GDINO(
		"./models/dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
		"./models/dino/GroundingDINO/weights/groundingdino_swint_ogc.pth",
		device=str(DEVICE),
	)

	print('Reading the stream frames and processing based on fps')
	video_stream = VidTensor(
		VIDEO_PATH,
		DEVICE,
		batch_size=BATCH_SIZE,
		target_fps=TARGET_FPS,
	)


	all_results = []
	oc_sort = OCSort(0.3, 120, 3, 0.3)

	print('Preprocessing frames and predicting / detecting objects specified')
	for batch in video_stream:
		curr_time = time.perf_counter()
		processed = load_frame_formated(batch, save_history_dir=None)
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete Enhancement")
		curr_time = time.perf_counter()		

		# batch_results = model.predict_with_classes(
		# 	processed,
		# 	CLASSES,
		# 	BOX_THRESHOLD,
		# 	TEXT_THRESHOLD,
		# )
		
		batch_results = model.predict_with_caption(
			processed,
			CLASSES[0],
			BOX_THRESHOLD,
			TEXT_THRESHOLD,
			oc_sort
		)

		all_results.extend(batch_results)
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete GDINO Detection")


	video_stream.close()

	print('Saving processed frames')
	if SAVE_ANNOTATED:
		curr_time = time.perf_counter()
		save_path = save_to_dir_anotated(video_stream.file_path, all_results, ANNOTATION_DIR)
		if save_path:
			print(f"Annotated video saved to {save_path}")
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete Annotation and Saving")


if __name__ == "__main__":
	run_pipeline()