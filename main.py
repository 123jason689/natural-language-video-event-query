import torch

from libs.gdino_process import (
	Model as GDINO,
	save_to_dir_anotated,
)
from libs.mobileviclip_process import Model as MobileViClip
from libs.mobileviclip_preprocessing import MobileViClipPreprocessor
from libs.preprocess import load_frame_formated
from libs.typings_ import VidTensor, ObjectMap
import time
from libs.ocsort.ocsort import OCSort
# import mobileviclip.models
# import mobileviclip.utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VIDEO_PATH = "./cam_footage.mp4"
# CLASSES = ["person white shirt sitting on a chair with a table"]

VIDEO_PATH = "./sitting.mp4"
PROMPT = "person black jacket and black shoes sitting on a bench"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
BATCH_SIZE = 16
TARGET_FPS = 6
SAVE_ANNOTATED = True
ANNOTATION_DIR = "./history"

GDINO_CONFIG = "./models/dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_BASE_CKPT = "./models/dino/GroundingDINO/weights/groundingdino_swint_ogc.pth"

MOBILE_CLIP_CONFIG = "models/mobileviclip/mobileviclip/scripts/evaluation/clip/zero_shot/mobileviclip_small/config.py"
MOBILE_CLIP_BASE_CKPT = "models/mobileviclip/weights/mobileclip_s2.pt" 
MOBILE_CLIP_FINETUNED_CKPT = "models/mobileviclip/weights/mobileviclip_small.pt"


def run_pipeline() -> None:
	print('Loading the model, make sure empty memory still available')
	
	gdino = GDINO(
		GDINO_CONFIG,
		GDINO_BASE_CKPT,
		device=str(DEVICE),
	)

	mvc = MobileViClip(
		MOBILE_CLIP_CONFIG,
		MOBILE_CLIP_BASE_CKPT,
		MOBILE_CLIP_FINETUNED_CKPT,
		device=str(DEVICE)
	)

	mvc_pp = MobileViClipPreprocessor(
        target_size=256, # MobileViCLIP small pre-trained weigths uses 256
        clip_length=8, 
		clip_duration=4.0,
        clip_stride=4,
        person_keywords=None
    )

	print('Reading the stream frames and processing based on fps')
	video_stream = VidTensor(
		VIDEO_PATH,
		DEVICE,
		batch_size=BATCH_SIZE,
		target_fps=TARGET_FPS,
	)


	all_results = []
	oc_sort = OCSort(BOX_THRESHOLD, 120, 3, 0.3, coasting_thresh=3)

	object_map = ObjectMap()

	print('Preprocessing frames and predicting / detecting objects specified')
	for batch in video_stream:
		curr_time = time.perf_counter()
		processed = load_frame_formated(batch, save_history_dir=None)
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete Enhancement")
		curr_time = time.perf_counter()		
		
		batch_results = gdino.predict_with_caption(
			processed,
			PROMPT,
			BOX_THRESHOLD,
			TEXT_THRESHOLD,
			oc_sort,
			object_map
		)

		all_results.extend(batch_results)
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete GDINO Detection")

	video_stream.close()

	print("Preprocess all results for MobileViCLIP")
	track_clips = mvc_pp.run(
		VIDEO_PATH, 
		all_results, 
		object_map, 
		top_k=3
	)

	results_timeline = mvc.find_event(track_clips, PROMPT, mvc_pp.clip_stride, TARGET_FPS, mvc_pp.clip_duration)

	print(f"\nRaw matching clips found: {len(results_timeline)}")

	# Use a threshold (e.g., 0.5s) to bridge small gaps where detection might flicker
	final_events = mvc.process_timeline(results_timeline, overlap_threshold=0.5)
	print(f"Merged into {len(final_events)} distinct events.")
	print("\n=== TOP DETECTED EVENTS ===")
	for i, res in enumerate(final_events[:5]):
		duration = res['end'] - res['start']
		print(f"{i+1}. Track {res['track_id']} | {res['start']:.1f}s - {res['end']:.1f}s ({duration:.1f}s) | Conf: {res['score']:.3f}")

	print('Saving processed frames')
	if SAVE_ANNOTATED:
		curr_time = time.perf_counter()
		save_path = save_to_dir_anotated(video_stream.file_path, all_results, object_map, ANNOTATION_DIR)
		if save_path:
			print(f"Annotated video saved to {save_path}")
		end_time = time.perf_counter()
		print(f"Took {(end_time - curr_time):.4f} seconds to complete Annotation and Saving")


if __name__ == "__main__":
	run_pipeline()