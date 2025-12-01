import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy


# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining image datasets
    internvid_v1=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_1.json",
        data_root="data/video_sample/InternVId-FLT_1",
        media_type="video",
        jump_filter=True
    ),
    internvid_v2=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_2.json",
        data_root="data/video_sample/InternVId-FLT_2",
        media_type="video",
        jump_filter=True
    ),
    internvid_v3=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_3.json",
        data_root="data/video_sample/InternVId-FLT_3",
        media_type="video",
        jump_filter=True
    ),
    internvid_v4=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_4.json",
        data_root="data/video_sample/InternVId-FLT_4",
        media_type="video",
        jump_filter=True
    ),
    internvid_v5=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_5.json",
        data_root="data/video_sample/InternVId-FLT_5",
        media_type="video",
        jump_filter=True
    ),
    internvid_v6=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_6.json",
        data_root="data/video_sample/InternVId-FLT_6",
        media_type="video",
        jump_filter=True
    ),
    internvid_v7=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_7.json",
        data_root="data/video_sample/InternVId-FLT_7",
        media_type="video",
        jump_filter=True
    ),
    internvid_v8=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_8.json",
        data_root="data/video_sample/InternVId-FLT_8",
        media_type="video",
        jump_filter=True
    ),
    internvid_v9=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_9.json",
        data_root="data/video_sample/InternVId-FLT_9",
        media_type="video",
        jump_filter=True
    ),
    internvid_v10=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_10.json",
        data_root="data/video_sample/InternVId-FLT_10",
        media_type="video",
        jump_filter=True
    ),
    internvid_v11=dict(
        anno_path="anno_downstream/InternVid/InternVid-10M-flt_11.json",
        data_root="data/video_sample/InternVId-FLT_11",
        media_type="video",
        jump_filter=True
    )
)

available_corpus["internvid_all"] = [
    available_corpus["internvid_v1"],
    available_corpus["internvid_v2"],
    available_corpus["internvid_v3"],
    available_corpus["internvid_v4"],
    available_corpus["internvid_v5"],
    available_corpus["internvid_v6"],
    available_corpus["internvid_v7"],
    available_corpus["internvid_v8"],
    available_corpus["internvid_v9"],
    available_corpus["internvid_v10"],
    available_corpus["internvid_v11"],
]

# ============== for validation =================
available_corpus["msrvtt_1k_test"] = dict(
    anno_path="anno_downstream/msrvtt_1k_test.json", 
    data_root="./data/MSRVTT_Videos",
    media_type="video"
)

available_corpus["didemo_ret_test"] = dict(
    anno_path="anno_downstream/didemo_ret_test.json",
    data_root="./data/didemo/processed_videos/test",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["anet_ret_val"] = dict(
    anno_path="anno_downstream/anet_ret_val.json",
    data_root="./data/anet_1.3_video_val_resize",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["k400_act_val"] = dict(
    anno_path="anno_downstream/kinetics400_validate.json",
    data_root="./data/kinetics_400_val_10s_320p",
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)

available_corpus["ucf101_act_test"] = dict(
    anno_path="anno_downstream/ucf101_act_test.json",
    data_root="./data/ucf101/videos",
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)


available_corpus["hmdb51_act_test"] = dict(
    anno_path="anno_downstream/hmdb51_act_test.json",
    data_root="./data/hmdb51/videos_test",
    media_type="video",
    prompt="kinetics",
    is_act_rec=True,
)

available_corpus["ssv2_mc_val"] = dict(
    anno_path="anno_downstream/ssv2_mc_val.json",
    data_root="./data/ssv2/20bn-something-something-v2",
    media_type="video",
)