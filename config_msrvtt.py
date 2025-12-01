from libs.configs.data import *
from libs.configs.model import *

# ========================= data ==========================
train_corpus = "msrvtt_1k_test" 
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])
test_types = ["ret_test"]
num_workers = 12

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 256
batch_size_test = 64 
max_txt_l = 32

inputs = dict(
    image_res=256, 
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    model_cls="MobileViCLIP_Small", 
    vision_encoder=dict(
        name="mobileclip_s2",
        img_size=256, 
        head_drop_path_rate=0.,
        attn_pool_num_heads=16,
        clip_embed_dim=512, 
        align_dim=512,
    ),
    text_encoder=dict(
        name="mobileclip_s2"
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=False,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=False,
    vision_ckpt_path="checkpoints/mobileclip_s2.pt", 
    load_vision_ckpt_from_internvideo2_stage2=False,
    text_ckpt_path="checkpoints/mobileclip_s2.pt",
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=1) 

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = False #True
use_bf16 = False #True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2_CLIP",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = "evaluation_mobileviclip_small_with_msrvtt_9k"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
auto_resume = True 
pretrained_path = "./checkpoints/mobileviclip_small.pt" 

deepspeed = dict(
    enable=False,
    stage=0,
)