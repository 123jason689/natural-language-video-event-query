import gdown
import os

os.makedirs("./models/mobileviclip/weights", exist_ok=True)

gdown.download("https://drive.google.com/uc?id=1BWioaoo8WYYry_Vw72wI-bnpDUzNWRb5", "./models/mobileviclip/weights/mobileviclip_small.pt")


### MobileViCLIP flash_attn dependency wheel installment 

import torch
import subprocess
import sys

# 1. Detect your environment
torch_ver = torch.__version__.split('+')[0]  # e.g., '2.5.1'
cuda_ver = torch.version.cuda.replace('.', '') # e.g., '121' for 12.1
py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}" # e.g., 'cp310'

# 2. Construct the correct wheel URL for Flash Attention
# We default to version 2.6.3 which is stable for most Colab environments
flash_attn_version = "2.6.3"
base_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{flash_attn_version}"
wheel_filename = f"flash_attn-{flash_attn_version}+cu{cuda_ver}torch{torch_ver}cxx11abiFALSE-{py_ver}-{py_ver}-linux_x86_64.whl"
wheel_url = f"{base_url}/{wheel_filename}"

print(f"Detected: PyTorch {torch_ver}, CUDA {cuda_ver}, Python {py_ver}")
print(f"Downloading and installing: {wheel_filename}...")

# 3. Force install the wheel directly
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url, "--no-build-isolation"])
    print("✅ Flash Attention installed successfully!")
except subprocess.CalledProcessError:
    print("❌ Failed to install the specific wheel. Trying a general fallback...")
    # Fallback: Downgrade PyTorch to 2.4 (Common fix for Colab)
    
    # print("Downgrading PyTorch to 2.4.1 to ensure compatibility...")
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.4.1+cu121", "torchvision==0.19.1+cu121", "--extra-index-url", "https://download.pytorch.org/whl/cu121"])
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"])