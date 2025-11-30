from setuptools import setup, find_packages

setup(
    name="mobileviclip",
    version="0.1.0",
    description="MobileViCLIP: An Efficient Video-Text Model for Mobile Devices",
    author="MCG-NJU",
    url="https://github.com/mcg-nju/mobileviclip",
    # This automatically finds 'models', 'utils', 'dataset' as packages
    # Standard package discovery - looks for folders with __init__.py
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "timm==0.5.4",           # Specific version often required for reproducibility
        "transformers>=4.28.1",
        "einops",
        "easydict",
        "open_clip_torch",       # For 'import open_clip'
        "spikingjelly",          # Found in mobileone.py
        "peft",                  # Found in internvideo2_clip_text.py
        "scipy",
        "pandas",
        "ftfy",
        "regex",
        "decord",
        "opencv-python",
        # Optional dependencies found in the codebase:
        "flash-attn",          # Uncomment if GPU supports it (required for InternVideo2)
        # "deepspeed",           # Uncomment for distributed training
        "coremltools",         # Found in convert_model.py
        "onnx",                # Found in convert_model.py
        "spikingjelly"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)