SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# YOLO_PATH="$SCRIPT_DIR/models/yoloe/yolov8l_latest.pt"

# if [ -f "$YOLO_PATH" ]; then
#     echo "Found data file at: $YOLO_PATH"
# else
#     echo "Downloading model weights into $YOLO_PATH"
#     curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt -o ${SCRIPT_DIR}/models/yoloe/yolov8l_latest.pt --create-dirs
#     echo "Model downloaded into $YOLO_PATH"
# fi

pip install -r requirements.txt --no-build-isolation

export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3"
echo $CUDA_HOME
git clone https://github.com/IDEA-Research/GroundingDINO.git "$SCRIPT_DIR/models/dino/GroundingDINO/"
cd $SCRIPT_DIR/models/dino/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cpu.cpp
sed -i 's/value.type().is_cuda()/value.is_cuda()/g' ms_deform_attn.h
grep "value.scalar_type()" ms_deform_attn_cuda.cu | head -n 2
cd $SCRIPT_DIR/models/dino/GroundingDINO/
cp $SCRIPT_DIR/libs/setup_edited_default_scripts/gdino_requirements.txt ./requirements.txt
pip install -v --no-build-isolation -e .
mkdir weights
cd weights
curl -s -L -O https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd $SCRIPT_DIR


