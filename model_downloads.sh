SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

YOLO_PATH="$SCRIPT_DIR/models/yoloe/yolov8l_latest.pt"

if [ -f "$YOLO_PATH" ]; then
    echo "Found data file at: $YOLO_PATH"
else
    echo "Downloading model weights into $YOLO_PATH"
    curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt -o ${SCRIPT_DIR}/models/yoloe/yolov8l_latest.pt --create-dirs
    echo "Model downloaded into $YOLO_PATH"
fi


