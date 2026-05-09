from ultralytics import YOLO

# Load base model
model = YOLO("yolov8s.pt")

# Train model
model.train(
    data="classroom.yaml",
    epochs=70,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,

    workers=4,
    device=0
)