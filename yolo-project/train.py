from ultralytics import YOLO

# 加载yolov8n预训练骨架，在coco128数据集上训练
model = YOLO("yolov8n.yaml")

if __name__ == "__main__":
    results = model.train(
        data="coco128.yaml",
        epochs=5,
        imgsz=640,
        device="cpu"
    )
