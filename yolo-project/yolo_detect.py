from ultralytics import YOLO
import cv2

# 加载官方预训练模型（自动下载 yolov8n.pt）
model = YOLO("yolov8n.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)
# 调低置信度阈值，更容易检出物体，面试演示首选conf=0.2
CONF_THRESHOLD = 0.2

print("=== YOLO实时目标检测 面试演示程序 ===")
print("操作：画面出现框代表识别成功，按 q 关闭窗口退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

    # YOLO推理
    results = model(frame, conf=CONF_THRESHOLD)

    # 在画面绘制检测框+标签
    annotated_frame = results[0].plot()

    # 控制台打印识别到的物体（面试官看终端输出加分）
    detected_objects = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        detected_objects.append(f"{cls_name}({conf:.2f})")
    if detected_objects:
        print(f"检测到：{detected_objects}")

    cv2.imshow("YOLO 目标检测演示", annotated_frame)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
