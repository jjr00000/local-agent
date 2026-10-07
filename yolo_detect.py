from ultralytics import YOLO
import cv2

def camera_detect():
    # 使用刚才训练完成的本地模型，不需要联网下载
    model = YOLO("runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败！")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            break
        result = model(frame)
        annotated = result[0].plot()
        cv2.imshow("YOLOv8 Camera Detect", annotated)

        obj_list = get_detected_objects(result)
        if obj_list:
            print("识别到物体：", obj_list)

        # 按q退出摄像头窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_detected_objects(result):
    boxes = result[0].boxes
    names = result[0].names
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    objects = [names[c] for c in cls_ids]
    return list(set(objects))

if __name__ == "__main__":
    camera_detect()
