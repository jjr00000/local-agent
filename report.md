# YOLOv8 目标检测项目实验报告
## 一、项目概述
本项目基于YOLOv8模型完成目标检测任务，使用coco128小型数据集进行模型训练，加载训练得到的权重文件，调用电脑摄像头实现实时画面目标检测。完整流程包含数据集加载、模型训练、权重保存、摄像头实时推理。

## 二、实验环境
- Python
- Ultralytics YOLOv8
- OpenCV
- 本地摄像头

## 三、项目流程
1. 下载coco128数据集，使用YOLOv8模型进行训练；
2. 训练完成，生成权重文件`best.pt`，存放于`runs/detect/train/weights/`目录；
3. 编写Python代码，读取本地摄像头画面，加载`best.pt`权重做实时推理；
4. 将检测结果绘制检测框展示在画面窗口，并在终端输出识别到的物体名称。

## 四、实验结果
程序成功运行，摄像头可以正常打开并读取画面，模型持续对每一帧图像进行推理。
在测试过程中，大部分场景输出`no detections`，**识别效果较差**，很难检出目标物体。

## 五、结果分析
原因：本次训练使用的`coco128`数据集仅包含128张图片，训练样本数量过少，模型泛化能力弱，模型精度低，因此在摄像头实时检测时很难识别画面中的物体。

## 六、改进方案
1. 使用更大规模的数据集进行训练，增加样本数量，提升模型检测精度；
2. 直接使用YOLOv8官方预训练权重`yolov8n.pt`，该模型在海量COCO数据集上预训练，识别效果更好；
3. 调小置信度阈值，降低识别门槛，更容易检出目标（代价：会产生更多误检）。

## 七、核心代码
```python
from ultralytics import YOLO
import cv2

def camera_detect():
    # 加载本地训练好的模型
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

"## 八、操作说明

1. 运行脚本：`python yolo_detect.py`
2. 程序启动后自动打开摄像头，实时检测画面内物体；
3. 退出方式：选中摄像头窗口，按下键盘`q`；或者在终端按`Ctrl+C`强制终止程序。

"## 九、工程日志（踩坑记录）

1. 卡点 1：coco128 数据集样本太少，模型识别效果差，大量输出 no detections
   - 解决思路：保留完整训练流程，作为演示工程，不强行追求高精度。
2. 卡点 2：尝试加载官方 yolov8n.pt 权重时，github 网络连接失败无法下载
   - 解决思路：改用本地训练得到的 best.pt，项目可以离线运行，规避网络问题。