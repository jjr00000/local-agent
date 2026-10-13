from ultralytics import YOLO
import requests
import json

# ========== 配置区，改成你自己的 ==========
YOLO_WEIGHT = "runs/detect/train/weights/best.pt"
LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL_NAME = "qwen"
# ========================================

def call_local_llm(object_list):
    prompt = f"""摄像头画面检测到的物体：{object_list}。
你是视觉场景助手，请基于识别出来的物品，给出简短实用的提示或建议，控制回答在两句话以内。
"""
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    resp = requests.post(LLM_API_URL, json=payload)
    res_json = resp.json()
    return res_json["choices"][0]["message"]["content"]

def yolo_camera_loop():
    model = YOLO(YOLO_WEIGHT)
    # 打开摄像头实时推理
    results_stream = model(source=0, stream=True, verbose=False)
    print("🎥【视觉场景智能助手】摄像头已启动，按键盘 q 退出窗口")

    for result in results_stream:
        cls_names = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_names.append(result.names[cls_id])
        # 去重
        cls_names = list(set(cls_names))
        if cls_names:
            print(f"\n🔍 YOLO检测到物体：{cls_names}")
            llm_ans = call_local_llm(cls_names)
            print(f"🤖场景助手建议：{llm_ans}")

if __name__ == "__main__":
    try:
        yolo_camera_loop()
    except KeyboardInterrupt:
        print("\n程序结束")
