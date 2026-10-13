import requests
from ultralytics import YOLO
import cv2

class AgentHarness:
    def __init__(self):
        # 本地模型API（LM Studio）
        self.llm_url = "http://127.0.0.1:1234/v1/chat/completions"
        # YOLO视觉工具
        self.yolo_model = YOLO("./runs/detect/train/weights/best.pt")
        self.conf = 0.2
        # 对话上下文
        self.messages = []
        # 工具注册表（Pi风格，技能插件池）
        self.tools = {
            "camera_detect": self.tool_camera
        }
        # 系统提示词
        self.system_prompt = """
你是【华小牛】，基于Pi Agent架构搭建的本地智能体Harness。
你拥有工具 camera_detect：调用摄像头，YOLO目标检测，识别画面物体。
规则：
1. 用户询问摄像头画面、镜头前面有什么、画面内容 → 调用 camera_detect
2. 普通文字问题，不需要视觉信息，直接回答，不要调用工具
3. 需要调用工具时，输出固定标记：[[USE_TOOL:camera_detect]]
4. 获取工具返回结果后，结合结果生成最终回答
"""
        self.messages.append({"role": "system", "content": self.system_prompt})

    # 视觉插件：摄像头YOLO检测
    def tool_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "摄像头读取失败"
        res = self.yolo_model(frame, conf=self.conf)
        detect_list = []
        for box in res[0].boxes:
            cls_name = self.yolo_model.names[int(box.cls)]
            conf_val = float(box.conf)
            detect_list.append(f"{cls_name}({conf_val:.2f})")
        cap.release()
        if detect_list:
            return f"摄像头画面检测结果：{', '.join(detect_list)}"
        else:
            return "画面没有识别到目标物体"

    # Harness核心Agent循环（Pi Agent核心）
    def run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        payload = {
            "model": "qwen2.5-1.5b-instruct.gguf",
            "messages": self.messages,
            "temperature":0.6
        }
        try:
            resp = requests.post(self.llm_url, json=payload, timeout=20)
            res_json = resp.json()
            # 增加判断
            if "choices" not in res_json:
                return f"⚠️LLM接口异常，返回内容：{res_json}"
            llm_resp = res_json["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️请求大模型失败，错误：{str(e)}"

        # 判断触发工具
        tag = "[[USE_TOOL:camera_detect]]"
        if tag in llm_resp:
            print(f"\n🤖【Harness】调用工具 camera_detect")
            tool_result = self.tools["camera_detect"]()
            print(f"🔧【工具返回】{tool_result}")
            self.messages.append({"role": "tool", "content": tool_result})
            # 二次请求
            try:
                payload2 = {
                    "model": "qwen2.5-1.5b-instruct.gguf",
                    "messages": self.messages,
                    "temperature":0.6
                }
                resp2 = requests.post(self.llm_url, json=payload2, timeout=20)
                res_json2 = resp2.json()
                if "choices" not in res_json2:
                    return f"⚠️二次请求接口异常：{res_json2}"
                final_ans = res_json2["choices"][0]["message"]["content"]
                self.messages.append({"role":"assistant", "content":final_ans})
                return final_ans
            except Exception as e:
                return f"⚠️二次调用LLM出错：{str(e)}"

        self.messages.append({"role":"assistant", "content":llm_resp})
        return llm_resp

if __name__ == "__main__":
    agent = AgentHarness()
    print("===== 华小牛 Harness (Pi-Agent架构) =====")
    print("本地Agent，YOLO视觉插件。输入 exit 退出\n")
    while True:
        user_msg = input(">>>你：")
        if user_msg.strip() == "exit":
            print("Harness 退出")
            break
        ans = agent.run(user_msg)
        print(f"🤖华小牛：{ans}\n")
