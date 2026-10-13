# Local-Agent｜人工智能协会创智部二面实战项目
> 本项目为创智部二面实战任务，实现本地大模型CLI对话、YOLO目标检测、Harness智能体调度框架与多模态视觉Agent创意作品。

## 项目简介
项目一共4个子任务：
1. **Task1**：基于LM Studio部署本地大模型，搭建CLI命令行对话程序，支持多轮上下文记忆，会话保存与加载。
2. **Task2**：YOLOv8 摄像头实时目标检测，本地离线图像识别。
3. **Task4**：搭建Harness智能体调度框架，实现大模型的工具调用机制，作为任务调度核心。
4. **Task5**：多模态创意作品，依托Harness框架，让大模型自主调用YOLO视觉工具，读取摄像头画面、理解场景并回答问题。

## 项目目录结构
```

local-agent/
├── harness/            # Task4 Harness 多模态智能体调度主程序
│   └── harness.py
├── task1-llm/          # Task1 本地大模型 CLI 对话代码
│   └── multi_modal_cli.py
├── yolo-project/       # Task2 YOLO 目标检测相关代码、权重文件
│   ├── train.py
│   ├── yolo_detect.py
│   └── yolov8n.pt
├── docs/               # 各任务详细开发日志、踩坑记录、实验报告
│   ├── task1.md
│   ├── task2.md
│   ├── task4.md
│   └── task5.md
├── assets/
├── datasets/
├── examples/
├── runs/
├── src/
├── .gitignore
├── README.md           # 本文件：项目总览与快速复现指南
├── report.md
└── requirements.txt
```
## 环境依赖
Python >= 3.10
一键安装全部依赖包：
```
bash
pip install -r requirements.txt
```

## 复现 & 运行步骤

> 
> 前置准备：打开 LM Studio，加载 GGUF 大模型，启动本地 API 服务（默认地址 `http://localhost:1234/v1`）

### Task1：本地大模型 CLI 对话

```
cd task1-llm
python multi_modal_cli.py
```

### Task2：YOLO 摄像头目标检测

```
cd yolo-project
python yolo_detect.py
```

### Task4 & Task5：Harness 智能体（工具调用 + 多模态联动）

```
cd harness
python harness.py
```

交互指令示例：

- 普通文字聊天：`你好，简单介绍一下自己`
- 调用摄像头视觉识别：`看看摄像头里面有什么物体`
- 退出程序：`exit`

## 开发日志

每个任务详细开发过程、遇到的问题、调试记录、实验分析，全部放在 `docs/` 目录下：

- docs/task1.md 任务 1 完整工程日志
- docs/task2.md 任务 2 完整工程日志
- docs/task4.md 任务 4 harness 智能体框架搭建记录
- docs/task5.md 任务 5 创意作品开发记录