# Local-Agent｜创智部二面实战项目
> 人工智能协会创智部二面实战任务，本地大模型 + YOLO视觉多模态智能体

## 📖 项目简介
本项目基于 LM Studio 部署本地 Qwen2.5-7B-Instruct 大模型，通过OpenAI兼容API接入Python CLI应用。
后续接入YOLO目标检测，实现摄像头画面识别，打造本地多模态智能体。

### ✨ 当前已实现功能
1. 本地大模型多轮对话，支持上下文记忆
2. 对话持久化：`save`保存会话 / `load`加载历史会话 / `exit`退出
3. 模块化工程结构，Git版本管理
4. 预留YOLO视觉检测接入接口

## 🧰 环境依赖
Python >=3.10
```bash
pip install -r requirements.txt

python src/main.py
README.md
YOLOv8 实时目标检测项目 · 二面实战任务
本项目为人工智能协会创智部二面实战任务，基于 Ultralytics YOLOv8 完成模型训练 + 摄像头实时推理部署，完整复现深度学习目标检测工程流程。
项目全程使用 Git 版本控制，记录完整开发日志，结构整洁、可一键复现。
📌 项目概述
本项目基于 YOLOv8 轻量级检测模型，使用官方 coco128 小型数据集完成快速训练，生成专属权重文件 best.pt，最终实现本地摄像头实时目标检测。
本项目完整覆盖二面实战要求：
- ✅ 完成 YOLO 实时推理部署（摄像头端）
- ✅ 完整工程化、规范化目录结构
- ✅ Git 版本控制 + 规范提交记录
- ✅ 完整文档记录（README + 工程日志报告）
💻 环境依赖
Python 3.11 + Pytorch CPU + Ultralytics + OpenCV
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python

📁 项目目录结构
local-agent/
├── yolo_detect.py     # 摄像头实时检测主程序
├── report.md          # 项目工程日志、实验记录、踩坑总结
├── README.md          # 项目说明文档（本文件）
├── .gitignore         # 忽略缓存、训练结果、大权重文件
└── runs/              # 自动生成的训练结果（不上传仓库）

🚀 项目运行流程
1. 模型训练
使用 coco128 数据集快速训练 5 轮，生成最优权重 best.pt
训练结果保存路径：runs/detect/train/weights/best.pt
2. 实时摄像头推理
运行摄像头检测程序：
python yolo_detect.py

- 自动打开本地摄像头
- 实时检测画面目标、自动画框
- 终端打印识别物体类别
- 按 q 退出程序
📊 实验结果与分析
实验现象
程序可正常启动、摄像头读取正常、模型推理正常。
由于训练数据集仅 128 张图片，样本量极小，模型泛化能力弱，因此多数场景输出no detections，识别精度有限。
问题复盘（工程日志核心）
- 问题1：模型识别效果差
  - 原因：coco128 数据集样本过少，模型泛化能力不足
  - 解决：保留训练流程，作为标准教学演示流程，不强行追求高精度
- 问题2：官方权重 yolov8n.pt 下载失败
  - 原因：网络无法连接 GitHub 官方资源
  - 解决：放弃在线下载，使用本地训练完成的 best.pt 权重，保证项目可离线运行
改进方案
- 使用更大规模 COCO 数据集训练，提升精度
- 使用官方预训练权重 yolov8n.pt 迁移学习
- 降低置信度阈值，提高检出率
🧠 项目实现思路（面试口述版）
本次任务我选择 YOLOv8 目标检测方向完成二面实战。首先搭建完整 Python 深度学习环境，基于 Ultralytics 框架进行模型训练，使用轻量化 coco128 数据集快速迭代得到模型权重。随后编写摄像头实时推理代码，实现端侧实时检测。
项目过程中我重点关注工程规范性，通过 Git 做版本管理，全程记录工程日志，整理清晰的 README 文档，保证项目可复现、可交接。同时主动分析模型缺陷、网络问题并给出合理取舍，保证项目流程完整、逻辑自洽。
✅ 任务完成度
- YOLO 模型训练 ✅
- 实时推理部署 ✅
- 工程结构规范化 ✅
- Git 版本控制 ✅
- 完整文档与工程日志 ✅
## 项目复现步骤
1. 安装依赖
```bash
pip install -r requirements.txt
python train.py
python yolo_detect.py
## AI 使用说明

本项目使用 AI 作为辅助工具：借助 AI 编写 YOLO 训练、摄像头推理代码，辅助撰写 Markdown 文档、调试 git 命令，梳理项目思路。项目整体规划、代码校验、问题排查均由本人主导完成。

## 项目结构说明

本仓库包含两部分内容：

- 根目录：YOLO 目标检测相关代码与文档（train.py、yolo_detect.py、report.md 等，为本轮二面 YOLO 实战任务）
- src/、backend/、docs/ 文件夹：为本项目内本地大模型相关模块