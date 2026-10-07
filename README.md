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
