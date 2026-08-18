# RMBG 本地抠图工具

基于 **BRIA AI 开源的 RMBG-2.0（BiRefNet 架构）** 的本地离线抠图 Web 工具。
AI 自动去背景，**图片全程本地推理、不上传任何云端**，支持发丝级边缘。

## 功能

- 单图 / 批量拖拽上传，自动去背景
- 输出透明 PNG（带 alpha 通道）
- 一键换纯色背景或自定义背景图
- 批量结果打包下载（ZIP）
- CPU 即可运行，单张约 2–5 秒

## 环境要求

- Python 3.11+
- Windows / macOS / Linux

## 快速开始

```bash
# 1. 创建并激活虚拟环境（任选一种）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动（首次会联网下载约 977MB 模型权重并缓存到 models/）
python app.py
```

启动后浏览器自动打开 http://127.0.0.1:7861

## 模型权重说明

仓库**不包含**模型文件（约 977MB）。首次运行 `app.py` 或 `inference.py` 时，
脚本会自动从 ModelScope 下载 `RMBG-2.0` 的 ONNX 权重并缓存到 `models/rmbg-2.0.onnx`。
之后离线可用，无需再次下载。

如需手动下载：

```bash
python -c "import inference; inference.ensure_model()"
```

## 文件结构

| 文件 | 说明 |
|------|------|
| `app.py` | Gradio Web 界面（单图 / 批量） |
| `inference.py` | 推理模块：加载 ONNX、生成 alpha 蒙版、自动下载模型 |
| `test_model.py` | 预处理模式验证脚本（可选） |
| `start.bat` | Windows 一键启动脚本 |
| `requirements.txt` | Python 依赖 |

## 许可

- 代码：MIT
- 模型：RMBG-2.0 由 BRIA AI 发布，采用**非商业使用（Non-Commercial）许可**，
  仅限个人本地使用，请勿用于商业用途。

## 原理简述

RMBG-2.0 采用 BiRefNet（Bilateral Reference Network）架构：
一条图像同时走「高分辨率细节支路」与「低分辨率语义支路」，两路互相参考，
在边缘精修模块（RRM）中逐像素输出 0~1 的 alpha 概率，
使发丝、薄纱等半透明边缘自然过渡，而非硬切。
