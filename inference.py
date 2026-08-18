"""RMBG-2.0 本地抠图 - 推理模块（共享给 Web 界面与测试脚本）

模型：BRIA AI 开源的 RMBG-2.0（BiRefNet 架构），ONNX 格式，CPU 可跑。
所有计算在本地完成，图片不出本机。
"""
import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rmbg-2.0.onnx")

# 模型权重不入库（约 977MB），首次运行时自动下载并缓存到本地 models/
MODEL_URL = "https://modelscope.cn/models/AI-ModelScope/RMBG-2.0/resolve/master/onnx/model.onnx"

# 默认预处理模式，验证脚本会据此确认/切换
# "raw"  = 仅 /255，RGB，NCHW（BRIA 官方 Space 用法）
# "norm" = (x/255 - mean)/std，RGB，NCHW（torchvision 标准用法）
MODE = "norm"

_TARGET = 1024  # 模型训练分辨率
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None
_input_name = None


def ensure_model():
    """模型缺失时自动从 ModelScope 下载并缓存到 models/。"""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp = MODEL_PATH + ".tmp"
    print(f"[模型] 未找到本地权重，正在从 ModelScope 下载 RMBG-2.0 ONNX（约 977MB）...")
    print(f"[模型] 来源: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    print(f"[模型] 下载完成: {MODEL_PATH}")


def load_session():
    global _session, _input_name
    if _session is None:
        ensure_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=so)
        _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def get_alpha(image, mode=MODE):
    """返回与原图同尺寸的 PIL 'L' 蒙版（255=前景，0=背景）。"""
    sess, in_name = load_session()
    w, h = image.size
    img = image.convert("RGB").resize((_TARGET, _TARGET), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    if mode == "norm":
        arr = (arr / 255.0 - _MEAN) / _STD
    else:
        arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))      # HWC -> CHW
    arr = np.expand_dims(arr, 0)            # -> NCHW
    out = sess.run(None, {in_name: arr})[0][0, 0]  # HW
    # 若输出为 logits（超出 [0,1]），做 sigmoid
    if out.max() > 1.0 or out.min() < 0.0:
        out = 1.0 / (1.0 + np.exp(-out))
    mask = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    mask = Image.fromarray(mask, "L").resize((w, h), Image.BILINEAR)
    return mask


def apply_background(rgba, bg_mode, bg_color, bg_image):
    """根据背景模式合成最终图，返回 PIL RGBA。"""
    if bg_mode == "透明背景" or bg_image is None and bg_mode == "自定义背景图":
        return rgba
    if bg_mode == "纯色背景":
        from PIL import ImageColor
        c = ImageColor.getrgb(bg_color)
        bg = Image.new("RGBA", rgba.size, (c[0], c[1], c[2], 255))
        return Image.alpha_composite(bg, rgba)
    if bg_mode == "自定义背景图" and bg_image is not None:
        bg = bg_image.convert("RGBA").resize(rgba.size, Image.BILINEAR)
        return Image.alpha_composite(bg, rgba)
    return rgba


if __name__ == "__main__":
    s, n = load_session()
    print("ONNX session loaded. input:", n)
