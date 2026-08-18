"""RMBG-2.0 本地抠图 Web 界面（Gradio）

功能：
- 单图 / 批量上传，自动去除背景
- 输出透明 PNG；也可一键换纯色背景或自定义背景图
- 全部本地推理，图片不出本机
"""
import os
import io
import zipfile
import tempfile
from PIL import Image
import gradio as gr

import inference
from inference import get_alpha, apply_background

BG_MODES = ["透明背景", "纯色背景", "自定义背景图"]


def process_one(img, bg_mode, bg_color, bg_image):
    if img is None:
        return None
    img = img.convert("RGB")
    mask = get_alpha(img, inference.MODE)
    rgba = img.copy().convert("RGBA")
    rgba.putalpha(mask)
    return apply_background(rgba, bg_mode, bg_color, bg_image)


def single_run(img, bg_mode, bg_color, bg_image):
    return process_one(img, bg_mode, bg_color, bg_image)


def batch_run(files, bg_mode, bg_color, bg_image):
    if not files:
        return [], None
    outs = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_PNG) as zf:
        for i, f in enumerate(files):
            try:
                img = Image.open(f.name if hasattr(f, "name") else f).convert("RGB")
            except Exception:
                continue
            mask = get_alpha(img, inference.MODE)
            rgba = img.copy().convert("RGBA")
            rgba.putalpha(mask)
            out = apply_background(rgba, bg_mode, bg_color, bg_image)
            outs.append(out)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            zf.writestr(f"cutout_{i:03d}.png", buf.getvalue())
    zip_path = os.path.join(tempfile.gettempdir(), "cutouts.zip")
    with open(zip_path, "wb") as fp:
        fp.write(zip_buf.getvalue())
    return outs, zip_path


with gr.Blocks(title="本地抠图工具 · RMBG-2.0") as demo:
    gr.Markdown("# 🪄 本地抠图工具（RMBG-2.0）\n"
                "AI 自动去背景，**图片全程本地处理、不上传云端**。支持发丝级边缘。")

    with gr.Tabs():
        with gr.Tab("单图"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="上传图片", type="pil", sources=["upload", "clipboard"], height=420)
                    bg_mode = gr.Radio(BG_MODES, value="透明背景", label="输出背景")
                    bg_color = gr.ColorPicker(label="背景色（选纯色时生效）", value="#ffffff")
                    bg_image = gr.Image(label="背景图（选自定义时生效）", type="pil", height=200)
                    run_btn = gr.Button("✂️ 开始抠图", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="结果（透明 PNG 可下载）", type="pil", height=420,
                                       image_mode="RGBA")
            run_btn.click(single_run, [img_in, bg_mode, bg_color, bg_image], img_out)

        with gr.Tab("批量"):
            with gr.Row():
                with gr.Column():
                    files_in = gr.File(label="上传多张图片", file_count="multiple", type="filepath")
                    bg_mode2 = gr.Radio(BG_MODES, value="透明背景", label="输出背景")
                    bg_color2 = gr.ColorPicker(label="背景色", value="#ffffff")
                    bg_image2 = gr.Image(label="背景图", type="pil", height=200)
                    run_btn2 = gr.Button("✂️ 批量抠图", variant="primary")
                with gr.Column():
                    gal_out = gr.Gallery(label="结果预览", type="pil", height=420)
                    zip_out = gr.File(label="下载全部（ZIP）")
            run_btn2.click(batch_run, [files_in, bg_mode2, bg_color2, bg_image2], [gal_out, zip_out])

    gr.Markdown("--- \n"
                "提示：透明背景导出为 PNG（带 alpha 通道），可直接拖进 PS / 剪映 / 海报工具。\n"
                "本工具基于 BRIA AI 开源模型 RMBG-2.0（非商业使用许可），仅供个人本地使用。")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True, theme=gr.themes.Soft())
