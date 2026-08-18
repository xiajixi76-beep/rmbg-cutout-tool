"""验证 RMBG-2.0 ONNX 的输入预处理模式（raw vs norm）。

下载一张带主体的测试照片，分别用两种预处理得到蒙版，
计算"二分类度"（前景/背景区分是否清晰），并保存可视化图供人工核对。
"""
import os
import io
import urllib.request
import numpy as np
from PIL import Image
import inference


def download_test():
    url = "https://picsum.photos/id/1027/800/800"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = urllib.request.urlopen(req, timeout=30)
    img = Image.open(io.BytesIO(r.read())).convert("RGB")
    path = os.path.join(os.path.dirname(__file__), "test_photo.jpg")
    img.save(path)
    return img, path


def bimodal_score(mask_arr):
    fg = (mask_arr > 230).mean()   # 近前景比例
    bg = (mask_arr < 25).mean()    # 近背景比例
    return fg + bg


def main():
    os.makedirs("test_output", exist_ok=True)
    img, path = download_test()
    print("测试图:", path, img.size)

    results = {}
    for mode in ["raw", "norm"]:
        mask = np.array(get_alpha_vis(img, mode))
        score = bimodal_score(mask)
        # 可视化：黑底白前景
        vis = Image.fromarray(((mask > 128) * 255).astype(np.uint8)).convert("L")
        vis_path = f"test_output/mask_{mode}.png"
        vis.save(vis_path)
        # 合成：背景置灰，前景保留
        arr = np.array(img)
        gray = arr.mean(axis=2).astype(np.uint8)
        comp = arr.copy()
        for c in range(3):
            comp[:, :, c] = np.where(mask > 128, arr[:, :, c], gray)
        comp = comp.astype(np.uint8)
        comp_path = f"test_output/composite_{mode}.png"
        Image.fromarray(comp).save(comp_path)
        results[mode] = score
        print(f"mode={mode:5s} bimodal_score={score:.3f}  -> {vis_path}, {comp_path}")

    best = max(results, key=results.get)
    print("\n推荐模式:", best, "（如两图都明显割裂则人工看图确认）")
    return best


def get_alpha_vis(img, mode):
    return np.array(inference.get_alpha(img, mode))


if __name__ == "__main__":
    main()
