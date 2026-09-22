"""步骤 3：用 ONNX Runtime 跑一遍 ONNX，确认图和数值都正常。

这一层的意义：把「模型本身的错」和「昇腾转换的错」分开。
ONNX 在这里跑不通，就不用去动 ATC 了。

用法:
    python scripts/check_onnx.py --onnx model/yolov8n.onnx --image data/images/bus.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_common import load_names, postprocess, preprocess, print_dets, read_image  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model/yolov8n.onnx")
    ap.add_argument("--image", default="data/images/bus.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save", default="logs/onnx_raw.npy")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"找不到 ONNX 文件: {args.onnx}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("=== ONNX Runtime 会话 ===")
    print("provider :", sess.get_providers()[0])
    print("输入     :", inp.name, inp.shape, inp.type)
    for o in sess.get_outputs():
        print("输出     :", o.name, o.shape, o.type)

    # 1) 确定性随机输入，只看形状和数值范围是否健康
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    y = sess.run(None, {inp.name: x})[0]
    print("\n=== 随机输入自检 ===")
    print("输出 shape  :", y.shape, "dtype:", y.dtype)
    print("输出 min/max:", round(float(y.min()), 4), "/", round(float(y.max()), 4))
    print("NaN/Inf     :", int(np.isnan(y).sum()), "/", int(np.isinf(y).sum()))
    assert y.shape == (1, 84, 8400), f"输出形状异常: {y.shape}，期望 (1, 84, 8400)"

    # 2) 真实图片，出检测框
    img = read_image(args.image)
    h0, w0 = img.shape[:2]
    blob, r, dw, dh = preprocess(img)
    pred = sess.run(None, {inp.name: blob})[0]
    print("\n=== 真实图片推理 ===")
    print(f"原图 {w0}x{h0}  letterbox ratio={r:.6f} pad=({dw},{dh})")
    print("输出 min/max:", round(float(pred.min()), 4), "/", round(float(pred.max()), 4))

    load_names()
    dets = postprocess(pred, r, dw, dh, w0, h0, args.conf, args.iou)
    print_dets(dets, tag="")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    np.save(args.save, pred)
    print(f"\n原始输出已存 {args.save}（供精度对比用）")
    print("【正常】ONNX 可被 ONNX Runtime 正常加载并推理，可以进入 ATC 转换")


if __name__ == "__main__":
    main()
