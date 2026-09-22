"""辅助：PyTorch 基线推理 —— 后续 ONNX / OM 的精度都以它为基准。

用法:
    python3 scripts/pt_baseline.py --image data/images/sample.jpg
"""
import argparse
import os
import sys

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/yolov8n.pt")
    ap.add_argument("--image", default="data/images/sample.jpg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save", default="logs/pt_baseline.npz")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    results = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device="cpu",         # 310B 上没有 CUDA，必须显式用 CPU
        save=True,
        # 注意：ultralytics 会把相对路径的 project 拼到它自己的 runs_dir 下
        # （结果是 runs/detect/results/...）。用绝对路径才能落到工程的 results/。
        project=os.path.abspath("results"),
        name="pt_result",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]
    n = 0 if r.boxes is None else len(r.boxes)
    print(f"=== PyTorch 基线推理 ===\n图片: {args.image}\n输入: {args.imgsz}  "
          f"conf={args.conf} iou={args.iou}\n检测框数量: {n}")

    if r.boxes is not None and n > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        names = [r.names[int(k)] for k in cls]
        order = np.argsort(-conf)
        for i in order:
            print(f"  {names[i]:<12s} conf={conf[i]:.4f} box={xyxy[i].round(2).tolist()}")

        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        np.savez(args.save, xyxy=xyxy, conf=conf, cls=cls, names=np.array(names))
        print(f"\n原始数值已存 {args.save}")
    else:
        print("  未检测到目标，检查 conf 阈值或图片内容")

    print(f"可视化图: results/pt_result/{os.path.basename(args.image)}")


if __name__ == "__main__":
    main()
