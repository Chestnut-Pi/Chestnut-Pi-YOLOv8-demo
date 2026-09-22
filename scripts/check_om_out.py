"""辅助：确认 OM（或 ONNX / ais_bench）输出的数据布局，避免后处理按错误的轴读取。

用法:
    python3 scripts/check_om_out.py --npy logs/om_raw_output.npy
    python3 scripts/check_om_out.py --file out/xxx.bin
    python3 scripts/check_om_out.py                    # 默认扫 out/**/*.bin
"""
import argparse
import glob
import os

import numpy as np


def report(arr, tag):
    print(f"\n--- {tag} ---")
    print(f"元素总数 : {arr.size}  dtype={arr.dtype}")
    hit = False
    for shape in [(1, 84, 8400), (1, 8400, 84)]:
        if int(np.prod(shape)) == arr.size:
            print(f"可 reshape 为 {shape}")
            a = arr.reshape(shape)
            print(f"  min/max = {a.min():.4f} / {a.max():.4f}   "
                  f"NaN={int(np.isnan(a).sum())} Inf={int(np.isinf(a).sum())}")
            hit = True
    if not hit:
        print("元素数不等于 84x8400，请确认：")
        print('  1) ATC 是否带了 --input_shape="images:1,3,640,640"')
        print("  2) 模型是否为 YOLOv8n（84 = 4 框坐标 + 80 类）")
    print("前 10 个值:", np.round(arr[:10].astype(np.float64), 4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="指定一个 .bin 文件")
    ap.add_argument("--npy", default=None, help="指定一个 .npy 文件")
    ap.add_argument("--glob", default="out/**/*.bin")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    dt = np.float32 if args.dtype == "float32" else np.float16

    if args.npy:
        report(np.load(args.npy), args.npy)
        return

    files = [args.file] if args.file else sorted(glob.glob(args.glob, recursive=True))
    print(f"找到输出文件: {len(files)} 个")
    if not files:
        print(f"（在 {args.glob} 下没有找到 .bin；先跑 ais_bench 或 infer_om.py）")
        return
    for f in files[:5]:
        report(np.fromfile(f, dtype=dt), f)


if __name__ == "__main__":
    main()
