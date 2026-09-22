"""步骤 2：把 YOLOv8n 的 PyTorch 权重导出为 ONNX（静态 shape，适配昇腾 ATC）。

为什么必须是静态 shape：
    ATC 在编译期就要确定每个张量的形状，动态轴（dynamic axes）会直接转换失败。
    所以这里固定 dynamic=False。

为什么默认 FP32：
    是否用 FP16 交给 ATC 阶段决定。310B 的 AI Core 本身按 FP16 计算，
    在 ATC 里指定 --input_fp16_nodes 反而会插入额外的精度转换算子，实测更慢
    （见工程文档 6.4 节）。

用法:
    python scripts/export_onnx.py                    # 默认 opset=11
    python scripts/export_onnx.py --opset 12         # 换 opset 重试
    python scripts/export_onnx.py --no-simplify      # 关闭 onnxsim
"""
import argparse
import os
import sys

import onnx
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DTYPE_NAME = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 10: "float16"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="model/yolov8n.pt", help="输入 PyTorch 权重")
    ap.add_argument("--onnx", default=None, help="输出 ONNX 路径，默认与权重同名")
    ap.add_argument("--imgsz", type=int, default=640, help="导出输入分辨率")
    ap.add_argument("--opset", type=int, default=11,
                    help="ONNX opset 版本；opset 11 在 CANN 7.0.RC1 上实测可用")
    ap.add_argument("--simplify", action="store_true", default=True, help="用 onnxslim/onnxsim 化简（默认开）")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false")
    ap.add_argument("--force", action="store_true", help="已存在输出文件时也重新导出（默认跳过）")
    return ap.parse_args()


def describe(path):
    """打印 ONNX 的关键信息，便于和 ATC 参数逐项核对。"""
    m = onnx.load(path)
    onnx.checker.check_model(m)
    print("\n=== ONNX 模型自检 ===")
    print("ir_version   :", m.ir_version)
    print("opset        :", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    print("producer     :", m.producer_name, m.producer_version)
    for tag, coll in (("输入", m.graph.input), ("输出", m.graph.output)):
        for v in coll:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
            dt = DTYPE_NAME.get(v.type.tensor_type.elem_type, v.type.tensor_type.elem_type)
            print(f"{tag}: name={v.name!r} shape={dims} dtype={dt}")
    print("算子总数     :", len(m.graph.node))
    ops = sorted({n.op_type for n in m.graph.node})
    print("算子种类     :", len(ops))
    print("算子列表     :", ", ".join(ops))


def main():
    args = parse_args()
    onnx_path = args.onnx or os.path.splitext(args.pt)[0] + ".onnx"

    if not os.path.exists(args.pt):
        raise SystemExit(f"找不到权重文件: {args.pt}，请先运行 scripts/run_all.py 下载 yolov8n.pt")

    if os.path.exists(onnx_path) and not args.force:
        print(f"已存在，跳过导出: {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
        print("（需要重新导出请加 --force）")
        describe(onnx_path)
        return

    model = YOLO(args.pt)

    # 关键参数：
    #   dynamic=False 静态 shape —— 昇腾 ATC 编译期必须确定 shape，动态轴会直接失败
    #   half=False    导出 FP32；是否用 FP16 在 ATC 阶段用 --input_fp16_nodes 决定
    #   simplify      先用 onnxsim 消掉冗余算子，减少 ATC 遇到不支持算子的概率
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        half=False,
    )

    # ultralytics 的返回值可能是 str，也可能是 list；统一成字符串
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, str) and os.path.abspath(out) != os.path.abspath(onnx_path):
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        os.replace(out, onnx_path)

    print(f"导出完成: {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
    describe(onnx_path)


if __name__ == "__main__":
    main()
