"""全链路一键脚本：PyTorch(.pt) -> ONNX(.onnx) -> 昇腾离线模型(.om)。

它把后面所有环节串起来，每一步都会先检查产物是否已存在，已存在则跳过（可用
--force 强制重跑）：

    1. 下载/校验  yolov8n.pt
    2. 导出        yolov8n.onnx   (ultralytics export, 静态 shape)
    3. 校验 ONNX   结构 + ONNX Runtime 推理
    4. ATC 转换    yolov8n_bs1.om (调用 atc 命令)
    5. 校验 OM     文件头/大小合法性

用法:
    python scripts/run_all.py                     # 走完整条链路
    python scripts/run_all.py --force             # 全部重跑
    python scripts/run_all.py --skip-onnx         # 已有 onnx，只跑 ATC
    python scripts/run_all.py --only export       # 只跑某一步

【重要】ATC 转换必须在已 source CANN 环境变量的 shell 中执行：

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def log(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def run(cmd, check=True):
    """跑子进程并把输出直接透传到终端（不捕获，避免沙箱下管道 EPERM）。"""
    print("$ " + " ".join(cmd), flush=True)
    ret = subprocess.call(cmd)
    if check and ret != 0:
        raise SystemExit(f"命令失败（exit={ret}）: {' '.join(cmd)}")
    return ret


def exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


# --------------------------------------------------------------------------
def step_pt(args):
    """步骤 1：准备 yolov8n.pt。"""
    log("步骤 1/5  准备 PyTorch 权重 yolov8n.pt")
    if exists(args.pt) and not args.force:
        print(f"已存在，跳过: {args.pt}  ({os.path.getsize(args.pt) / 1e6:.2f} MB)")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.pt)), exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset
    print(f"下载: {PT_URL}")
    attempt_download_asset(args.pt)
    if not exists(args.pt):
        raise SystemExit("下载 yolov8n.pt 失败，请检查网络，或手动下载后放到 model/ 目录")
    print(f"完成: {args.pt}  ({os.path.getsize(args.pt) / 1e6:.2f} MB)")


def step_export(args):
    """步骤 2：pt -> onnx。"""
    log("步骤 2/5  导出 ONNX（静态 shape，适配 ATC）")
    if exists(args.onnx) and not args.force:
        print(f"已存在，跳过: {args.onnx}  ({os.path.getsize(args.onnx) / 1e6:.2f} MB)")
        return
    run([sys.executable, os.path.join(HERE, "export_onnx.py"),
         "--pt", args.pt, "--onnx", args.onnx,
         "--imgsz", str(args.imgsz), "--opset", str(args.opset)])


def step_check_onnx(args):
    """步骤 3：ONNX 结构与数值自检。"""
    log("步骤 3/5  校验 ONNX（结构 + ONNX Runtime 推理）")
    run([sys.executable, os.path.join(HERE, "check_onnx.py"),
         "--onnx", args.onnx, "--image", args.image])


def step_atc(args):
    """步骤 4：onnx -> om，调用昇腾 ATC。"""
    log("步骤 4/5  ATC 转换为 OM 离线模型")
    if exists(args.om) and not args.force:
        print(f"已存在，跳过: {args.om}  ({os.path.getsize(args.om)} 字节)")
        return
    run([sys.executable, os.path.join(HERE, "atc_convert.py"),
         "--onnx", args.onnx, "--out", os.path.splitext(args.om)[0],
         "--soc-version", args.soc_version, "--imgsz", str(args.imgsz)])


def step_check_om(args):
    """步骤 5：OM 产物合法性检查。"""
    log("步骤 5/5  校验 OM 产物")
    if not exists(args.om):
        raise SystemExit(f"未找到 OM 文件: {args.om}")
    size = os.path.getsize(args.om)
    with open(args.om, "rb") as f:
        head = f.read(8)
    print(f"文件      : {args.om}")
    print(f"大小      : {size} 字节 ({size / 1e6:.2f} MB)")
    print(f"文件头    : {head!r}")
    if size < 1024 * 1024:
        raise SystemExit("OM 文件过小，转换很可能没有真正成功")
    print("【正常】OM 产物非空且大小合理")


# --------------------------------------------------------------------------
STEPS = {
    "pt": step_pt,
    "export": step_export,
    "check-onnx": step_check_onnx,
    "atc": step_atc,
    "check-om": step_check_om,
}
ORDER = ["pt", "export", "check-onnx", "atc", "check-om"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(ROOT, "model", "yolov8n.pt"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "model", "yolov8n.onnx"))
    ap.add_argument("--om", default=os.path.join(ROOT, "model", "yolov8n_bs1.om"))
    ap.add_argument("--image", default=os.path.join(ROOT, "data", "images", "sample.jpg"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=11,
                    help="opset 11 是在 CANN 7.0.RC1 上实测可用的版本")
    ap.add_argument("--soc-version", default="Ascend310B1")
    ap.add_argument("--force", action="store_true", help="强制重跑所有步骤")
    ap.add_argument("--skip-onnx", action="store_true", help="跳过 ONNX 校验步骤")
    ap.add_argument("--only", choices=ORDER, default=None, help="只跑指定的一步")
    args = ap.parse_args()

    t0 = time.time()
    todo = [args.only] if args.only else list(ORDER)
    if args.skip_onnx and "check-onnx" in todo:
        todo.remove("check-onnx")

    for name in todo:
        STEPS[name](args)

    log(f"全链路完成，总耗时 {time.time() - t0:.1f} 秒")
    print(f"  PT   : {args.pt}")
    print(f"  ONNX : {args.onnx}")
    print(f"  OM   : {args.om}")
    print("\n下一步（在开发板上跑 OM 推理）:")
    print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
    print("  python3 scripts/infer_om.py --model model/yolov8n_bs1.om "
          "--image data/images/sample.jpg")


if __name__ == "__main__":
    main()
