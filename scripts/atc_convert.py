"""步骤 4：调用昇腾 ATC，把 ONNX 编译成 .om 离线模型。

前置条件（缺一不可，否则报 E10001 / E10054 之类的迷惑错误）：
    1. 已安装 CANN Toolkit（本工程用的是 CANN 8.0.RC1 x86_64）
    2. 已执行  source /usr/local/Ascend/ascend-toolkit/set_env.sh
       —— ATC 靠环境变量推断 host_env_os / host_env_cpu，没 source 会报
          「host_env_os 参数无效」，而命令里根本没有这个参数
    3. TBE 的 Python 依赖齐全（scipy / decorator / attrs / psutil / synr 等）

用法:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python scripts/atc_convert.py --onnx model/yolov8n.onnx --out model/yolov8n_bs1

等价的原生命令（本脚本就是拼出它）:
    atc --model=model/yolov8n.onnx \\
        --framework=5 \\
        --output=model/yolov8n_bs1 \\
        --input_format=NCHW \\
        --input_shape="images:1,3,640,640" \\
        --soc_version=Ascend310B1 \\
        --log=error
"""
import argparse
import os
import shutil
import subprocess
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model/yolov8n.onnx")
    ap.add_argument("--out", default="model/yolov8n_bs1", help="输出前缀，ATC 会自动补 .om")
    ap.add_argument("--soc-version", default="Ascend310B1",
                    help="目标芯片型号；栗子派 Atlas 200I DK A2 为 Ascend310B1")
    ap.add_argument("--input-name", default="images", help="ONNX 输入节点名，必须完全一致")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--framework", type=int, default=5, help="5 = ONNX")
    ap.add_argument("--input-format", default="NCHW")
    ap.add_argument("--log", default="error")
    ap.add_argument("--atc", default=None, help="atc 可执行文件路径，默认从 PATH 里找")
    ap.add_argument("--keep-kernel-meta", action="store_true",
                    help="保留 kernel_meta 中间目录（排查编译问题用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    return ap.parse_args()


def check_env(explicit_atc):
    """确认 atc 可用，并给出可操作的报错。返回 (atc_path, env)。"""
    env = os.environ.copy()

    # atc.bin 链接了 libascend_hal.so，这个库正常情况下由 NPU 驱动包提供。
    # 在**没装 NPU 驱动**的机器（如本 WSL2）上，toolkit 自带一份 stub 版本，
    # 必须把 stub 目录加进库搜索路径，否则只报一句
    #   "error while loading shared libraries: libascend_hal.so"
    #
    # 这里【前置】stub 是安全的：环境变量只注入给 atc 这个子进程，
    # 不会影响当前 shell 后续 import acl。
    # 但**不要**把这种写法搬去改 ~/.bashrc —— 全局前置会让 stub 里的
    # 空壳 libascendcl.so 抢先加载，导致 pyACL 报
    #   ImportError: acl.so: undefined symbol: aclprofSetConfig
    home = env.get("ASCEND_HOME_PATH") or env.get("ASCEND_TOOLKIT_HOME") or ""
    stub = os.path.join(home, "runtime", "lib64", "stub") if home else ""
    if stub and os.path.isdir(stub):
        cur = env.get("LD_LIBRARY_PATH", "")
        if stub not in cur.split(":"):
            env["LD_LIBRARY_PATH"] = f"{stub}:{cur}" if cur else stub

    atc = explicit_atc or shutil.which("atc")
    if atc:
        return atc, env

    print("【异常】在 PATH 中找不到 atc 命令。", file=sys.stderr)
    if home:
        print(f"  检测到 ASCEND_HOME_PATH={home}，但 atc 不在 PATH 中。", file=sys.stderr)
        cand = os.path.join(home, "bin", "atc")
        if os.path.exists(cand):
            print(f"  直接使用: {cand}", file=sys.stderr)
            return cand, env
    else:
        print("  没有检测到 ASCEND_HOME_PATH，说明 CANN 环境变量没配置。", file=sys.stderr)
    print("  处理办法：source /usr/local/Ascend/ascend-toolkit/set_env.sh", file=sys.stderr)
    print("  若 CANN 未安装，请先安装 Ascend-cann-toolkit（见 README 第 3 节）。", file=sys.stderr)
    raise SystemExit(2)


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"找不到 ONNX 文件: {args.onnx}")

    atc, child_env = check_env(args.atc)

    out_prefix = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    om_path = out_prefix + ".om"

    cmd = [
        atc,
        f"--model={args.onnx}",
        f"--framework={args.framework}",
        f"--output={out_prefix}",
        f"--input_format={args.input_format}",
        f"--input_shape={args.input_name}:1,3,{args.imgsz},{args.imgsz}",
        f"--soc_version={args.soc_version}",
        f"--log={args.log}",
    ]

    print("=== ATC 转换参数 ===")
    print(f"atc 路径     : {atc}")
    print(f"输入 ONNX    : {args.onnx}  ({os.path.getsize(args.onnx) / 1e6:.2f} MB)")
    print(f"输出 OM      : {om_path}")
    print(f"目标芯片     : {args.soc_version}")
    print(f"输入节点     : {args.input_name}:1,3,{args.imgsz},{args.imgsz}")
    print(f"ASCEND_HOME  : {os.environ.get('ASCEND_HOME_PATH', '(未设置)')}")
    print("\n完整命令:\n  " + " \\\n  ".join(cmd))

    if args.dry_run:
        print("\n[--dry-run] 未执行")
        return

    # kernel_meta 是 TBE 的算子编译缓存。残留的旧缓存会干扰新的编译，
    # 尤其在换过 soc_version 或改过输入 shape 之后，必须先删干净。
    if os.path.isdir("kernel_meta"):
        print("\n清理 kernel_meta 缓存目录")
        shutil.rmtree("kernel_meta", ignore_errors=True)

    print("\n开始转换。yolov8n 在 4 核 ARM CPU 上约需 10 分钟（16 核 x86 约 30 秒），请勿中断……\n",
          flush=True)
    t0 = time.time()
    ret = subprocess.call(cmd, env=child_env)
    cost = time.time() - t0

    if not args.keep_kernel_meta and os.path.isdir("kernel_meta"):
        shutil.rmtree("kernel_meta", ignore_errors=True)

    print(f"\nATC 退出码: {ret}   耗时: {cost:.1f} 秒")

    if ret != 0:
        print("\n【异常】ATC 转换失败。常见原因：", file=sys.stderr)
        print("  1) 没 source set_env.sh            -> 报 host_env_os 无效，见文件头注释", file=sys.stderr)
        print("  2) TBE 依赖缺失                    -> EC0010: No module named 'scipy'", file=sys.stderr)
        print("  3) numpy 2.x 与 TBE 不兼容         -> EC0010: np.float_ was removed，降到 1.26.4",
              file=sys.stderr)
        print("  4) --input_shape 节点名写错        -> E10016: Opname [...] is not found", file=sys.stderr)
        print("  5) 漏写 --soc_version              -> E10054", file=sys.stderr)
        raise SystemExit(ret)

    if not os.path.exists(om_path):
        raise SystemExit(f"ATC 返回 0 但没找到 OM 文件: {om_path}")

    size = os.path.getsize(om_path)
    print("\n=== 转换成功 ===")
    print(f"OM 文件 : {om_path}")
    print(f"大小    : {size} 字节 ({size / 1e6:.2f} MB)")
    print("\n板端验证:")
    print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
    print(f"  python3 scripts/infer_om.py --model {om_path} --image data/images/bus.jpg")


if __name__ == "__main__":
    main()
