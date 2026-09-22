"""环境自检：确认 yolov8_demo 环境满足「pt -> ONNX -> OM」全链路要求。

用法（在 WSL2 里，已激活 yolov8_demo 环境）:
    python scripts/check_env.py

退出码 0 = 全部通过；非 0 = 有项目未通过。

为什么做成脚本而不是一行 `python -c "..."`：
    多行命令粘贴到交互式终端时，只要收尾的引号/EOF 丢一个，bash 就会停在
    续行提示符 `>` 上等待，表现为「敲了命令但没有任何输出」。
    写成文件、用 `python scripts/check_env.py` 一行调用，就不会有这个坑。
"""
import os
import sys

FAIL = []
WARN = []


def ok(msg):
    print(f"[正常] {msg}")


def bad(msg, fix=""):
    print(f"[异常] {msg}")
    if fix:
        print(f"       处理：{fix}")
    FAIL.append(msg)


def warn(msg):
    print(f"[注意] {msg}")
    WARN.append(msg)


def section(title):
    print(f"\n=== {title} ===")


def main():
    print("=" * 62)
    print("yolov8_demo 环境自检")
    print("=" * 62)
    print(f"Python : {sys.version.split()[0]}  ({sys.executable})")

    # ---------------- Python 侧依赖 ----------------
    section("1. Python 侧依赖")

    mods = {}
    for name in ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime",
                 "onnxsim", "onnxslim", "scipy", "decorator", "attr", "psutil",
                 "synr", "absl", "tornado"):
        try:
            mods[name] = __import__(name)
        except Exception as e:                                  # noqa: BLE001
            mods[name] = None
            bad(f"{name} 导入失败: {type(e).__name__}: {e}",
                f"pip install -r requirements.txt "
                f"--extra-index-url https://download.pytorch.org/whl/cpu")

    for name in ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime"):
        m = mods.get(name)
        if m is not None:
            print(f"  {name:<12s} {getattr(m, '__version__', '?')}")

    # numpy 必须 <2：CANN 的 TBE 用了 numpy 2.0 已删除的 np.float_
    np_mod = mods.get("numpy")
    if np_mod is not None:
        if np_mod.__version__.startswith("1."):
            ok(f"numpy {np_mod.__version__} < 2（满足 CANN TBE 要求）")
        else:
            bad(f"numpy {np_mod.__version__} >= 2，CANN TBE 会报 "
                f"EC0010: np.float_ was removed",
                'pip install "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "ml_dtypes<0.6"')

    # cv2 必须是完整的 4.x：numpy 2.x 会把 opencv 顶到 5.x，导致 cv2 模块残缺
    cv2_mod = mods.get("cv2")
    if cv2_mod is not None:
        if not getattr(cv2_mod, "__file__", None):
            bad("cv2.__file__ 为 None，opencv 安装不完整（常见于 "
                "opencv-python 与 opencv-python-headless 互相覆盖）",
                "pip uninstall -y opencv-python opencv-python-headless && "
                "pip install --no-cache-dir opencv-python-headless==4.11.0.86")
        elif not hasattr(cv2_mod, "setNumThreads"):
            bad("cv2 缺少 setNumThreads，模块不完整（numpy 与 opencv 版本不匹配）",
                "pip install \"numpy==1.26.4\" \"opencv-python-headless==4.11.0.86\"")
        else:
            cv2_mod.setNumThreads(1)
            ok(f"cv2 {cv2_mod.__version__} 完整可用（setNumThreads 正常）")

    # extra-index 没加会装成 CUDA 版 torch，体积大且 310B 用不到
    t = mods.get("torch")
    if t is not None:
        if "cpu" in t.__version__:
            ok(f"torch {t.__version__} 是 CPU 版")
        else:
            warn(f"torch {t.__version__} 不是 CPU 版（可能是 CUDA 版）。"
                 f"310B 上用不到 CUDA，建议重装："
                 f"pip install torch==2.5.1 torchvision==0.20.1 "
                 f"--index-url https://download.pytorch.org/whl/cpu")

    # ---------------- CANN / ATC ----------------
    section("2. CANN / ATC（onnx -> om 必需）")

    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if ascend_home:
        ok(f"ASCEND_HOME_PATH={ascend_home}")
    else:
        bad("未设置 ASCEND_HOME_PATH，说明 CANN 环境变量没配置",
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh")

    atc = None
    for p in (os.path.join(ascend_home, "bin", "atc") if ascend_home else "",
              "/usr/local/Ascend/ascend-toolkit/latest/bin/atc"):
        if p and os.path.exists(p):
            atc = p
            break
    if atc:
        ok(f"找到 atc: {atc}")
    else:
        bad("找不到 atc 命令（ONNX 转 OM 会做不了）",
            "安装 CANN Toolkit，见 README 第 3 节；或 bash env/setup_yolov8_demo.sh")

    stub = (os.path.join(ascend_home, "runtime", "lib64", "stub")
            if ascend_home else "")
    if stub and os.path.isdir(stub):
        ok("找到 libascend_hal.so stub 目录（无 NPU 驱动时必需）")
    elif ascend_home:
        warn(f"未找到 stub 目录 {stub}；atc 可能报 "
             f"error while loading shared libraries: libascend_hal.so")

    # TBE 用的是【系统 python3】，不是当前 conda 环境的 python
    section("3. 系统 python3 的 TBE 依赖（ATC 实际使用的解释器）")
    sys_py = "/usr/bin/python3"
    if os.path.exists(sys_py) and os.path.realpath(sys_py) != os.path.realpath(sys.executable):
        import subprocess
        code = ("import numpy, scipy;"
                "print(numpy.__version__, hasattr(numpy,'float_'))")
        try:
            r = subprocess.run([sys_py, "-c", code], capture_output=True,
                               text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                ver, has_float = r.stdout.split()
                if has_float == "True":
                    ok(f"系统 python3 numpy {ver}，np.float_ 可用")
                else:
                    bad(f"系统 python3 的 numpy {ver} >= 2（TBE 会失败）",
                        f"pip install --target=/usr/local/lib/python3.10/dist-packages "
                        f"--upgrade scipy decorator psutil synr absl-py tornado && "
                        f"rm -rf /usr/local/lib/python3.10/dist-packages/numpy* && "
                        f"pip install --target=/usr/local/lib/python3.10/dist-packages "
                        f"--no-deps --force-reinstall numpy==1.26.4")
            else:
                bad("系统 python3 缺少 TBE 依赖（scipy/numpy 等）",
                    "见 README 3.1 第 6 步")
        except Exception as e:                                  # noqa: BLE001
            warn(f"探测系统 python3 失败: {e}")
    else:
        warn("系统 python3 与当前解释器相同或不存在，跳过")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 62)
    if FAIL:
        print(f"自检未通过：{len(FAIL)} 项异常，{len(WARN)} 项注意")
        for f in FAIL:
            print(f"  - {f}")
        print("=" * 62)
        return 1

    print("环境就绪" + (f"（{len(WARN)} 项注意，见上）" if WARN else ""))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
