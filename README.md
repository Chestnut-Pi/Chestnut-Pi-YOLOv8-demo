# Chestnut-Pi-YOLOv8-demo — YOLOv8n 全链路环境（pt → ONNX → OM）

[English](README_EN.md) | **简体中文**

在 **WSL2 (Ubuntu 22.04 x86_64)** 中从零搭建的 `yolov8_demo` 环境，装齐
`yolov8n.pt → yolov8n.onnx → yolov8n_bs1.om` 全链路依赖，并**已实际跑通全流程**。

目标硬件：**栗子派昇腾 310B**（Atlas 200I DK A2，`soc_version=Ascend310B1`）。

---

## 0. 实测结果（本工程已完成）

一条命令跑完整条链路，全程 **28.4 秒**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/run_all.py --force
```

| 步骤 | 产物 | 实测结果 |
|---|---|---|
| 1. 准备权重 | `model/yolov8n.pt` | 6.55 MB |
| 2. 导出 ONNX | `model/yolov8n.onnx` | **12.85 MB**，opset 11，`images[1,3,640,640]` → `output0[1,84,8400]`，算子 259→**234**（onnxslim 化简） |
| 3. 校验 ONNX | — | ONNX Runtime 推理出 **5 个目标**（bus 0.8434 + person ×4），NaN/Inf 均为 0 |
| 4. ATC 转换 | `model/yolov8n_bs1.om` | **7,173,922 字节（7.17 MB）**，耗时 **25.0 秒**（16 核 x86），`ATC run success` |
| 5. 校验 OM | — | 文件头 `IMOD`，内嵌 `Ascend310B1` / `images` / `output0` |

转换全过程日志留档在 `results/convert_log.txt`。

**与板端实测对比**（板端环境：栗子派 310B，CANN 7.0.RC1，4 核 ARM）：

| 项目 | 板端实测 | 本工程（WSL2 x86） | 差异 |
|---|---|---|---|
| ONNX 节点/形状 | `images[1,3,640,640]` → `output0[1,84,8400]` | 完全一致 | 0 |
| ONNX 大小 | 12.85 MB | 12.85 MB | 0 |
| OM 大小 | 7,184,263 字节 | 7,173,922 字节 | **0.15%** |
| ATC 耗时 | 约 10 分钟（9m27s） | 25 秒 | 22.7×（16 核 vs 4 核） |

> 实测确认：`run_all.py` 的第 4 步（ATC 转换）在 WSL2 内即可完成，
> **不需要插着 NPU，也不需要装 NPU 驱动**。只有 OM 推理（第 6 步）
> 必须搬到开发板上跑，因为要访问 `/dev/davinci*`。

**可复现性**：连续多次 `--force` 全量重跑，OM 大小稳定为 7,173,922 字节，
ATC 退出码均为 0。OM 文件 MD5 每次略有不同（ATC 会把构建时间戳写进产物），
属正常现象，不影响加载与推理。

---

## 1. 目录结构

```
Chestnut-Pi-YOLOv8-demo/
├── README.md                      本文件（中文）：环境搭建 + 全流程操作说明
├── README_EN.md                   英文版说明（English）
├── requirements.txt               全链路 Python 依赖（pip freeze 实测导出，57 包）
├── .gitignore                     忽略 ATC 临时产物（kernel_meta 等）
├── env/
│   ├── setup_yolov8_demo.sh       ★一键搭建（conda + pip + CANN + 系统 python3 + 自动验证）
│   ├── install_requirements.sh    只装 Python 依赖（专门修「requirements.txt 装不上」）
│   └── requirements.template.txt  修正版依赖清单（一行一个包，附格式说明）
├── scripts/
│   ├── yolo_common.py             预处理/后处理公共实现（三层共用，保证可比性）
│   ├── pt_baseline.py             [层1] PyTorch 基线推理
│   ├── export_onnx.py             [层2] pt -> onnx 导出
│   ├── check_onnx.py              [层2] ONNX 结构与数值自检
│   ├── atc_convert.py             [层3] onnx -> om（Python 封装 atc 命令）
│   ├── atc_convert.sh             [层3] onnx -> om（纯 shell 等价版本）
│   ├── infer_om.py                [板端] pyACL 跑 OM 推理
│   ├── check_om_out.py            辅助：确认 OM 输出张量布局
│   ├── check_env.py               环境自检（退出码 0 = 全链路依赖就绪）
│   └── run_all.py                 一键串起整条链路（5 步）
├── model/
│   ├── yolov8n.pt                 PyTorch 权重（6.55 MB）
│   ├── yolov8n.onnx               导出产物（12.85 MB）
│   └── yolov8n_bs1.om             ATC 转换产物（7.17 MB）★ 最终交付
├── data/
│   ├── images/bus.jpg             测试图（810x1080，含 bus + person）
│   └── coco_names.txt             80 类名称
├── logs/                          运行后生成的原始输出（*.npy / 性能 txt，不入库）
└── results/
    ├── convert_log.txt            全链路转换实测日志（留档证据）
    └── pt_result/                 运行 pt_baseline.py 后生成的基线可视化（不入库）
```

---

## 2. 环境总览

| 项目 | 版本 | 说明 |
|---|---|---|
| 宿主 | Windows 11 + WSL2 | Ubuntu 22.04.5 LTS, x86_64, 16 核 / 15 GB |
| conda 环境 | `yolov8_demo` | `$HOME/miniconda3/envs/yolov8_demo`（可用 `CONDA_ROOT` 覆盖） |
| Python | 3.10.20 | CANN 8.0.RC1 支持 3.7 ~ 3.11 |
| PyTorch | 2.5.1+cpu | 310B 无 CUDA，装 CPU 版 |
| Ultralytics | 8.3.40 | 负责 pt → onnx 导出 |
| ONNX / ORT | 见 requirements.txt | 导出格式 + CPU 侧数值校验 |
| **CANN Toolkit** | **8.0.RC1** (x86_64) | 提供 `atc` 模型转换工具 |
| CANN kernels | 310b 算子包 | 310B 的算子实现 |
| NumPy | **1.26.4** | **必须 <2**，CANN TBE 用了 `np.float_` |
| OpenCV | 4.11.0.86 | 需与 numpy<2 兼容 |

### 2.1 三个环节各需要什么

| 环节 | 依赖 | 是否必须 |
|---|---|---|
| pt → onnx | `torch` + `ultralytics` + `onnx` + `onnxslim` | WSL2 内完成 |
| onnx 校验 | `onnxruntime` | WSL2 内完成 |
| **onnx → om** | **CANN Toolkit 的 `atc`** + TBE 的 Python 依赖 | WSL2 内完成（**不需要 NPU、不需要驱动**） |
| om 推理 | 板端 pyACL（随板端 CANN 提供）+ `/dev/davinci*` | **必须在开发板上** |

> **关键点**：ATC 是**编译器**，在 x86 PC 上就能把 ONNX 编译成 310B 的 OM，
> 不需要插着 NPU。但 OM 只能在**同型号芯片**上加载运行——所以推理那一步
> 必须搬到开发板上（本工程的 `scripts/infer_om.py`）。

### 2.2 两个容易卡住的隐藏依赖

1. **`libascend_hal.so`**：`atc.bin` 动态链接了这个库，它正常由 **NPU 驱动包**提供。
   本机没装驱动，会直接报
   `error while loading shared libraries: libascend_hal.so`。
   CANN Toolkit **自带一份 stub 版本**，加进库搜索路径即可：

   ```bash
   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub"
   ```

   > ⚠️ **stub 必须追加在最后，不能放在最前面。**
   > `runtime/lib64/stub/` 里有一批与真实库**同名**的空壳库
   > （`libascendcl.so`、`libacl_dvpp.so` 等，只有几十 KB，不含实现符号）。
   > 若 stub 排在前面，这些空壳库会抢先被加载，导致 **pyACL 导入失败**：
   > `ImportError: .../acl.so: undefined symbol: aclprofSetConfig`
   >
   > 这个错误的隐蔽之处在于：`atc` 只需要 stub 里的 `libascend_hal.so`，
   > 两种顺序下都能正常工作，**只有用 pyACL 推理时才会暴露**。

   `scripts/atc_convert.py` 与 `atc_convert.sh` 已自动检测并补上这一步
   （它们只在 atc 子进程里注入该路径，不污染当前 shell）。


2. **ATC 的 TBE 用系统 `python3`，不是 conda 环境的 python**。
   即使 `conda activate yolov8_demo` 之后 `numpy` 已是 1.26.4，
   ATC 仍可能报 `np.float_ was removed`——因为它走的是 `/usr/bin/python3`。
   必须给系统 python3 也装一份 `numpy<2` 与 TBE 依赖（见 3.1 第 6 步）。

---

## 3. 一键搭建环境

```bash
cd /path/to/Chestnut-Pi-YOLOv8-demo

# 推荐：完整搭建（conda + pip + CANN + 系统 python3 依赖），最后自动分层验证
bash env/setup_yolov8_demo.sh

# 其它用法
bash env/setup_yolov8_demo.sh --cn             # pip 走清华镜像
bash env/setup_yolov8_demo.sh --no-cann        # 只装 Python 侧
bash env/setup_yolov8_demo.sh --verify-only    # 只跑验证
bash env/setup_yolov8_demo.sh --env-name myenv # 自定义环境名

# 只装 Python 依赖（不装 CANN）
bash env/install_requirements.sh
```

脚本分 7 步：环境自检 → 创建 conda 环境 → 装 pt/ONNX 依赖 → 修正 numpy/opencv →
装 CANN → 配环境变量 → **给系统 python3 补 TBE 依赖** → 分层验证。
全部幂等，可重复执行。

实测：这条命令在干净环境下跑完后，直接 `run_all.py --force` 即可产出 OM
（已验证：7,173,922 字节，ATC 耗时 28.4 秒，总耗时 32.2 秒）。

安装完成后：

```bash
source ~/.bashrc          # 自动 source CANN set_env.sh 并激活环境
conda env list | grep yolov8_demo
atc --version
```

### 3.0 关于「照着 requirements.txt 装不上」

`pip install requirements.txt` 这种写法（没有 `-r`）以及把多个包写在一行，
都会让 pip 直接报错退出。四个必踩的坑与修法：

| 坑 | 报错 | 修法 |
|---|---|---|
| 命令少了 `-r` | `Could not find a version that satisfies the requirement requirements.txt` | `pip install -r requirements.txt` |
| 一行写多个包 | `Invalid requirement: 'torch==2.5.1+cpu  torchvision==...': Expected comma...` | 一行一个包 |
| `torch==2.5.1+cpu` | `No matching distribution found for torch==2.5.1+cpu` | 加 `--extra-index-url https://download.pytorch.org/whl/cpu` |
| `ml_dtypes==0.6.0` + `numpy==1.26.4` | `ResolutionImpossible`（0.6.0 要求 numpy>=2） | 放宽为 `ml_dtypes<0.6` |

`env/install_requirements.sh` 会自动处理这四条 —— 你甚至可以直接把
**一行多包的原始文件**丢给它，它会先规范化再安装（已实测：17 行含 57 个包，
从零装到全部可用）。

> 根目录 `requirements.txt` 已按上表修正（`ml_dtypes<0.6`），安装示例里也补上了
> `--extra-index-url`，所以现在这条命令可以直接装通：
>
> ```bash
> pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
> ```

### 3.1 手工安装（等价步骤，便于排查）

```bash
# --- 1) conda 环境 ---
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n yolov8_demo python=3.10
conda activate yolov8_demo

# --- 2) pt -> ONNX 依赖 ---
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.3.40 onnx onnxruntime onnxsim opencv-python-headless

# --- 3) CANN 8.0.RC1 (x86_64) ---
mkdir -p /root/ascend_pkgs && cd /root/ascend_pkgs
B="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1"
curl -LO "$B/Ascend-cann-toolkit_8.0.RC1_linux-x86_64.run"
curl -LO "$B/Ascend-cann-kernels-310b_8.0.RC1_linux.run"
chmod +x *.run
sudo ./Ascend-cann-toolkit_8.0.RC1_linux-x86_64.run --install --quiet
# --quiet 仍会 fork 一个 xfce4-terminal 显示进度条，装完不退出，需手动清掉
pkill -f 'Ascend-cann-toolkit'; pkill -f 'xfce4-terminal -e'
sudo ./Ascend-cann-kernels-310b_8.0.RC1_linux.run --install --quiet
pkill -f 'Ascend-cann-kernels'; pkill -f 'xfce4-terminal -e'

# --- 4) 环境变量 + 库路径（无 NPU 驱动时必须加 stub） ---
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
# 注意：stub 必须追加在最后（不能写成 stub:$LD_LIBRARY_PATH），
# 否则 stub 里的空壳 libascendcl.so 会抢先加载，导致 import acl 失败
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub"' >> ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/.bashrc

# --- 5) conda 侧 TBE 依赖 ---
pip install "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
            decorator attrs psutil scipy synr absl-py tornado

# --- 6) 系统 python3 侧 TBE 依赖（关键！ATC 的 TBE 用的是系统 python3） ---
# Debian 自带 pip 22 不支持 --root-user-action，用 conda 的新版 pip + --target
CONDA_PIP="$HOME/miniconda3/envs/yolov8_demo/bin/pip"
SYS_SP=/usr/local/lib/python3.10/dist-packages
$CONDA_PIP install --target="$SYS_SP" --upgrade \
    "numpy==1.26.4" scipy decorator psutil synr absl-py tornado
/usr/bin/python3 -c "import numpy; print(numpy.__version__, hasattr(numpy,'float_'))"
```

> **第 6 步最容易漏，也最难排查**：conda 环境里的 numpy 明明已经是 1.26.4，
> ATC 却仍然报 `np.float_ was removed`——因为 ATC 的 TBE 走的是系统 `python3`。

---

## 4. 全链路跑一遍

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh    # 每次新终端都要执行
conda activate yolov8_demo
cd /path/to/Chestnut-Pi-YOLOv8-demo

python scripts/run_all.py                             # 一键走完 5 步
```

也可分步执行，便于定位问题：

```bash
# [层1] PyTorch 基线 —— 后续所有精度都以它为基准
python scripts/pt_baseline.py --image data/images/bus.jpg

# [层2] pt -> onnx（opset 11，静态 shape）
python scripts/export_onnx.py --pt model/yolov8n.pt --onnx model/yolov8n.onnx

# [层2] ONNX 自检（结构 + ONNX Runtime 数值）
python scripts/check_onnx.py --onnx model/yolov8n.onnx --image data/images/bus.jpg

# [层3] onnx -> om（这一步最慢，约 5~10 分钟）
python scripts/atc_convert.py --onnx model/yolov8n.onnx --out model/yolov8n_bs1

# [板端] OM 推理（在开发板上执行）
python3 scripts/infer_om.py --model model/yolov8n_bs1.om --image data/images/bus.jpg
```

### 4.1 ATC 等价原生命令

`scripts/atc_convert.py` 拼出的就是这条命令：

```bash
atc --model=model/yolov8n.onnx \
    --framework=5 \
    --output=model/yolov8n_bs1 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend310B1 \
    --log=error
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `--framework` | 5 | 5 = ONNX，必须写对 |
| `--output` | model/yolov8n_bs1 | 输出前缀，ATC 自动补 `.om` |
| `--input_format` | NCHW | 图像模型固定 NCHW |
| `--input_shape` | images:1,3,640,640 | 节点名**必须**与 ONNX 里一致，写错报 E10016 |
| `--soc_version` | Ascend310B1 | 栗子派 310B 的芯片型号 |
| `--log` | error | 只留错误输出 |

也提供了纯 shell 的等价封装：`bash scripts/atc_convert.sh`（支持用环境变量覆盖
`SOC_VERSION` / `INPUT_SHAPE` 等参数，适合在 CI 里直接调用）。

---

## 4.2 OM 只能在开发板上推理

`scripts/infer_om.py` 用的是板端 pyACL，需要 `/dev/davinci0` 设备节点，
所以**必须在开发板上执行**。把整个工程目录拷到板子（或通过 NFS/SSH 共享）后：

```bash
# 在开发板上
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 scripts/infer_om.py --model model/yolov8n_bs1.om --image data/images/bus.jpg
```

脚本会打印模型信息（输入/输出 shape、dtype、字节数）、检测结果，
并把可视化图存到 `results/om_result.jpg`、原始输出存到 `logs/om_raw_output.npy`
（可用 `--npy logs/om_raw_output.npy` 喂给 `check_om_out.py` 核对布局）。

> 板端 CANN 版本需要与转换时**兼容**。本工程用 CANN 8.0.RC1 转换；
> 若板端是 CANN 7.0.RC1，建议在板上用同版本 ATC 重新转换一次（参数完全相同）。

---

## 5. 踩过的坑（按报错索引）

| 报错 / 现象 | 原因 | 处理 |
|---|---|---|
| `CondaToSNonInteractiveError` | conda 26+ 要求先接受 Anaconda 源 ToS | `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`（两个 channel 都要） |
| `conda env list` 里没有环境，pip 却装进了 base | `conda create` 失败但脚本没检查退出码，`conda activate` 静默失败 | 用**绝对路径**的 `$CONDA_ROOT/envs/<env>/bin/python -m pip`，不依赖 activate |
| **`atc.bin: error while loading shared libraries: libascend_hal.so`** | 本机没装 NPU 驱动，该库正常由驱动包提供 | toolkit 自带 stub，**追加**到库路径末尾：`export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:.../runtime/lib64/stub"`（脚本已自动处理） |
| **`ImportError: acl.so: undefined symbol: aclprofSetConfig`** | stub 路径被放在了 `LD_LIBRARY_PATH` **最前面**，空壳 `libascendcl.so` 抢先加载 | 把 stub 改到**最后**：`export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:.../runtime/lib64/stub"`。注意 `atc` 在两种顺序下都能用，只有 pyACL 会暴露此问题 |
| **CANN 安装器 `--install --quiet` 之后永久卡住不退出** | 它会 fork 一个 `xfce4-terminal` 显示进度条，装完该进程不退 | `pkill -f 'Ascend-cann-toolkit'; pkill -f 'xfce4-terminal -e'`，再检查 `/usr/local/Ascend/ascend-toolkit/latest/bin/atc` 是否存在 |
| atc 报 `host_env_os 参数无效` | 没 source `set_env.sh`，ATC 靠环境变量推断 host 平台 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| **`EC0010: ... [AttributeError: np.float_ was removed ...]`**（conda 里 numpy 已是 1.26.4 仍然报） | **ATC 的 TBE 用的是系统 `python3`，不是 conda 环境的 python** | 用 `pip install --target=/usr/local/lib/python3.10/dist-packages numpy==1.26.4 ...` 给系统 python3 也装一份（README 3.1 第 6 步） |
| `EC0010: Failed to import Python module [No module named 'scipy']` | TBE 的 Python 依赖没装 | `pip install decorator attrs psutil scipy synr absl-py tornado` |
| `E10016: Opname [input] ... is not found` | `--input_shape` 的节点名与 ONNX 不一致 | YOLOv8 是 `images`，用 `check_onnx.py` 确认 |
| `E10054: [--soc_version] is empty` | 漏写 `--soc_version` | 补 `--soc_version=Ascend310B1` |
| `module 'cv2' has no attribute 'setNumThreads'`，ultralytics 导入失败 | `opencv-python` 与 `opencv-python-headless` 同时装，互相覆盖 | 卸载 `opencv-python`，只留 `opencv-python-headless` |
| ultralytics 导出时提示 `simplifier failure: No module named 'onnxslim'` | ultralytics 8.3.x 优先用 onnxslim（不是 onnxsim），且是运行时才装、当次不生效 | `pip install onnxslim` 后重跑；或忽略（不化简也能转 OM） |
| ATC 卡住不动 / 负载飙升 | `--precision_mode=force_fp32` 会让 ATC 卡死 | **不要加这个参数** |
| 指定 FP16 反而变慢 | 310B AI Core 本身按 FP16 算，指定输入 FP16 要额外插转换算子 | 用默认（不加 `--input_fp16_nodes`） |
| `Import acl 失败` | 板端没 source CANN 环境变量 | 在板上 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `RuntimeError: 输入字节数不匹配` | ATC 的 `--input_shape` 与预处理输出形状不一致 | 两者都必须是 `(1,3,640,640)` |

---

## 6. 分层验证思路

三层各自独立可验证，这是排查问题的关键——**任何一层跑不通，都不必往下走**：

```
[层1] PyTorch    推理出框  ──→  作为精度基准
   ↓ 导出
[层2] ONNX       ONNX Runtime 能加载、能出框  ──→  证明「模型和图」没问题
   ↓ ATC 编译
[层3] OM         板端能加载、能出框  ──→  证明「转换」没问题
```

如果 ONNX 正常而 OM 异常，问题一定在 ATC 转换（参数、TBE 依赖、soc_version）；
如果 ONNX 本身就异常，就不用去动 ATC。

板端实测（CANN 7.0.RC1，栗子派 310B）三层一致性：

```
PyTorch 基线      5 个目标（bus 0.8729，person ×4）
ONNX              opset 11，images[1,3,640,640] → output0[1,84,8400]，12.85 MB
OM                与 ONNX 余弦相似度 0.99999976，逐框 IoU ≥ 0.9961
端到端性能        36.7 ms / 27.0 FPS（预处理 16.0 + 推理 15.2 + 后处理 5.3）
纯推理吞吐        15.18 ms / 65.9 FPS
```

---

## 7. 重新导出 requirements.txt

```bash
conda activate yolov8_demo
pip freeze > requirements.txt
```

注意 `requirements.txt` 只是**结果快照**。真正需要人工钉版本的只有几个：

```
numpy==1.26.4                    # 必须 <2：CANN TBE 依赖 np.float_
opencv-python-headless==4.11.0.86
torch==2.5.1
ultralytics==8.3.40
onnx==1.23.0
```

其余都是这些包的传递依赖，可以随版本浮动。

---

## 8. 注意事项

- **CANN 必须装在 Linux 文件系统**（`/usr/local/Ascend`），不要装在 `/mnt/d/`
  这类 Windows 挂载盘：NTFS 不支持符号链接与可执行权限，安装会失败。
- **目录名与文件名一律用英文**。CANN 工具链对中文路径支持不稳定，会导致转换失败。
- **不要随意 `apt upgrade`**。板端 Ubuntu 与 CANN 是配套验证过的，升级系统库可能
  破坏 CANN 的依赖关系。升级前先备份 OM 模型。
- 每次开新终端都要 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，
  `setup_yolov8_demo.sh` 已把它写进 `~/.bashrc`。
- ATC 转换期间会刷大量 `ImportWarning` / `SyntaxWarning`，这是 TBE 的正常噪音，
  只要最后是 `ATC run success` 就没问题。
- OM 只能在**同型号芯片 + 兼容 CANN 版本**上加载。跨芯片（如 310B → 310P）需要重新转换。
