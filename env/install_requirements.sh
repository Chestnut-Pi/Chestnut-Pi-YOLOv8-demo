#!/usr/bin/env bash
# ===========================================================================
#  只装 Python 依赖（不含 CANN），专门解决「照抄 requirements.txt 装不上」的问题。
#
#  用法:
#     bash env/install_requirements.sh                    # 用 requirements.template.txt
#     bash env/install_requirements.sh my_req.txt         # 装自己的文件
#     bash env/install_requirements.sh -n myenv           # 指定 conda 环境名
#     bash env/install_requirements.sh --cn               # 走清华镜像
#
#  ---------------------------------------------------------------------------
#  为什么不能直接 `pip install -r requirements.txt`？四个必踩的坑：
#
#  坑 1  requirements.txt 语法是【一行一个 requirement】
#        写成 "torch==2.5.1  torchvision==0.20.1" 会报
#        Invalid requirement: Expected comma (within version specifier)...
#        -> 本脚本会先用 `tr` 把空格展开成换行，再交给 pip
#
#  坑 2  torch==2.5.1+cpu 在 PyPI 上不存在
#        +cpu 是 PyTorch 官方源独有的 local version 标签，PyPI 只有 CUDA 版
#        -> 本脚本统一加 --extra-index-url https://download.pytorch.org/whl/cpu
#
#  坑 3  ml_dtypes==0.6.0 要求 numpy>=2.0.0，与 numpy==1.26.4 互斥
#        -> pip 报 ResolutionImpossible，一个包都装不上
#        -> 本脚本自动把 ml_dtypes 放宽到 <0.6
#
#  坑 4  ultralytics 会拉 opencv-python，与 opencv-python-headless 抢 cv2 模块
#        -> 报 module 'cv2' has no attribute 'setNumThreads'
#        -> 本脚本最后卸掉 opencv-python 并删净 cv2 残留目录
#
#  另外：pip install requirements.txt（不带 -r）是错的，pip 会把它当成包名
# ===========================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-yolov8_demo}"
CONDA_ROOT="${CONDA_ROOT:-/home/$USER/miniconda3}"
USE_CN=0
REQ_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cn)     USE_CN=1 ;;
    -n)       ENV_NAME="$2"; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    -*)       echo "未知参数: $1" >&2; exit 2 ;;
    *)        REQ_FILE="$1" ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "$REQ_FILE" ] || REQ_FILE="$HERE/requirements.template.txt"

log()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[正常]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[注意]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[异常]\033[0m %s\n' "$*" >&2; exit 1; }

PIP_EXTRA=()
[ "$USE_CN" = "1" ] && PIP_EXTRA+=(-i https://pypi.tuna.tsinghua.edu.cn/simple)

log "1/5  定位 conda 环境"
for cand in "$CONDA_ROOT" "$HOME/miniconda3" "$HOME/anaconda3" /root/miniconda3 /opt/conda; do
  [ -x "$cand/bin/conda" ] && { CONDA_ROOT="$cand"; break; }
done
[ -x "$CONDA_ROOT/bin/conda" ] || die "找不到 conda，请用 CONDA_ROOT=... 指定"

export PATH="$CONDA_ROOT/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# conda 26+ 必须先接受 Anaconda 源 ToS，否则 conda create 静默失败
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "复用已存在的环境: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10 || die "conda create 失败"
fi
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || die "找不到 $PY"
echo "Python: $($PY -V 2>&1)"
"$PY" -m pip install -U pip setuptools wheel "${PIP_EXTRA[@]}" -q

log "2/5  规范化 requirements 文件（把一行多包展开成一行一个）"
[ -f "$REQ_FILE" ] || die "找不到 $REQ_FILE"
NORM="$(mktemp)"
# 去掉注释行、把空格/制表符当分隔符展开、过滤空行
sed -e 's/#.*$//' "$REQ_FILE" | tr -s ' \t' '\n' | sed -e 's/^[[:space:]]*//' -e '/^$/d' > "$NORM"
echo "原始行数: $(grep -cve '^[[:space:]]*$' "$REQ_FILE" || true)  ->  展开后条目数: $(wc -l < "$NORM")"

# 解开 numpy<2 与 ml_dtypes 0.6.0 的死锁
if grep -qE '^ml_dtypes==0\.6\.0$' "$NORM"; then
  warn "检测到 ml_dtypes==0.6.0（要求 numpy>=2.0.0），与 numpy<2 冲突 -> 放宽为 ml_dtypes<0.6"
  sed -i 's/^ml_dtypes==0\.6\.0$/ml_dtypes<0.6/' "$NORM"
fi

log "3/5  安装（torch 走官方 CPU 源）"
# 先单独装 torch/torchvision：它们只在 PyTorch 官方源里有 CPU 版
TORCH_LINES="$(grep -E '^(torch|torchvision)==' "$NORM" || true)"
if [ -n "$TORCH_LINES" ]; then
  # PyPI 上不存在 +cpu 的包，去掉后缀再交给 --extra-index-url 解析
  TORCH_REQ="$(echo "$TORCH_LINES" | sed 's/+cpu$//')"
  echo "$TORCH_REQ" | while read -r r; do [ -n "$r" ] && echo "  torch 组件: $r"; done
  # shellcheck disable=SC2086
  "$PY" -m pip install $TORCH_REQ --extra-index-url https://download.pytorch.org/whl/cpu \
    || die "torch 安装失败"
  grep -vE '^(torch|torchvision)==' "$NORM" > "$NORM.2" && mv "$NORM.2" "$NORM"
fi

"$PY" -m pip install "${PIP_EXTRA[@]}" -r "$NORM" || die "依赖安装失败"

log "4/5  修正 numpy / opencv"
# 必须先把 opencv-python 与 headless 版【都】卸掉：两者都提供 cv2 模块，
# 只卸一个会让 cv2 目录残留，cv2.__file__ 变成 None，
# ultralytics 导入时仍报 `module 'cv2' has no attribute 'setNumThreads'`
"$PY" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
SP="$CONDA_ROOT/envs/$ENV_NAME/lib/python3.10/site-packages"
# 【实测坑】只删 cv2 目录和 opencv_python-* 的 dist-info，不要用
# `cv2.*.dist-info` 这种通配 —— 它会把 opencv_python_headless-*.dist-info
# 一起删掉，随后 pip 认为包还在、不再重装，结果 cv2 彻底没了。
rm -rf "$SP/cv2" "$SP"/opencv_python-*.dist-info 2>/dev/null || true
# 注意顺序：先装 headless，最后再钉 numpy（避免 pip 把 numpy 顶回 2.x）
"$PY" -m pip install --no-cache-dir "${PIP_EXTRA[@]}" \
      "opencv-python-headless==4.11.0.86" "ml_dtypes<0.6" || die "opencv-python-headless 安装失败"
"$PY" -m pip install --no-cache-dir "${PIP_EXTRA[@]}" "numpy==1.26.4" \
  || die "numpy 降级失败（CANN TBE 要求 numpy<2）"

# 自检：cv2 必须真的能导入，否则提前失败而不是留个半坏环境
"$PY" -c "import cv2; print('cv2', cv2.__version__, '->', cv2.__file__)" \
  || die "cv2 不可用，请执行：pip install --force-reinstall --no-cache-dir opencv-python-headless==4.11.0.86"

log "5/5  验证"
FAIL=0
chk() { if "$@" >/dev/null 2>&1; then ok "$1 $2 $3"; else warn "未通过: $*"; FAIL=$((FAIL+1)); fi; }

"$PY" - <<'EOF'
import sys
import numpy, cv2, torch, ultralytics, onnx, onnxruntime
print("torch       :", torch.__version__)
print("ultralytics :", ultralytics.__version__)
print("onnx        :", onnx.__version__)
print("onnxruntime :", onnxruntime.__version__)
print("opencv      :", cv2.__version__)
print("numpy       :", numpy.__version__)
bad = []
if not numpy.__version__.startswith("1."):
    bad.append(f"numpy 必须 <2（CANN TBE 依赖 np.float_），当前 {numpy.__version__}")
if not cv2.__version__.startswith("4."):
    bad.append(f"opencv 必须是 4.x（与 numpy<2 配套），当前 {cv2.__version__}")
cv2.setNumThreads(1)
if bad:
    print("\n".join("【异常】" + b for b in bad)); sys.exit(1)
print("【正常】Python 侧依赖就绪，满足 pt -> ONNX 与 CANN TBE 的要求")
EOF

rm -f "$NORM"
if [ "$FAIL" -gt 0 ]; then
  warn "有 $FAIL 项未通过"
  exit 1
fi
ok "完成。注意：CANN（atc 命令）不在这里安装，请跑 env/setup_yolov8_demo.sh"
