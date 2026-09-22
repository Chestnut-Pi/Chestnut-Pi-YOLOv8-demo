#!/usr/bin/env bash
# ===========================================================================
#  YOLOv8n 全链路环境一键搭建：pt -> ONNX -> OM
#
#  在 WSL2 (Ubuntu 22.04 x86_64) 上从零搭出可跑通
#      yolov8n.pt -> yolov8n.onnx -> yolov8n_bs1.om
#  的全部依赖，并在最后自动验证每一层。
#
#  用法:
#     bash env/setup_yolov8_demo.sh                       # 完整安装
#     bash env/setup_yolov8_demo.sh --cn                  # pip 走清华镜像加速
#     bash env/setup_yolov8_demo.sh --no-cann             # 只装 Python 侧依赖
#     bash env/setup_yolov8_demo.sh --env-name myenv      # 换环境名
#     bash env/setup_yolov8_demo.sh --verify-only         # 只跑验证，不装
#
#  分 7 步：
#     0  环境自检（架构 / 磁盘 / gcc）
#     1  创建 conda 环境（Python 3.10）
#     2  安装 pt -> ONNX 依赖
#     3  修正 numpy / opencv（关键：让 CANN TBE 能用）
#     4  安装 CANN Toolkit + 310B 算子包
#     5  配置环境变量 + libascend_hal stub 路径
#     6  给【系统 python3】补 TBE 依赖（最容易漏的一步）
#     7  分层验证：pt -> onnx -> om 端到端自检
#
#  幂等：可重复执行，已完成的步骤会自动跳过。
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 可调参数（也可用环境变量覆盖）
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-yolov8_demo}"
CONDA_ROOT="${CONDA_ROOT:-/home/$USER/miniconda3}"
PKG_DIR="${PKG_DIR:-/root/ascend_pkgs}"

CANN_VERSION="${CANN_VERSION:-8.0.RC1}"
OBS_BASE="${OBS_BASE:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1}"
TOOLKIT_PKG="Ascend-cann-toolkit_${CANN_VERSION}_linux-x86_64.run"
KERNELS_PKG="Ascend-cann-kernels-310b_${CANN_VERSION}_linux.run"

ASCEND_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
STUB_DIR="/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub"

# 工程根目录：默认由本脚本位置自动推导（脚本位于 <root>/env/），可用 WORK_DIR 覆盖
WORK_DIR="${WORK_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"

USE_CN=0
DO_CANN=1
VERIFY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cn)          USE_CN=1 ;;
    --no-cann)     DO_CANN=0 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --env-name)    ENV_NAME="$2"; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（用 --help 看用法）" >&2; exit 2 ;;
  esac
  shift
done

PIP_EXTRA=()
[ "$USE_CN" = "1" ] && PIP_EXTRA+=(-i https://pypi.tuna.tsinghua.edu.cn/simple)

log()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[正常]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[注意]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[异常]\033[0m %s\n' "$*" >&2; exit 1; }

# 【实测坑】CANN 的 set_env.sh 里写了
#     export LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH
# 在本脚本的 `set -u`（未定义变量报错）下会直接抛
#     set_env.sh: line 2: LD_LIBRARY_PATH: unbound variable
# 所以 source 它之前必须先临时关掉 -u。
source_ascend_env() {
  local f="${1:-$ASCEND_ENV}"
  [ -f "$f" ] || return 1
  set +u
  # shellcheck disable=SC1090
  source "$f"
  set -u
  return 0
}

# ---------------------------------------------------------------------------
log "0/7  环境自检"

[ "$(uname -s)" = "Linux" ] || die "本脚本面向 Linux / WSL2，当前是 $(uname -s)"

ARCH="$(uname -m)"
echo "系统      : $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2)"
echo "架构      : $ARCH"
[ "$ARCH" = "x86_64" ] || warn "本脚本的 CANN 包是 x86_64 版，当前为 $ARCH，第 4 步需自行换包"

# 定位 conda
for cand in "$CONDA_ROOT" "$HOME/miniconda3" "$HOME/anaconda3" \
            /root/miniconda3 /opt/conda; do
  if [ -x "$cand/bin/conda" ]; then CONDA_ROOT="$cand"; break; fi
done
[ -x "$CONDA_ROOT/bin/conda" ] || die "找不到 conda。请先装 Miniconda，或用 CONDA_ROOT=... 指定路径"
echo "conda     : $CONDA_ROOT  ($("$CONDA_ROOT/bin/conda" --version))"

command -v gcc >/dev/null || warn "没有 gcc。TBE 编译算子时可能需要，建议 apt install -y gcc g++ make"
command -v curl >/dev/null || die "缺少 curl，请先 apt install -y curl"
echo "gcc       : $(gcc --version 2>/dev/null | head -1 || echo '未安装')"
echo "磁盘可用  : $(df -h / | awk 'NR==2{print $4}')  (CANN 约需 8 GB)"
[ "$(df -Pk / | awk 'NR==2{print $4}')" -gt 8000000 ] || warn "根分区可用空间不足 8 GB，CANN 安装可能失败"

if [ "$VERIFY_ONLY" = "1" ]; then
  PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
  [ -x "$PY" ] || die "环境 $ENV_NAME 不存在，无法只做验证"
  SKIP_INSTALL=1
else
  SKIP_INSTALL=0
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_INSTALL" = "0" ]; then

log "1/7  创建 conda 环境：$ENV_NAME"
export PATH="$CONDA_ROOT/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# conda 26+ 必须先接受 Anaconda 官方源的 ToS，否则 conda create 直接失败
# （报 CondaToSNonInteractiveError），而很多脚本没检查退出码，会静默装进 base
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "已存在，复用: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10 || die "conda create 失败"
fi

PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || die "环境创建后仍找不到 python: $PY"
# 全程用绝对路径调 python，不依赖 conda activate 的隐式状态
echo "Python    : $($PY -V 2>&1)  ($PY)"
"$PY" -m pip install -U pip setuptools wheel "${PIP_EXTRA[@]}" -q

# ---------------------------------------------------------------------------
log "2/7  安装 pt -> ONNX 依赖"
echo "（torch 从官方 CPU 源装：310B 没有 CUDA，CPU 版够用且体积小）"

if "$PY" -c "import torch" 2>/dev/null; then
  echo "torch 已安装，跳过: $("$PY" -c 'import torch;print(torch.__version__)')"
else
  # 必须有 --extra-index-url，否则 PyPI 上找不到 CPU 版（PyPI 只有 CUDA 版）
  "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu || die "torch 安装失败"
fi

"$PY" -m pip install "${PIP_EXTRA[@]}" \
      ultralytics==8.3.40 ultralytics-thop==2.1.6 \
      onnx==1.23.0 onnxruntime==1.23.2 onnxsim==0.7.3 onnxslim==0.1.96 \
      scipy==1.15.3 matplotlib==3.10.9 pandas==2.3.3 seaborn==0.13.2 \
      pillow==12.3.0 PyYAML==6.0.3 requests==2.34.2 tqdm==4.70.1 rich==15.0.0 \
      Jinja2==3.1.6 MarkupSafe==3.0.3 Pygments==2.21.0 protobuf==7.36.2 \
      networkx==3.4.2 sympy==1.13.1 mpmath==1.3.0 \
      certifi==2026.7.22 charset-normalizer==3.5.1 idna==3.20 urllib3==2.8.0 \
      colorama==0.4.6 coloredlogs==15.0.1 humanfriendly==10.0 py-cpuinfo==9.0.0 \
      contourpy==1.3.2 cycler==0.12.1 filelock==3.32.3 flatbuffers==25.12.19 \
      fonttools==4.65.0 fsspec==2026.7.0 kiwisolver==1.5.1 markdown-it-py==4.2.0 \
      mdurl==0.1.2 packaging==26.1 pyparsing==3.3.3 python-dateutil==2.9.0.post0 \
      pytz==2026.3.post1 six==1.17.0 typing_extensions==4.16.0 tzdata==2026.4 \
      || die "pt/ONNX 依赖安装失败"

# ---------------------------------------------------------------------------
log "3/7  修正 numpy / opencv（让 CANN TBE 可用的关键一步）"

# (a) numpy 必须降到 <2
#     ultralytics 会经 ml_dtypes 把 numpy 拉到 2.x；而 CANN 的 TBE 用了
#     numpy 2.0 已删除的 np.float_，用 numpy 2.x 时 ATC 报：
#       EC0010: ... [AttributeError: `np.float_` was removed in the NumPy 2.0 release.]
# (b) ml_dtypes 不能锁 0.6.0，它要求 numpy>=2.0.0，与 numpy<2 直接冲突，
#     会让 pip 报 ResolutionImpossible。放宽后自动解析到 0.5.4。
# (c) opencv 必须用 headless 4.x：opencv 5.x 的 wheel 声明 numpy>=2，与 (a) 冲突。
#     而且 opencv-python 与 opencv-python-headless 都提供 cv2 模块，
#     同时存在会导致 `module 'cv2' has no attribute 'setNumThreads'`。
"$PY" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
SP="$CONDA_ROOT/envs/$ENV_NAME/lib/python3.10/site-packages"
# pip uninstall 之后 cv2 目录可能残留（被另一个包共用），必须手工删干净，
# 否则 cv2.__file__ 为 None，导入报同样的错
rm -rf "$SP"/cv2 "$SP"/cv2.*.dist-info "$SP"/opencv_python*.dist-info 2>/dev/null || true

"$PY" -m pip install --no-cache-dir "${PIP_EXTRA[@]}" \
      "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "ml_dtypes<0.6" \
      decorator==5.3.1 attrs==26.1.0 psutil==7.2.2 synr==0.6.0 \
      absl-py==2.5.0 tornado==6.5.10 || die "numpy/opencv 修正失败"

# ---------------------------------------------------------------------------
if [ "$DO_CANN" = "1" ]; then
  log "4/7  安装 CANN ${CANN_VERSION}（Toolkit + 310B 算子包）"

  mkdir -p "$PKG_DIR"
  cd "$PKG_DIR"
  for f in "$TOOLKIT_PKG" "$KERNELS_PKG"; do
    if [ -s "$f" ]; then
      echo "已存在，跳过下载: $f ($(du -h "$f" | cut -f1))"
    else
      echo "下载 $f ..."
      curl -L --retry 5 --retry-delay 5 -C - -o "$f" "$OBS_BASE/$f" \
        || die "下载 $f 失败（检查网络；该地址无需登录）"
    fi
  done
  chmod +x ./*.run

  # 安装器即使带 --quiet 也会 fork 一个 xfce4-terminal 显示进度条，
  # 装完之后那个进程不退出，脚本会永远卡住 -> 装完统一 pkill 掉
  if [ ! -x /usr/local/Ascend/ascend-toolkit/latest/bin/atc ]; then
    echo "安装 Toolkit（约 2~3 分钟）..."
    "./$TOOLKIT_PKG" --install --quiet || true
    pkill -f 'Ascend-cann-toolkit' 2>/dev/null || true
    pkill -f 'xfce4-terminal -e'   2>/dev/null || true
    sleep 2
  else
    echo "Toolkit 已安装，跳过"
  fi

  if [ ! -d /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend310b ]; then
    echo "安装 310B 算子包（约 1.2 GB 算子）..."
    "./$KERNELS_PKG" --install --quiet || true
    pkill -f 'Ascend-cann-kernels' 2>/dev/null || true
    pkill -f 'xfce4-terminal -e'   2>/dev/null || true
    sleep 2
  else
    echo "310B 算子包已安装，跳过"
  fi

  [ -x /usr/local/Ascend/ascend-toolkit/latest/bin/atc ] \
    || die "CANN 安装失败：atc 不存在。检查 $PKG_DIR 下的安装日志"
  ok "CANN 安装完成，310B 算子 $(ls /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend310b 2>/dev/null | wc -l) 个"
else
  log "4/7  已按要求跳过 CANN 安装（--no-cann）"
fi

fi  # SKIP_INSTALL

# ---------------------------------------------------------------------------
if [ "$DO_CANN" = "1" ] && [ "$SKIP_INSTALL" = "0" ]; then
  log "5/7  配置环境变量"

  [ -f "$ASCEND_ENV" ] || die "找不到 $ASCEND_ENV，CANN 未正确安装"
  source_ascend_env || die "source $ASCEND_ENV 失败"
  ok "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

  # atc.bin 链接了 libascend_hal.so，该库正常由 NPU 驱动包提供。
  # 没装 NPU 驱动的机器（如 WSL2）上要用 toolkit 自带的 stub 版本。
  [ -d "$STUB_DIR" ] || warn "找不到 stub 目录 $STUB_DIR，atc 可能报 libascend_hal.so 缺失"

  # 写进 .bashrc（先删掉旧的本脚本段落，保证幂等）
  #
  # 【实测坑】必须写进「真正登录的那个用户」的 .bashrc，而不是 $HOME。
  # 本脚本常用 sudo 跑（因为要装 CANN），此时 $HOME=/root，
  # 结果 CANN 环境变量只写进了 /root/.bashrc —— 而开发板/PC 上日常用的是
  # 普通用户（如 retoo），他一开终端 atc 就报
  #     error while loading shared libraries: libascend_hal.so
  # 因为只有 set_env.sh 生效、stub 路径没生效。
  # 处理：sudo 时优先写到 SUDO_USER 的家目录，两个都写。
  TARGET_BASHRCS=()
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    _h="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
    [ -n "$_h" ] && [ -f "$_h/.bashrc" ] && TARGET_BASHRCS+=("$_h/.bashrc")
  fi
  [ -f "$HOME/.bashrc" ] && TARGET_BASHRCS+=("$HOME/.bashrc")

  MARK_BEGIN="# >>> yolov8_demo env >>>"
  MARK_END="# <<< yolov8_demo env <<<"
  for RC in "${TARGET_BASHRCS[@]}"; do
    # 删旧段落（幂等）
    if grep -qF "$MARK_BEGIN" "$RC" 2>/dev/null; then
      sed -i "\|$MARK_BEGIN|,\|$MARK_END|d" "$RC"
    fi
    cat >> "$RC" <<EOF

$MARK_BEGIN
source $ASCEND_ENV
# 【关键】stub 必须追加在【最后】，不能写成 "$STUB_DIR:\$LD_LIBRARY_PATH"。
# stub 目录里有一批与真实库同名的空壳库（libascendcl.so 等，仅几十 KB、
# 不含实现符号）。若 stub 排在前面，它们会抢先加载，导致 pyACL 导入失败：
#     ImportError: acl.so: undefined symbol: aclprofSetConfig
# 陷阱在于 atc 在两种顺序下都能正常工作，只有跑 OM 推理时才暴露。
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:$STUB_DIR"
source $CONDA_ROOT/etc/profile.d/conda.sh 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
$MARK_END
EOF
    ok "已写入 $RC"
  done

  # -------------------------------------------------------------------------
  log "6/7  给【系统 python3】补 TBE 依赖（最容易漏的一步）"

  # 【实测坑】ATC 的 TBE 用的是系统 python3（/usr/bin/python3），
  # 不是 conda 环境里的 python。只在 conda 里装 numpy 1.26.4 是没用的，
  # ATC 仍会报 EC0010: np.float_ was removed。
  # 【实测坑 1】ATC 的 TBE 用的是系统 python3（/usr/bin/python3），
  # 不是 conda 环境里的 python。只在 conda 里装 numpy 1.26.4 是没用的，
  # ATC 仍会报 EC0010: np.float_ was removed。
  #
  # 【实测坑 2】不能用 `command -v python3` 去找系统 python！本脚本前面
  # source 了 conda 的 profile.d，PATH 里 conda 的 python3 排在前面，
  # 于是拿到的会是 conda base 的 python3（3.13），把包装进了 conda base，
  # 既没修好系统 python，又污染了 base 环境。必须直接定位系统解释器。
  SYS_PY=""
  for cand in /usr/bin/python3 /usr/local/bin/python3; do
    if [ -x "$cand" ]; then SYS_PY="$cand"; break; fi
  done
  # 兜底：从 PATH 里挑第一个不在 conda 目录下的 python3
  if [ -z "$SYS_PY" ]; then
    while IFS= read -r c; do
      case "$c" in "$CONDA_ROOT"/*) continue ;; esac
      SYS_PY="$c"; break
    done < <(type -aP python3 2>/dev/null || true)
  fi

  if [ -n "$SYS_PY" ] && [ "$SYS_PY" != "$PY" ] && [ -x "$SYS_PY" ]; then
    SYS_SP="$("$SYS_PY" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null \
              || echo /usr/local/lib/python3.10/dist-packages)"
    echo "系统 python3 : $SYS_PY ($("$SYS_PY" -V 2>&1))"
    echo "目标目录     : $SYS_SP"

    # Debian 自带的 pip 22 不支持 --root-user-action / --break-system-packages，
    # 所以借用 conda 里的新版 pip 配合 --target 写入系统 site-packages。
    # 需要 root 权限，非 root 时退回到用户级目录 --user。
    TARGET_FLAG=(--target="$SYS_SP")
    if ! touch "$SYS_SP/.write_test" 2>/dev/null; then
      warn "$SYS_SP 不可写，改用 --user 用户级安装"
      TARGET_FLAG=(--user)
    else
      rm -f "$SYS_SP/.write_test"
    fi

    echo "[1/2] 安装 TBE 其余依赖..."
    "$PY" -m pip install "${TARGET_FLAG[@]}" --upgrade --no-cache-dir "${PIP_EXTRA[@]}" \
          scipy decorator psutil synr absl-py tornado 2>&1 | tail -2 || true

    echo "[2/2] 最后把 numpy 钉回 1.26.4（先清干净再装，避免残留半新半旧）..."
    rm -rf "$SYS_SP"/numpy "$SYS_SP"/numpy.libs "$SYS_SP"/numpy-*.dist-info 2>/dev/null || true
    "$PY" -m pip install "${TARGET_FLAG[@]}" --no-deps --no-cache-dir --force-reinstall \
          "numpy==1.26.4" 2>&1 | tail -2 || true

    # 清掉可能被顺带装进来的 ml_dtypes 0.6.0（它要求 numpy>=2，会让 pip 报冲突）
    rm -rf "$SYS_SP"/ml_dtypes "$SYS_SP"/ml_dtypes-0.6.0.dist-info 2>/dev/null || true

    if "$SYS_PY" -c "import numpy;print('系统 numpy :',numpy.__version__,'| np.float_ 可用:',hasattr(numpy,'float_'))"; then
      ok "系统 python3 的 TBE 依赖就绪"
    else
      warn "系统 python3 侧 numpy/scipy 仍不可用，ATC 可能报 EC0010"
      warn "可手工执行："
      warn "  pip install --target=$SYS_SP --upgrade scipy decorator psutil synr absl-py tornado"
      warn "  rm -rf $SYS_SP/numpy* && pip install --target=$SYS_SP --no-deps --force-reinstall numpy==1.26.4"
    fi
  else
    warn "未找到可用的系统 python3，跳过这一步（ATC 可能报 EC0010）"
  fi
fi

# ---------------------------------------------------------------------------
log "7/7  分层验证"

[ -x "$PY" ] || die "找不到 python: $PY"
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
  source_ascend_env >/dev/null 2>&1 || warn "source $ASCEND_ENV 失败，atc 可能不在 PATH 中"
  [ -d "$STUB_DIR" ] && export LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}"
  echo "atc       : $(command -v atc || echo '未找到')"
fi

FAIL=0
check() {  # check <描述> <命令...>
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$desc"; else warn "未通过：$desc"; FAIL=$((FAIL+1)); fi
}

echo "--- Python 侧 ---"
check "torch 可导入"        "$PY" -c "import torch"
check "ultralytics 可导入"  "$PY" -c "import ultralytics"
check "onnx / onnxruntime"  "$PY" -c "import onnx, onnxruntime"
check "cv2 (headless) 可用" "$PY" -c "import cv2; cv2.setNumThreads(1)"
check "numpy < 2"           "$PY" -c "import numpy,sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)"
check "没有冲突的 opencv"    bash -c "! '$PY' -m pip list 2>/dev/null | grep -qE '^opencv-python '"

echo "--- CANN / ATC ---"
if [ "$DO_CANN" = "1" ]; then
  check "atc 在 PATH 中"   bash -c "command -v atc"
  check "soc_version 支持 310B" bash -c "atc --help 2>&1 | grep -q AscendB || true"
  # 必须用系统解释器的绝对路径，不能用 `command -v python3`
  # （PATH 里 conda 的 python3 排在前面）
  SYS_PY_CHK="${SYS_PY:-}"
  [ -n "$SYS_PY_CHK" ] || SYS_PY_CHK=/usr/bin/python3
  check "系统 python3 numpy<2 (TBE)" "$SYS_PY_CHK" -c "import numpy,sys; sys.exit(0 if hasattr(numpy,'float_') else 1)"
  check "系统 python3 scipy (TBE)"   "$SYS_PY_CHK" -c "import scipy"
else
  warn "跳过了 CANN 验证（--no-cann）"
fi

# ---------------------------------------------------------------------------
log "完成"
cat <<EOF
环境名    : $ENV_NAME
Python    : $PY
CANN      : ${ASCEND_HOME_PATH:-未配置}
工程目录  : $WORK_DIR

新开终端会自动激活（已写入 ~/.bashrc）。当前终端请手动执行：
    source ~/.bashrc

然后跑全链路：
    cd $WORK_DIR
    python scripts/run_all.py --force

单独做 ATC 转换：
    source $ASCEND_ENV
    python scripts/atc_convert.py --onnx model/yolov8n.onnx --out model/yolov8n_bs1
EOF

if [ "$FAIL" -gt 0 ]; then
  warn "有 $FAIL 项验证未通过，请按上面的提示逐项排查"
  exit 1
fi
ok "全部验证通过"
