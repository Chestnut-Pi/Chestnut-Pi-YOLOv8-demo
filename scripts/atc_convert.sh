#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ATC 命令行封装（shell 版）。函数体与 scripts/atc_convert.py 完全一致，
# 适合不想用 Python 或需要在 CI / 脚本里直接调用的场景。
#
#   bash scripts/atc_convert.sh                       # 默认参数
#   bash scripts/atc_convert.sh model/yolov8n.onnx model/yolov8n_bs1
#   SOC_VERSION=Ascend310B1 bash scripts/atc_convert.sh
#
# 前置：必须先 source CANN 的环境变量脚本，否则 atc 会报
#       「host_env_os 参数无效」这种让人摸不着头脑的错误。
# ---------------------------------------------------------------------------
set -euo pipefail

ONNX="${1:-model/yolov8n.onnx}"
OUT="${2:-model/yolov8n_bs1}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"
INPUT_NAME="${INPUT_NAME:-images}"
INPUT_SHAPE="${INPUT_SHAPE:-1,3,640,640}"
INPUT_FORMAT="${INPUT_FORMAT:-NCHW}"
FRAMEWORK="${FRAMEWORK:-5}"          # 5 = ONNX
LOG_LEVEL="${LOG_LEVEL:-error}"

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

# ---- 配置 CANN 环境变量 ----
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -f "$ASCEND_ENV" ]; then
    # shellcheck disable=SC1090
    source "$ASCEND_ENV"
  else
    echo "【异常】找不到 $ASCEND_ENV，CANN 可能未安装。" >&2
    echo "       请先执行 env/setup_yolov8_demo.sh 安装 CANN。" >&2
    exit 2
  fi
fi

# atc.bin 链接了 libascend_hal.so，它随 toolkit 以 stub 形式提供。
# 在**没有装 NPU 驱动**的机器（如本 WSL2）上必须把 stub 目录加进搜索路径，
# 否则报 "error while loading shared libraries: libascend_hal.so"。
STUB_DIR="$ASCEND_HOME_PATH/runtime/lib64/stub"
if [ -d "$STUB_DIR" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$STUB_DIR:"*) ;;
    *) export LD_LIBRARY_PATH="$STUB_DIR:${LD_LIBRARY_PATH:-}" ;;
  esac
fi

if ! command -v atc >/dev/null 2>&1; then
  echo "【异常】PATH 中找不到 atc。请确认已 source $ASCEND_ENV" >&2
  exit 2
fi

[ -f "$ONNX" ] || { echo "【异常】找不到 ONNX 文件: $ONNX" >&2; exit 2; }

OUT_PREFIX="${OUT%.om}"
mkdir -p "$(dirname "$OUT_PREFIX")"

echo "=== ATC 转换 ==="
echo "ONNX        : $ONNX  ($(stat -c%s "$ONNX") 字节)"
echo "输出        : ${OUT_PREFIX}.om"
echo "soc_version : $SOC_VERSION"
echo "input_shape : ${INPUT_NAME}:${INPUT_SHAPE}"
echo "ASCEND_HOME : $ASCEND_HOME_PATH"

# kernel_meta 是 TBE 算子编译缓存。换过 soc_version 或 input_shape 之后，
# 残留缓存会干扰新编译，必须先删。
[ -d kernel_meta ] && rm -rf kernel_meta

start=$(date +%s)
atc --model="$ONNX" \
    --framework="$FRAMEWORK" \
    --output="$OUT_PREFIX" \
    --input_format="$INPUT_FORMAT" \
    --input_shape="${INPUT_NAME}:${INPUT_SHAPE}" \
    --soc_version="$SOC_VERSION" \
    --log="$LOG_LEVEL"
ret=$?
end=$(date +%s)

rm -rf kernel_meta 2>/dev/null || true
echo "ATC 退出码: $ret   耗时: $((end - start)) 秒"

if [ "$ret" -ne 0 ]; then
  echo "【异常】ATC 转换失败，常见原因：" >&2
  echo "  1) 未 source set_env.sh        -> host_env_os 无效" >&2
  echo "  2) TBE 依赖缺失                -> EC0010: No module named 'scipy'" >&2
  echo "  3) numpy 2.x 不兼容            -> EC0010: np.float_ was removed" >&2
  echo "  4) input_shape 节点名写错      -> E10016" >&2
  echo "  5) 漏写 soc_version            -> E10054" >&2
  exit "$ret"
fi

if [ ! -f "${OUT_PREFIX}.om" ]; then
  echo "【异常】ATC 返回 0 但没有生成 OM 文件" >&2
  exit 3
fi

SIZE=$(stat -c%s "${OUT_PREFIX}.om")
echo ""
echo "=== 转换成功 ==="
echo "OM 文件 : ${OUT_PREFIX}.om"
echo "大小    : $SIZE 字节 ($(awk "BEGIN{printf \"%.2f\", $SIZE/1000000}") MB)"
