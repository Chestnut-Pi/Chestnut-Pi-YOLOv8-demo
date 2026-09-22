"""步骤 6：在昇腾 310B 上用 pyACL 跑 OM 离线模型（纯 Python，不依赖 ais_bench）。

这个脚本要在**开发板上**跑，不是在 PC/WSL2 上跑（PC 上没有 NPU 设备节点）。

用法:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 scripts/infer_om.py --model model/yolov8n_bs1.om --image data/images/sample.jpg

关于 pyACL 的返回值约定（在 CANN 7.0.RC1 的栗子派 310B 上实测）：
    acl.init() / acl.rt.set_device()            -> 只返回 ret
    acl.rt.malloc() / acl.mdl.load_from_file()  -> 返回 (句柄, ret) 元组
    acl.mdl.create_desc() / create_dataset() /
    acl.create_data_buffer()                    -> 只返回句柄（整数指针）
    acl.rt.memcpy()                             -> 只返回 ret
    往 dataset 里挂 buffer 用 acl.mdl.add_dataset_buffer(ds, buf)，
    不是 ds.add(buf) —— 后者在本版本上会 AttributeError。

    不同 CANN 版本的返回值约定可能不同，若报 TypeError（cannot unpack
    non-iterable / too many values），先按上面的表核对一遍。
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_common import (  # noqa: E402
    draw, load_names, postprocess, preprocess, print_dets, read_image, write_image,
)

try:
    import acl
except ImportError:
    sys.exit(
        "导入 acl 失败。请先执行：\n"
        "  source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        "再运行本脚本。"
    )

# ---- pyACL 常量 ----
ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# ---- ACL 数据类型 -> numpy 类型 ----
ACL_DTYPE = {
    0: np.float32,    # ACL_FLOAT
    1: np.float16,    # ACL_FLOAT16
    2: np.int8,       # ACL_INT8
    3: np.int32,      # ACL_INT32
    4: np.uint8,      # ACL_UINT8
    6: np.uint16,     # ACL_UINT16
    8: np.int64,      # ACL_INT64
    9: np.uint64,     # ACL_UINT64
    10: np.float64,   # ACL_DOUBLE
    11: np.bool_,     # ACL_BOOL
}
ACL_DTYPE_NAME = {0: "FP32", 1: "FP16", 2: "INT8", 3: "INT32", 4: "UINT8",
                  6: "UINT16", 8: "INT64", 9: "UINT64", 10: "DOUBLE", 11: "BOOL"}


def _host_ptr(buf):
    """取 ctypes 缓冲区的主机地址（memcpy 需要整数地址）。"""
    return ctypes.addressof(buf)


class AscendModel:
    """封装一个 OM 模型的加载 / 推理 / 释放。

    与官方样例一致：设备侧内存和 dataset 只在初始化时建一次，循环推理复用，
    避免每次 execute 都 malloc/free 带来的额外开销。
    """

    def __init__(self, om_path, device_id=0):
        self.om_path = om_path
        self.device_id = device_id
        self._released = False

        self._check(acl.init(), "acl.init 失败")
        self._check(acl.rt.set_device(device_id),
                    f"acl.rt.set_device({device_id}) 失败，检查 /dev/davinci0 权限")

        self.model_id, ret = acl.mdl.load_from_file(om_path)
        self._check(ret, f"加载模型失败({om_path})，确认 OM 的 soc_version 与本板芯片一致")

        self.desc = acl.mdl.create_desc()
        self._check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc 失败")

        self.num_inputs = acl.mdl.get_num_inputs(self.desc)
        self.num_outputs = acl.mdl.get_num_outputs(self.desc)

        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i)
                         for i in range(self.num_inputs)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i)
                          for i in range(self.num_outputs)]
        self.in_dtypes = [acl.mdl.get_input_data_type(self.desc, i)
                          for i in range(self.num_inputs)]
        self.out_dtypes = [acl.mdl.get_output_data_type(self.desc, i)
                           for i in range(self.num_outputs)]

        # 输入/输出的 shape（用于校验和后处理）
        self.in_dims = []
        for i in range(self.num_inputs):
            d, _ = acl.mdl.get_input_dims(self.desc, i)
            self.in_dims.append(tuple(d["dims"]))
        self.out_dims = []
        for i in range(self.num_outputs):
            d, _ = acl.mdl.get_output_dims(self.desc, i)
            self.out_dims.append(tuple(d["dims"]))

        print("=== 模型信息 ===")
        print(f"文件      : {om_path}")
        print(f"输入个数  : {self.num_inputs}   输出个数: {self.num_outputs}")
        for i in range(self.num_inputs):
            print(f"  输入[{i}] shape={self.in_dims[i]} "
                  f"dtype={ACL_DTYPE_NAME.get(self.in_dtypes[i], self.in_dtypes[i])} "
                  f"bytes={self.in_sizes[i]}")
        for i in range(self.num_outputs):
            print(f"  输出[{i}] shape={self.out_dims[i]} "
                  f"dtype={ACL_DTYPE_NAME.get(self.out_dtypes[i], self.out_dtypes[i])} "
                  f"bytes={self.out_sizes[i]}")

        self._build_datasets()

    @staticmethod
    def _check(ret, msg):
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{msg} (ret={ret})")

    # ---------------- 资源 ----------------
    def _build_datasets(self):
        """申请设备内存并组装输入/输出 dataset（只做一次，循环复用）。"""
        self.dev_in = []
        self.in_ds = acl.mdl.create_dataset()
        for i in range(self.num_inputs):
            ptr, ret = acl.rt.malloc(self.in_sizes[i], ACL_MEM_MALLOC_HUGE_FIRST)
            self._check(ret, "为输入申请设备内存失败")
            self.dev_in.append(ptr)
            buf = acl.create_data_buffer(ptr, self.in_sizes[i])
            _, ret = acl.mdl.add_dataset_buffer(self.in_ds, buf)
            self._check(ret, "把输入 buffer 挂到 dataset 失败")

        self.dev_out = []
        self.out_ds = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            ptr, ret = acl.rt.malloc(self.out_sizes[i], ACL_MEM_MALLOC_HUGE_FIRST)
            self._check(ret, "为输出申请设备内存失败")
            self.dev_out.append(ptr)
            buf = acl.create_data_buffer(ptr, self.out_sizes[i])
            _, ret = acl.mdl.add_dataset_buffer(self.out_ds, buf)
            self._check(ret, "把输出 buffer 挂到 dataset 失败")

        # 主机侧接收缓冲，同样复用
        self.host_out = [ctypes.create_string_buffer(s) for s in self.out_sizes]

    def _destroy_datasets(self):
        for ds in (getattr(self, "in_ds", None), getattr(self, "out_ds", None)):
            if not ds:
                continue
            n = acl.mdl.get_dataset_num_buffers(ds)
            for i in range(n):
                b = acl.mdl.get_dataset_buffer(ds, i)
                if b:
                    acl.destroy_data_buffer(b)
            acl.mdl.destroy_dataset(ds)
        for ptr in getattr(self, "dev_in", []):
            acl.rt.free(ptr)
        for ptr in getattr(self, "dev_out", []):
            acl.rt.free(ptr)
        self.dev_in, self.dev_out = [], []

    # ---------------- 推理 ----------------
    def infer(self, input_np):
        """input_np: numpy 数组（FP32）→ 返回 ([numpy...], [字节数...])"""
        np_in_dt = ACL_DTYPE.get(self.in_dtypes[0], np.float32)
        data = np.ascontiguousarray(input_np, dtype=np_in_dt)

        if data.nbytes != self.in_sizes[0]:
            raise RuntimeError(
                f"输入字节数不匹配：模型期望 {self.in_sizes[0]}，实际 {data.nbytes}。\n"
                f"  模型输入 shape={self.in_dims[0]} "
                f"dtype={ACL_DTYPE_NAME.get(self.in_dtypes[0])}\n"
                f"  预处理输出 shape={data.shape} dtype={data.dtype}\n"
                f"  请检查 ATC 的 --input_shape 与预处理是否一致"
            )

        host_in = ctypes.create_string_buffer(data.tobytes(), data.nbytes)
        self._check(
            acl.rt.memcpy(self.dev_in[0], self.in_sizes[0],
                          _host_ptr(host_in), data.nbytes,
                          ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D memcpy 失败")

        self._check(acl.mdl.execute(self.model_id, self.in_ds, self.out_ds),
                    "acl.mdl.execute 失败")

        outputs = []
        for i, size in enumerate(self.out_sizes):
            host = self.host_out[i]
            self._check(
                acl.rt.memcpy(_host_ptr(host), size,
                              self.dev_out[i], size,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                "D2H memcpy 失败")
            dt = ACL_DTYPE.get(self.out_dtypes[i], np.float32)
            outputs.append(np.frombuffer(host.raw, dtype=dt).astype(np.float32).copy())

        return outputs, list(self.out_sizes)

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            self._destroy_datasets()
            acl.mdl.unload(self.model_id)
            acl.mdl.destroy_desc(self.desc)
        except Exception as e:                      # noqa: BLE001
            print("释放模型资源时告警:", e)


def acl_finalize(device_id=0):
    """进程退出前调用一次，释放 ACL 上下文。"""
    try:
        acl.rt.reset_device(device_id)
        acl.finalize()
    except Exception as e:                          # noqa: BLE001
        print("释放 ACL 上下文时告警:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/yolov8n_bs1.om")
    ap.add_argument("--image", default="data/images/sample.jpg")
    ap.add_argument("--out", default="results/om_result.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--loop", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    img = read_image(args.image)
    h0, w0 = img.shape[:2]

    blob, r, dw, dh = preprocess(img)
    print(f"\n=== 预处理 ===\n原图 {w0}x{h0}  blob {blob.shape} {blob.dtype}  "
          f"letterbox ratio={r:.6f} pad=({dw},{dh})")

    load_names()
    model = AscendModel(args.model, args.device)
    try:
        # 预热：首次 execute 含算子加载开销，不计入性能统计
        for _ in range(args.warmup):
            model.infer(blob)

        t0 = time.perf_counter()
        for _ in range(args.loop):
            outputs, sizes = model.infer(blob)
        t1 = time.perf_counter()
        pure_ms = (t1 - t0) / args.loop * 1000

        pred = outputs[0]
        print(f"\n=== 模型原始输出 ===\n元素数={pred.size}  dtype={pred.dtype}  "
              f"min/max={pred.min():.4f}/{pred.max():.4f}")

        dets = postprocess(pred, r, dw, dh, w0, h0, args.conf, args.iou)
        print(f"\n=== 检测结果（conf>{args.conf} iou>{args.iou}）===")
        print_dets(dets, tag="")

        write_image(args.out, draw(img, dets))
        print(f"\n[可视化] 已保存 {args.out}")

        os.makedirs("logs", exist_ok=True)
        np.save("logs/om_raw_output.npy", pred)
        np.save("logs/om_dets.npy", dets)
        with open("logs/om_perf.txt", "w", encoding="utf-8") as f:
            f.write(f"pure_infer_ms_avg={pure_ms:.3f}\n")
            f.write(f"warmup={args.warmup}\nloop={args.loop}\n")
            f.write(f"input_shape={blob.shape}\ninput_dtype={blob.dtype}\n")
        print(f"\n>>> [性能] 单次推理平均耗时 {pure_ms:.3f} ms  ({1000 / pure_ms:.2f} FPS)")
        print("    该数值含每次推理的 H2D/D2H 拷贝；设备内存与 dataset 已复用。")
    finally:
        model.release()
        acl_finalize(args.device)


if __name__ == "__main__":
    main()
