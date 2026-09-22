"""YOLOv8 预处理 / 后处理公共实现。

pt、ONNX、OM 三层共用这里的同一套代码，保证三者的数值可以直接对比，
避免「因为前后处理不一致而误判精度」。

坐标约定：
    letterbox 先把原图等比缩放并填充到 640x640，推理输出因此落在 640x640 画布上。
    postprocess 负责把坐标反变换回原图，返回的框一律是原图像素坐标。
"""
import os

import cv2
import numpy as np

INPUT_SIZE = 640
NUM_CLASSES = 80
NUM_ANCHORS = 8400          # 640/8、640/16、640/32 三个尺度之和：40²+20²+10² = 8400
COCO_NAMES = None


# --------------------------------------------------------------------------
# 预处理
# --------------------------------------------------------------------------
def letterbox(img, size=INPUT_SIZE, color=114):
    """等比例缩放 + 灰边填充，返回 (canvas, ratio, dw, dh)。

    color=114 是 YOLO 官方的填充灰度。padding 用整数除法，保证 cv2.resize
    的取整方向与 ultralytics 内部一致，否则会出现 ±1 像素的偏差。
    """
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def preprocess(img, size=INPUT_SIZE):
    """BGR HWC uint8 -> RGB CHW float32 [0,1]，返回 (1,3,size,size)。"""
    canvas, r, dw, dh = letterbox(img, size)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob[None, ...]), r, dw, dh


# --------------------------------------------------------------------------
# 后处理
# --------------------------------------------------------------------------
def xywh2xyxy(x):
    """中心点+宽高 -> 左上右下。"""
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_thres):
    """纯 numpy 实现的 NMS，避免引入 torchvision / torchvision.ops 依赖。"""
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size > 0:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idx = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


def postprocess(pred, r, dw, dh, w0, h0, conf_thres=0.25, iou_thres=0.45):
    """YOLOv8 输出 (1,84,8400) -> 原图坐标检测框，返回 (N,6) [x1,y1,x2,y2,conf,cls]。

    注意：导出的 ONNX/OM 图里已经包含 sigmoid，所以这里不再做激活。
    """
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    if p.size != (NUM_CLASSES + 4) * NUM_ANCHORS:
        raise ValueError(
            f"输出元素数异常: {p.size}（期望 {(NUM_CLASSES + 4) * NUM_ANCHORS}）。"
            "请先运行 scripts/check_onnx.py 或 check_om_out.py 确认输出布局与轴序。"
        )
    p = p.reshape(NUM_CLASSES + 4, NUM_ANCHORS)

    boxes_xywh = p[:4, :].T          # (8400,4)，已是 640x640 letterbox 坐标
    scores_all = p[4:, :]            # (80,8400)
    class_ids = scores_all.argmax(axis=0)
    confs = scores_all.max(axis=0)

    mask = confs > conf_thres
    if not np.any(mask):
        return np.zeros((0, 6), dtype=np.float32)

    boxes = xywh2xyxy(boxes_xywh[mask])
    confs, class_ids = confs[mask], class_ids[mask]

    # 按类别分别做 NMS，避免不同类别的框互相抑制
    keep_all = []
    for cid in np.unique(class_ids):
        sel = np.where(class_ids == cid)[0]
        keep = nms(boxes[sel], confs[sel], iou_thres)
        keep_all.append(sel[keep])
    keep_all = np.concatenate(keep_all) if keep_all else np.array([], dtype=np.int64)

    boxes, confs, class_ids = boxes[keep_all], confs[keep_all], class_ids[keep_all]

    # letterbox 画布坐标 -> 原图坐标
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    return np.concatenate([boxes, confs[:, None],
                           class_ids[:, None].astype(np.float32)], axis=1)


# --------------------------------------------------------------------------
# 可视化 / 工具
# --------------------------------------------------------------------------
def load_names(path=None):
    """加载 COCO 类别名。返回 list 或 None（找不到文件时）。"""
    global COCO_NAMES
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "data", "coco_names.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            COCO_NAMES = [ln.strip() for ln in f if ln.strip()]
    return COCO_NAMES


def class_name(cid):
    if COCO_NAMES and 0 <= int(cid) < len(COCO_NAMES):
        return COCO_NAMES[int(cid)]
    return f"class_{int(cid)}"


def draw(img, dets, names=True):
    """在图上画框，返回新图（不修改入参）。"""
    out = img.copy()
    for x1, y1, x2, y2, conf, cid in dets:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        color = COLORS[int(cid) % len(COLORS)]
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{class_name(cid)} {conf:.2f}" if names else f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], max(0, p1[1] - th - 4)),
                      (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(out, label, (p1[0] + 1, max(th, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
]


def print_dets(dets, tag=""):
    """打印检测结果；返回检测框数量。"""
    n = len(dets)
    print(f"{tag}检测到 {n} 个目标:")
    for x1, y1, x2, y2, conf, cid in dets:
        print(f"  {class_name(cid):<12s} conf={conf:.4f} "
              f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    return n


def read_image(path):
    """读图，失败时给出明确报错。支持中文路径。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到图片: {path}")
    return img


def write_image(path, img):
    """写图，支持中文路径（cv2.imwrite 在 Windows 中文路径下会失败）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"编码图片失败: {path}")
    buf.tofile(path)
