# 设置环境变量，限制线程数为1，python的并行并不是真正的并行，因为GIL锁
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import copy
import sys

sys.path.insert(0, './yolov5')

import argparse
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS, letterbox
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from tracker.utils.parser import get_config
from tracker.deep_sort import DeepSort
from viz_count import LineCrossingCounter, render_frame

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

COUNTER = LineCrossingCounter(line_ratio=0.5)

# 全局缓存：避免每次点「开始」都重新加载 YOLO（很慢）
_YOLO_CACHE = {}


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_xyxy(boxes, scores, iou_thres: float = 0.5) -> list[int]:
    """同类重叠框再压一遍。"""
    if len(boxes) == 0:
        return []
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou_xyxy(boxes[i], boxes[j]) < iou_thres]
    return keep


def _inter_over_smaller(a, b) -> float:
    """交集 / 较小框面积；用于「大框套小框」去重。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return float(inter / smaller) if smaller > 0 else 0.0


def _center_close(a, b, rel: float = 0.35) -> bool:
    """两框中心是否过近（相对平均边长）。"""
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    aw = max(1.0, a[2] - a[0])
    ah = max(1.0, a[3] - a[1])
    bw = max(1.0, b[2] - b[0])
    bh = max(1.0, b[3] - b[1])
    scale = 0.5 * ((aw + bw) * 0.5 + (ah + bh) * 0.5)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return dist < scale * rel


def _filter_dets(boxes, scores, clss, min_side: float = 12.0, nms_iou: float = 0.35,
                 frame_wh=None, custom_2cls: bool = True, person_min_conf: float = 0.0):
    """去掉过小框 + 强力去重（防一车多框）。"""
    if len(boxes) == 0:
        return boxes, scores, clss
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    scores = np.asarray(scores, dtype=np.float32).copy()
    clss = np.asarray(clss, dtype=np.int32).copy()
    fw = float(frame_wh[0]) if frame_wh else 1.0
    fh = float(frame_wh[1]) if frame_wh else 1.0
    frame_area = max(fw * fh, 1.0)

    if custom_2cls:
        for i in range(len(clss)):
            x1, y1, x2, y2 = boxes[i]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            ar = bw / bh
            area_ratio = (bw * bh) / frame_area
            if int(clss[i]) == 0:
                if person_min_conf > 0 and float(scores[i]) < person_min_conf:
                    scores[i] = -1.0
                    continue
                if ar >= 1.0 and area_ratio >= 0.04:
                    clss[i] = 1
                elif area_ratio >= 0.08:
                    clss[i] = 1
                elif ar <= 0.28 and bh > fh * 0.25:
                    scores[i] = -1.0
            elif int(clss[i]) == 1:
                if ar <= 0.35 and bh > fh * 0.3:
                    scores[i] = -1.0
    else:
        for i in range(len(clss)):
            if int(clss[i]) == 0 and person_min_conf > 0 and float(scores[i]) < person_min_conf:
                scores[i] = -1.0

    valid = scores >= 0
    boxes, scores, clss = boxes[valid], scores[valid], clss[valid]
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    # 1) 按类 NMS：车严（防碎框），人松（并排人群别并掉）
    keep_all = []
    for c in np.unique(clss):
        idx = np.where(clss == c)[0]
        ok = [
            k for k, box in enumerate(boxes[idx])
            if (box[2] - box[0]) >= min_side and (box[3] - box[1]) >= min_side
        ]
        if not ok:
            continue
        idx = idx[ok]
        # car 严去重；person 松（并排人群）
        thr = 0.30 if int(c) == 1 else 0.55
        local = _nms_xyxy(boxes[idx].tolist(), scores[idx].tolist(), iou_thres=thr)
        keep_all.extend(idx[local].tolist())
    if not keep_all:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    keep_all = sorted(keep_all)
    boxes, scores, clss = boxes[keep_all], scores[keep_all], clss[keep_all]

    # 2) 全局再压：车严；人只去掉几乎完全重叠的碎框（并排人群要保留）
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: list[int] = []
    for i in order:
        bi = boxes[i]
        ai = float((bi[2] - bi[0]) * (bi[3] - bi[1]))
        ci = int(clss[i])
        drop_i = False
        for j in keep:
            bj = boxes[j]
            aj = float((bj[2] - bj[0]) * (bj[3] - bj[1]))
            cj = int(clss[j])
            iou = _iou_xyxy(bi, bj)
            ios = _inter_over_smaller(bi, bj)
            same = ci == cj

            if same and ci == 0:
                # 人：只有高度重叠才去重（站一起的人 IoU 通常不高）
                if iou < 0.55 and ios < 0.70:
                    continue
            elif same and ci == 1:
                # 车：严去重
                close = _center_close(bi, bj, rel=0.55)
                if iou < 0.2 and ios < 0.35 and not close:
                    continue
            else:
                # 跨类：主要压「人框盖在车上」
                close = _center_close(bi, bj, rel=0.35)
                if iou < 0.25 and ios < 0.45 and not close:
                    continue
                if ci == 1 and cj == 0 and ai > aj * 1.05:
                    keep.remove(j)
                    break
                if ci == 0 and cj == 1:
                    drop_i = True
                    break
                # 其它跨类：分高者已在 keep
                if iou >= 0.25 or ios >= 0.45 or close:
                    drop_i = True
                    break
                continue

            # 同类需要抑制
            drop_i = True
            break
        if not drop_i:
            keep.append(i)
    keep = sorted(keep)
    return boxes[keep], scores[keep], clss[keep]


def _dedup_tracks(tracks: np.ndarray, iou_thres: float = 0.3) -> np.ndarray:
    """追踪后再压：车严、人松。"""
    if tracks is None or len(tracks) == 0:
        return np.empty((0, 6), dtype=np.float32)
    tracks = np.asarray(tracks, dtype=np.float32)
    n = len(tracks)
    areas = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
    order = sorted(range(n), key=lambda i: float(areas[i]), reverse=True)
    keep: list[int] = []
    for i in order:
        suppressed = False
        replace = None
        bi = tracks[i][:4]
        ci = int(tracks[i][5])
        for j in list(keep):
            bj = tracks[j][:4]
            cj = int(tracks[j][5])
            iou = _iou_xyxy(bi, bj)
            ios = _inter_over_smaller(bi, bj)
            if ci == 0 and cj == 0:
                # 人并排：不要因中心近就合并
                if iou < 0.55 and ios < 0.70:
                    continue
                suppressed = True
                break
            if ci == 1 and cj == 1:
                close = _center_close(bi, bj, rel=0.55)
                if iou < iou_thres and ios < 0.35 and not close:
                    continue
                suppressed = True
                break
            # 跨类
            close = _center_close(bi, bj, rel=0.35)
            if iou < 0.25 and ios < 0.45 and not close:
                continue
            if ci == 1 and cj == 0:
                replace = j
                break
            if ci == 0 and cj == 1:
                suppressed = True
                break
            suppressed = True
            break
        if suppressed:
            continue
        if replace is not None:
            keep.remove(replace)
        keep.append(i)
    return tracks[keep]


class SimpleIoUTracker:
    """无 ReID 的轻量追踪（比 DeepSort 快一个数量级，适合 CPU / Gradio）。"""

    def __init__(self, iou_thresh: float = 0.1, max_age: int = 60, dist_thresh: float = 0.35):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.dist_thresh = dist_thresh  # 中心点距离相对画面对角线（放宽，减少换号）
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def update(self, boxes_xyxy, class_ids, frame_wh=None) -> np.ndarray:
        boxes = [list(map(float, b[:4])) for b in boxes_xyxy]
        clss = [int(c) for c in class_ids]
        diag = 1.0
        if frame_wh is not None:
            diag = float((frame_wh[0] ** 2 + frame_wh[1] ** 2) ** 0.5) or 1.0

        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1

        matched_det = set()
        matched_tid = set()
        pairs = []
        for di, box in enumerate(boxes):
            cx, cy = self._center(box)
            for tid, tr in self.tracks.items():
                if tr["cls"] != clss[di]:
                    continue
                iou = _iou_xyxy(box, tr["box"])
                tx, ty = self._center(tr["box"])
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / diag
                # IoU 优先；隔帧位移大时用中心距离兜底（阈值已放宽）
                if iou >= self.iou_thresh:
                    score = 1.0 + iou
                elif dist <= self.dist_thresh:
                    score = 1.0 - dist
                else:
                    score = -1.0
                if score >= 0:
                    pairs.append((score, di, tid))
        pairs.sort(reverse=True)
        for score, di, tid in pairs:
            if di in matched_det or tid in matched_tid:
                continue
            self.tracks[tid] = {"box": boxes[di], "cls": clss[di], "age": 0}
            matched_det.add(di)
            matched_tid.add(tid)

        for di, box in enumerate(boxes):
            if di in matched_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": box, "cls": clss[di], "age": 0}

        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]

        outs = []
        for tid, tr in self.tracks.items():
            if tr["age"] > 0:
                continue
            x1, y1, x2, y2 = tr["box"]
            outs.append([x1, y1, x2, y2, tid, tr["cls"]])
        return np.asarray(outs, dtype=np.float32) if outs else np.empty((0, 6), dtype=np.float32)


def _to_gradio_file(path: str | None) -> str | None:
    """把结果视频拷到系统临时目录，避免 Gradio cache 权限错误。"""
    if not path:
        return None
    src = Path(path)
    if not src.is_file():
        return None
    import tempfile
    import shutil as _shutil

    dst = Path(tempfile.gettempdir()) / f"vc_{src.name}"
    _shutil.copy2(src, dst)
    return str(dst)


def _get_cached_yolo(weight, device, dnn=False):
    key = (str(weight), str(device), bool(dnn))
    if key not in _YOLO_CACHE:
        model = DetectMultiBackend(weight, device=device, dnn=dnn)
        stride, names, pt = model.stride, model.names, model.pt
        model.warmup(imgsz=(1, 3, 320, 320))
        _YOLO_CACHE[key] = (model, stride, names, pt)
        LOGGER.info(f"YOLO cached: {key[0]}")
    return _YOLO_CACHE[key]


def detect_fast(opt):
    """Gradio 快速路径：跳过 DeepSort ReID + 跳帧解码 + 小分辨率 + 模型缓存。"""
    source = opt.source
    weight = opt.yolo_model[0] if isinstance(opt.yolo_model, (list, tuple)) else opt.yolo_model
    device = select_device(opt.device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    frame_stride = max(1, int(getattr(opt, "frame_stride", 8)))
    max_frames = int(getattr(opt, "max_frames", 40))
    imgsz = int(getattr(opt, "imgsz", [320])[0] if isinstance(opt.imgsz, list) else getattr(opt, "imgsz", 320))
    imgsz = check_img_size(imgsz, s=32)

    yield None, (
        f"快速模式：跳过 DeepSort。\n"
        f"隔帧={frame_stride}，最多={max_frames or '不限'}，输入={imgsz}，设备={device}\n"
        f"正在加载/复用模型…"
    ), None

    model, stride, names, pt = _get_cached_yolo(weight, device, getattr(opt, "dnn", False))
    if isinstance(names, dict):
        # ok
        pass

    COUNTER.reset()
    # 放宽关联：隔帧后同一人不要反复开新 ID
    tracker = SimpleIoUTracker(
        iou_thresh=0.08,
        max_age=max(80, frame_stride * 8),
        dist_thresh=0.4,
    )

    # 始终写到仓库 outputs/，不要用权重绝对路径当目录名
    repo_root = Path(__file__).resolve().parents[1]
    save_dir = repo_root / "outputs" / "runs" / "track" / "gradio_fast"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(save_dir / f"{Path(source).stem}_fast.mp4")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        yield None, f"无法打开视频：{source}", None
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # 预览/写盘缩小宽度，4K 全分辨率写盘极慢
    out_w = min(960, w0 if w0 > 0 else 960)
    scale = out_w / w0 if w0 > 0 else 1.0
    out_h = int((h0 if h0 > 0 else 540) * scale)
    # 播放更慢更长：固定约 6fps，每帧停留更久（不要用原视频 fps/stride，否则只有几秒）
    playback_fps = float(getattr(opt, "playback_fps", 6.0))
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, playback_fps),
        (out_w, out_h),
    )

    processed = 0
    frame_idx = -1
    last_rgb = None
    t_all0 = time.time()

    while True:
        # 跳帧：grab 丢弃，不 decode，比逐帧读再 continue 快很多
        for _ in range(frame_stride - 1):
            if not cap.grab():
                break
            frame_idx += 1
        ok, im0 = cap.read()
        if not ok:
            break
        frame_idx += 1

        if max_frames > 0 and processed >= max_frames:
            break
        processed += 1

        # 推理用缩小图
        im_rs = cv2.resize(im0, (out_w, out_h), interpolation=cv2.INTER_AREA) if scale < 0.999 else im0
        img, _, _ = letterbox(im_rs, imgsz, stride=stride, auto=pt)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det
        )

        det = pred[0]
        tracks = np.empty((0, 6), dtype=np.float32)
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im_rs.shape).round()
            boxes = det[:, :4].detach().cpu().numpy()
            scores = det[:, 4].detach().cpu().numpy()
            clss = det[:, 5].detach().cpu().numpy().astype(np.int32)
            # 车在远景可能较小；并做形态纠错（宽大框勿标成人）
            custom = "phase_d" in str(weight).replace("\\", "/").lower()
            person_min = float(getattr(opt, "person_min_conf", 0.0) or 0.0)
            boxes, scores, clss = _filter_dets(
                boxes, scores, clss, min_side=8.0, nms_iou=0.30,
                frame_wh=(out_w, out_h), custom_2cls=custom, person_min_conf=person_min,
            )
            if len(boxes):
                tracks = tracker.update(boxes, clss, frame_wh=(out_w, out_h))
                tracks = _dedup_tracks(tracks, iou_thres=0.28)

        rendered = render_frame(im_rs, tracks, names, COUNTER, colors, draw_line=False)
        writer.write(rendered)
        last_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - t_all0
        fps_eff = processed / elapsed if elapsed > 0 else 0
        summary = (
            f"快速模式运行中\n"
            f"已处理 {processed}"
            + (f"/{max_frames}" if max_frames > 0 else "")
            + f" 帧 | 源进度 #{frame_idx}"
            + (f"/{total}" if total else "")
            + f" | 约 {fps_eff:.2f} 处理帧/秒\n"
            + COUNTER.summary_text(COUNTER._last_person_now, COUNTER._last_vehicle_now)
            + "\n（画面左上数字=当前帧人数/车数）"
        )
        yield last_rgb, summary, None

    cap.release()
    writer.release()
    elapsed = time.time() - t_all0
    done = (
        f"完成（快速模式）。耗时 {elapsed:.1f}s，处理 {processed} 帧。\n"
        f"{COUNTER.summary_text(COUNTER._last_person_now, COUNTER._last_vehicle_now)}\n\n"
        f"视频已保存：{out_path}"
    )
    yield last_rgb, done, _to_gradio_file(out_path)


def detect(opt, grstatus=False):  # gradio可视化时需要加一个参数
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.imgsz, opt.evaluate, opt.half, \
            opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Gradio / 无头：绝不弹窗；并强制写出处理后视频
    preview_every = int(getattr(opt, 'preview_every', 5))
    frame_stride = max(1, int(getattr(opt, 'frame_stride', 1)))
    max_frames = int(getattr(opt, 'max_frames', 0))  # 0=不限制
    last_saved_video = None
    last_det_rgb = None
    processed = 0
    if grstatus:
        show_vid = False
        save_vid = True
        # 先立刻回传状态，避免界面一直转圈
        yield None, "正在加载模型与视频，请稍候…", None

    COUNTER.reset()
    COUNTER.line_ratio = float(getattr(opt, 'line_ratio', 0.5))

    # Initialize
    device = select_device(opt.device)  # 设备选择 cpu还是gpu
    half &= device.type != 'cpu'  # 半精度

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:  # 是否需要评估，在测试整个pipeline时可以使用
        if os.path.exists(out):  # 判断是否存在输出文件夹，有就删掉
            pass
            shutil.rmtree(out)  # 删掉输出文件夹
        os.makedirs(out)  # # 新建输出文件夹

    # Directories（必须用 stem，不能用绝对路径字符串，否则会写到 weights/ 下）
    if type(yolo_model) is str:
        exp_name = Path(yolo_model).stem
    elif type(yolo_model) is list and len(yolo_model) == 1:
        exp_name = Path(yolo_model[0]).stem
    else:
        exp_name = "ensemble"
    exp_name = exp_name + "_" + Path(deep_sort_model).stem
    # project 固定到仓库 outputs，避免相对路径跑飞
    project_dir = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "track"
    save_dir = increment_path(project_dir / exp_name, exist_ok=exist_ok)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 根据组装的路径创建文件夹

    # 加载yolo模型
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图片的输入尺寸

    # 半精度
    half &= pt and device.type != 'cpu'  # 半精度只支持cuda，也就是GPU运行时
    if pt:
        model.model.half() if half else model.model.float()

    # 配置数据加载器
    vid_path, vid_writer = None, None
    # 检查环境是否支持结果实时显示，也就是Matplotlib和opencv
    if show_vid:
        show_vid = check_imshow()

    # 数据加载器
    if webcam:  # 如果输入是摄像头，也就是source 为0
        show_vid = check_imshow() if not grstatus else False
        cudnn.benchmark = True  # 使用cudnn去批量加速图片推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)  # 因为是摄像头，所以走加载视频流的方式
        nr_sources = len(dataset)  # 获取数据集的长度
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # 如果不是摄像头，就走加载图片
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources  # 结果列表的创建初始化

    # 根据配置文件初始化deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # 根据输入的数量来创建deepsort的数量，达到并行处理的目的
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )  # deepsort模型加载
        )
    outputs = [None] * nr_sources  # 输出的数量要和输入相对应

    # 获取类别的名称，并分配相应的颜色
    names = model.module.names if hasattr(model, 'module') else model.names

    # 运行追踪器，model表示Yolo模型，deepsort_list里装的是deepsort模型
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # 预热阶段，模型加载起来都要先预热，相当于快速推理，将模型加载到内存里，以便后续使用
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):  # 开始循环遍历数据集中的数据
        # Gradio 加速：隔帧处理，并可限制最多处理帧数（否则 CPU 上像一直加载）
        if frame_idx % frame_stride != 0:
            continue
        if max_frames > 0 and processed >= max_frames:
            break
        processed += 1

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)  # 将数据加载到GPU或者CPU上
        im = im.half() if half else im.float()  # uint8 to fp16/32  半精度的数据类型转换
        im /= 255.0  # 0 - 255 to 0.0 - 1.0  #归一化，加快计算
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # 预测
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)  # yolo预测，获取类别，置信度，边框
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS非极大值抑制
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # 检测结果后处理
        for i, det in enumerate(pred):  # 循环遍历检测结果
            seen += 1
            if webcam:  # 摄像头，if else过程是用来获取保存路径名称
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # 数据结果txt文件名称及位置。
            s += '%gx%g ' % im.shape[2:]
            imc = im0.copy() if save_crop else im0  # save_crop，保存目标裁切结果
            imo = copy.deepcopy(im0)
            annotator = Annotator(im0, line_width=2, pil=not ascii)  # 标签处理

            w, h = im0.shape[1], im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                boxes = det[:, :4].detach().cpu().numpy()
                confs_np = det[:, 4].detach().cpu().numpy()
                clss_np = det[:, 5].detach().cpu().numpy().astype(np.int32)
                custom = "phase_d" in str(yolo_model).replace("\\", "/").lower()
                person_min = float(getattr(opt, "person_min_conf", 0.0) or 0.0)
                boxes, confs_np, clss_np = _filter_dets(
                    boxes, confs_np, clss_np, min_side=10.0, nms_iou=0.45,
                    frame_wh=(w, h), custom_2cls=custom, person_min_conf=person_min,
                )

                # Print results
                for c in np.unique(clss_np) if len(clss_np) else []:
                    n = int((clss_np == c).sum())
                    try:
                        nm = names[int(c)]
                    except Exception:
                        nm = str(c)
                    s += f"{n} {nm}{'s' * (n > 1)}, "

                if len(boxes):
                    xywhs = xyxy2xywh(torch.from_numpy(boxes))
                    confs = torch.from_numpy(confs_np)
                    clss = torch.from_numpy(clss_np).float()

                    # 将检测结果送到deepsort里面
                    t4 = time_sync()
                    outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4
                    outputs[i] = _dedup_tracks(
                        np.asarray(outputs[i], dtype=np.float32) if len(outputs[i]) else np.empty((0, 6)),
                        iou_thres=0.5,
                    )
                else:
                    t4 = t5 = time_sync()
                    outputs[i] = np.empty((0, 6))
                    deepsort_list[i].increment_ages()

                # 保存结果
                if len(outputs[i]) > 0:
                    for output in outputs[i]:
                        bboxes = output[0:4]
                        tid = output[4]
                        cls = output[5]

                        if save_txt:
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, tid, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_crop:
                            c = int(cls)
                            crop_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(
                                bboxes, imc,
                                file=save_dir / 'crops' / crop_name / names[c] / f'{tid}' / f'{p.stem}.jpg',
                                BGR=True,
                            )

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')
                outputs[i] = np.empty((0, 6))

            # 黄线撞线 + ID 框 + 左上 OSD
            tracks = outputs[i] if outputs[i] is not None else np.empty((0, 6))
            im0 = render_frame(im0, tracks, names, COUNTER, colors)

            # Gradio：预览 Detection + 进度（每帧都更新摘要）
            if grstatus:
                last_det_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                progress = (
                    f"处理中：第 {processed} 帧"
                    + (f" / 最多 {max_frames}" if max_frames > 0 else "")
                    + f"（源帧 #{frame_idx}，步长 {frame_stride}）\n"
                    + COUNTER.summary_text()
                )
                if processed == 1 or processed % max(1, preview_every) == 0:
                    yield last_det_rgb, progress, None
            elif show_vid:
                try:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)
                except cv2.error:
                    show_vid = False

            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    last_saved_video = save_path
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    for vw in vid_writer:
        if isinstance(vw, cv2.VideoWriter):
            vw.release()

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if last_saved_video:
            LOGGER.info(f'Processed video: {last_saved_video}')
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)

    if grstatus:
        yield (
            last_det_rgb,
            COUNTER.summary_text(
                getattr(COUNTER, "_last_person_now", None),
                getattr(COUNTER, "_last_vehicle_now", None),
            ),
            _to_gradio_file(last_saved_video),
        )


def _build_default_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='../weights/yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='../video/smoke_5s_720p.mp4', help='source; official test video under ../video/')
    parser.add_argument('--output', type=str, default='../outputs/inference/output', help='output folder')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 2, 5, 7],
                        help='filter by class: person/car/bus/truck')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="tracker/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=str((ROOT / '../outputs/runs/track').resolve()), help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-ratio', type=float, default=0.5, help='yellow counting line y = h * ratio')
    opt = parser.parse_args([])
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def _as_video_path(video) -> str:
    """兼容 Gradio Video / File 返回值。"""
    if video is None:
        raise ValueError("请先上传视频文件")
    # gradio FileData / dict
    if isinstance(video, dict):
        path = video.get("path") or video.get("name") or video.get("video")
        if not path:
            raise ValueError("无法解析上传视频路径")
        return str(path)
    # 有的版本返回带 .path 属性的对象
    path = getattr(video, "path", None) or getattr(video, "name", None)
    if path:
        return str(path)
    return str(video)


def _parse_classes(text: str):
    text = (text or "").strip()
    if not text:
        return [0, 2, 5, 7]
    parts = [p for p in text.replace(",", " ").split() if p]
    return [int(p) for p in parts]


def _weight_choices() -> list[tuple[str, str]]:
    """返回 [(显示名, 绝对路径), ...]，便于下拉里认出 london2。"""
    repo = Path(__file__).resolve().parents[1]
    candidates = [
        ("phase_d_london3（自训练 人/车+花车，推荐）", repo / "outputs" / "runs" / "train" / "phase_d_london3" / "weights" / "best.pt"),
        ("phase_d_london2（自训练 人/车）", repo / "outputs" / "runs" / "train" / "phase_d_london2" / "weights" / "best.pt"),
        ("phase_d_london（自训练 人/车）", repo / "outputs" / "runs" / "train" / "phase_d_london" / "weights" / "best.pt"),
        ("phase_d（自训练 人/车）", repo / "outputs" / "runs" / "train" / "phase_d" / "weights" / "best.pt"),
        ("yolov5n（COCO 预训练，80类）", repo / "weights" / "yolov5n.pt"),
        ("yolov5s（COCO 预训练，80类）", repo / "weights" / "yolov5s.pt"),
    ]
    found = [(label, str(p.resolve())) for label, p in candidates if p.exists()]
    if not found:
        p = (repo / "weights" / "yolov5n.pt").resolve()
        found = [("yolov5n（COCO）", str(p))]
    return found


def _resolve_weight(weight) -> str:
    """下拉可能是显示名、路径，或 Gradio (label, value)。"""
    if isinstance(weight, (list, tuple)) and len(weight) >= 2:
        return str(weight[1])
    s = str(weight or "").strip()
    for label, path in _weight_choices():
        if s == label or s == path or Path(s).name == Path(path).name and "phase_d" in path:
            if s == label or s == path:
                return path
    # 显示名模糊匹配
    for label, path in _weight_choices():
        if s in label or label in s:
            return path
    return s


def build_demo(opt):
    import gradio as gr

    weight_list = _weight_choices()  # [(label, path), ...]
    default_label, default_path = weight_list[0]
    # Gradio Dropdown：用 label 作 choices，内部再映射到路径
    weight_labels = [lab for lab, _ in weight_list]
    label_to_path = {lab: path for lab, path in weight_list}

    def run_detect(
        video, weight, device, classes_text, conf_thres, frame_stride, max_frames, playback_fps,
        use_deepsort, vehicle_priority,
    ):
        if video is None:
            yield None, "请先上传视频文件，再点「开始检测计数」", None
            return
        weight_path = label_to_path.get(str(weight), _resolve_weight(weight))
        opt.source = _as_video_path(video)
        opt.yolo_model = [weight_path]
        opt.device = device or "cpu"
        opt.classes = _parse_classes(classes_text)
        opt.conf_thres = float(conf_thres)
        opt.iou_thres = 0.45  # YOLO NMS 更严，减少同车碎框
        opt.agnostic_nms = True
        opt.max_det = 100
        opt.show_vid = False
        opt.save_vid = True
        opt.exist_ok = True
        opt.preview_every = 1
        opt.frame_stride = int(frame_stride)
        opt.max_frames = int(max_frames)
        opt.playback_fps = float(playback_fps)
        opt.person_min_conf = 0.0

        # 车辆优先：略降阈值抓弱车，但不要压到 0.15（会一车多框）
        if vehicle_priority:
            opt.conf_thres = min(max(opt.conf_thres, 0.2), 0.28)
            opt.person_min_conf = 0.45
            opt.imgsz = [640, 640]
        elif use_deepsort:
            opt.imgsz = [640, 640]
        else:
            opt.imgsz = [416, 416]

        wlow = weight_path.replace("\\", "/").lower()
        if "phase_d" in wlow:
            opt.classes = [0, 1]

        out_video = None
        preview = None
        summary = ""
        try:
            with torch.no_grad():
                if use_deepsort:
                    gen = detect(opt, grstatus=True)
                else:
                    gen = detect_fast(opt)
                for preview, summary, video_path in gen:
                    if video_path:
                        out_video = video_path
                    yield preview, summary, out_video
            if out_video:
                yield preview, (summary or "") + f"\n\n完成。视频：{out_video}", out_video
        except Exception as e:
            import traceback
            yield None, f"出错：{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", None

    tips = (
        "**花车漏检怎么办**\n"
        "1. 勾选 **「车辆优先（花车/弱车）」**，置信度可再降到 0.15～0.2\n"
        "2. 权重可试 `yolov5s（COCO）` + 类别 `0,2,5,7`（大巴更稳；花车仍可能难）\n"
        "3. **根本办法**：多抽几帧把花车标成 `car`，再重新训练（现在车样本远少于人）\n"
    )

    with gr.Blocks(title="客流 / 车流计数") as demo:
        gr.Markdown("# 告诉车流量计数器")
        gr.Markdown(tips)
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.File(
                    label="上传原视频（mp4/avi/mov/mkv）",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    type="filepath",
                )
                weight = gr.Dropdown(
                    choices=weight_labels,
                    value=default_label,
                    label="YOLO 权重（默认 london3）",
                    allow_custom_value=False,
                )
                device = gr.Radio(choices=["cpu", "0"], value="cpu", label="设备")
                classes = gr.Textbox(
                    value="0,1",
                    label="检测类别 ID（0=人 1=车）",
                )
                conf = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="置信度（人少→略降；车多框→略升）")
                frame_stride = gr.Slider(1, 20, value=2, step=1, label="隔帧 stride（越大越快；计数不准就调小）")
                max_frames = gr.Slider(0, 500, value=150, step=10, label="最多处理帧数（0=不限制；越大结果越长）")
                playback_fps = gr.Slider(1, 15, value=4, step=1, label="输出播放帧率（越小越慢越长）")
                vehicle_priority = gr.Checkbox(
                    value=False,
                    label="车辆优先（仅花车仍漏时勾选；勾选后更容易叠框）",
                )
                use_deepsort = gr.Checkbox(value=False, label="使用 DeepSort")
                btn = gr.Button("开始检测计数", variant="primary")
            with gr.Column(scale=2):
                out_preview = gr.Image(label="处理后预览", type="numpy")
                out_summary = gr.Textbox(label="进度 / 计数摘要", lines=10)
                out_video = gr.File(label="处理后视频（跑完可下载）")

        btn.click(
            fn=run_detect,
            inputs=[
                video_in,
                weight,
                device,
                classes,
                conf,
                frame_stride,
                max_frames,
                playback_fps,
                use_deepsort,
                vehicle_priority,
            ],
            outputs=[out_preview, out_summary, out_video],
        )
    return demo


if __name__ == '__main__':
    import tempfile

    opt = _build_default_opt()
    demo = build_demo(opt)
    repo = Path(__file__).resolve().parents[1]
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        allowed_paths=[
            str(repo / "outputs"),
            str(repo / "weights"),
            str(repo / "video"),
            str(repo / "code"),
            tempfile.gettempdir(),
        ],
    )
