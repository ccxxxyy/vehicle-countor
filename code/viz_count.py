"""斜向撞线计数 + 左上中文 OSD，供 main.py / webui.py 共用。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_CN = {
    "person": "行人",
    "bicycle": "自行车",
    "car": "车辆",
    "motorcycle": "摩托",
    "bus": "车辆",
    "truck": "车辆",
    "train": "车辆",
}

VEHICLE_NAMES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}

_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]

# 伦敦视频默认计数线（归一化坐标 0~1）
# 左=近景灯柱底部；右=赌场外墙 LED 屏正下方人行道地面
DEFAULT_LINE_NORM = (0.755, 0.64, 0.04, 0.92)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def class_cn(name: str) -> str:
    return CLASS_CN.get(str(name).lower(), str(name))


def is_vehicle(name: str) -> bool:
    return str(name).lower() in VEHICLE_NAMES


def _side(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """点相对有向线段 AB 的叉积符号（>0 / <0 分居两侧）。"""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _seg_intersect(p1, p2, q1, q2) -> bool:
    """两线段是否相交（含端点）。"""
    def orient(a, b, c):
        v = _side(c[0], c[1], a[0], a[1], b[0], b[1])
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1

    def on_seg(a, b, c):
        return (
            min(a[0], b[0]) - 1e-6 <= c[0] <= max(a[0], b[0]) + 1e-6
            and min(a[1], b[1]) - 1e-6 <= c[1] <= max(a[1], b[1]) + 1e-6
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_seg(p1, p2, q1):
        return True
    if o2 == 0 and on_seg(p1, p2, q2):
        return True
    if o3 == 0 and on_seg(q1, q2, p1):
        return True
    if o4 == 0 and on_seg(q1, q2, p2):
        return True
    return False


class LineCrossingCounter:
    """
    斜向撞线计数。
    轨迹穿越该线段时计数一次；方向按穿越前后位移的 x 分量判定向左/向右。
    """

    # 抑制新 ID 出现就误计
    MIN_HITS = 4
    MIN_TRAVEL_FRAC = 0.012
    # 普通车：脚点到黄线超过该比例不算「已到线」
    VEHICLE_NEAR_FRAC = 0.028
    # 超大目标（大巴）启用贴线横向位移兜底
    HUGE_AREA_FRAC = 0.12

    def __init__(
        self,
        line_norm: Tuple[float, float, float, float] | None = None,
        line_ratio: float | None = None,
    ):
        # (x1,y1,x2,y2) 归一化；line_ratio 仅兼容旧水平线调用
        if line_norm is not None:
            self.line_norm = tuple(float(v) for v in line_norm)
        elif line_ratio is not None:
            r = float(line_ratio)
            self.line_norm = (0.0, r, 1.0, r)
        else:
            self.line_norm = DEFAULT_LINE_NORM
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.crossed = 0
        self.left = 0
        self.right = 0
        # 兼容旧字段名
        self.up = 0
        self.down = 0
        self.person_total = 0
        self.vehicle_total = 0
        self.person_crossed = 0
        self.vehicle_crossed = 0
        self.person_left = 0
        self.person_right = 0
        self.vehicle_left = 0
        self.vehicle_right = 0
        self.person_up = 0
        self.person_down = 0
        self.vehicle_up = 0
        self.vehicle_down = 0
        self.latest = ""
        self.latest_person = ""
        self.latest_vehicle = ""
        self._seen_ids: set = set()
        self._seen_person_ids: set = set()
        self._seen_vehicle_ids: set = set()
        self._prev_cxy: Dict[int, Tuple[float, float]] = {}
        self._hist: Dict[int, List[Tuple[float, float]]] = {}
        self._hits: Dict[int, int] = {}
        self._first_side: Dict[int, float] = {}  # 首次相对黄线的侧符号
        self._track_cls: Dict[int, str] = {}
        self._counted_cross: set = set()  # track_id 撞线只计一次
        self._last_person_now = 0
        self._last_vehicle_now = 0

    def line_xyxy(self, w: int, h: int) -> Tuple[int, int, int, int]:
        x1n, y1n, x2n, y2n = self.line_norm
        return (
            int(round(x1n * w)),
            int(round(y1n * h)),
            int(round(x2n * w)),
            int(round(y2n * h)),
        )

    # 兼容旧代码
    def line_y(self, h: int) -> int:
        _, y1, _, y2 = self.line_norm
        return int(round(((y1 + y2) * 0.5) * h))

    @staticmethod
    def _kind(class_name: str, cls_id: int | None = None) -> str:
        name = str(class_name).lower().strip()
        if name in ("person", "行人", "people", "pedestrian"):
            return "person"
        if is_vehicle(name) or name in ("车辆", "汽车"):
            return "vehicle"
        if cls_id is not None:
            if int(cls_id) == 0:
                return "person"
            if int(cls_id) == 1:
                return "vehicle"
        return "other"

    def observe_track(self, track_id: int, class_name: str, cls_id: int | None = None) -> None:
        tid = int(track_id)
        kind = self._kind(class_name, cls_id=cls_id)
        self._track_cls[tid] = str(class_name)
        if tid not in self._seen_ids:
            self._seen_ids.add(tid)
            self.total = len(self._seen_ids)
        if kind == "person" and tid not in self._seen_person_ids:
            self._seen_person_ids.add(tid)
            self.person_total = len(self._seen_person_ids)
        elif kind == "vehicle" and tid not in self._seen_vehicle_ids:
            self._seen_vehicle_ids.add(tid)
            self.vehicle_total = len(self._seen_vehicle_ids)

    def update(
        self,
        track_id: int,
        bbox_xyxy,
        frame_h: int,
        class_name: str,
        cls_id: int | None = None,
        frame_w: int | None = None,
    ) -> Optional[str]:
        """若穿越计数线返回方向 'left'|'right'，否则 None。"""
        tid = int(track_id)
        x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        kind = self._kind(class_name, cls_id=cls_id)
        if kind == "other":
            return None

        w = int(frame_w) if frame_w and frame_w > 0 else max(1, int(x2 * 2))
        h = int(frame_h)
        diag = float((w * w + h * h) ** 0.5) or 1.0
        area_ratio = (bw * bh) / max(float(w * h), 1.0)

        if kind == "person":
            ar = bw / bh
            if ar < 0.22 or bw < 0.012 * w or bh < 0.04 * h:
                return None
            if x1 < 0.02 * w and bw < 0.06 * w:
                return None

        # 行人/车辆都用底边中心（贴近路面）；大巴中心常悬在线上方导致永不穿越
        cx = (x1 + x2) * 0.5
        cy = float(y2)
        self.observe_track(tid, class_name, cls_id=cls_id)

        hits = self._hits.get(tid, 0) + 1
        self._hits[tid] = hits
        hist = self._hist.setdefault(tid, [])
        hist.append((cx, cy))
        if len(hist) > 10:
            del hist[:-10]

        ax, ay, bx, by = self.line_xyxy(w, h)
        q0 = (float(ax), float(ay))
        q1 = (float(bx), float(by))
        side_now = _side(cx, cy, q0[0], q0[1], q1[0], q1[1])
        if tid not in self._first_side and abs(side_now) > 1e-6:
            self._first_side[tid] = 1.0 if side_now > 0 else -1.0

        prev = self._prev_cxy.get(tid)
        self._prev_cxy[tid] = (cx, cy)
        if prev is None:
            return None
        if tid in self._counted_cross:
            return None
        if hits < self.MIN_HITS:
            return None

        p0 = (float(prev[0]), float(prev[1]))
        p1 = (cx, cy)
        lx, ly = float(bx - ax), float(by - ay)
        llen2 = lx * lx + ly * ly + 1e-6

        def _dist_point_to_seg(px: float, py: float) -> float:
            t = max(0.0, min(1.0, ((px - ax) * lx + (py - ay) * ly) / llen2))
            nx, ny = ax + t * lx, ay + t * ly
            return float(((px - nx) ** 2 + (py - ny) ** 2) ** 0.5)

        dist_now = _dist_point_to_seg(cx, cy)
        near_lim = self.VEHICLE_NEAR_FRAC * max(w, h)

        # 方向：用最近若干帧横向位移
        i0 = max(0, len(hist) - 6)
        dx = float(hist[-1][0] - hist[i0][0])
        dy = float(hist[-1][1] - hist[i0][1])
        step = (dx * dx + dy * dy) ** 0.5

        first = self._first_side.get(tid)
        side_flip = (
            first is not None
            and abs(side_now) > 1e-6
            and (side_now > 0) != (first > 0)
        )

        # —— 主规则：脚点轨迹与黄线相交（真正穿过）——
        crossed = _seg_intersect(p0, p1, q0, q1)
        if not crossed:
            s0 = _side(p0[0], p0[1], q0[0], q0[1], q1[0], q1[1])
            s1 = side_now
            if s0 != 0 and s1 != 0 and (s0 > 0) != (s1 > 0):
                # 侧翻且中点贴近线段才认（避免远处抖动假侧翻）
                mx, my = (p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5
                if _dist_point_to_seg(mx, my) <= 0.045 * max(w, h):
                    crossed = True

        # —— 车辆：禁止「离线很远」的 near/盖线兜底——
        if not crossed and kind == "vehicle":
            bottom_hits = _seg_intersect((x1, y2), (x2, y2), q0, q1)
            # 普通车：必须脚点已贴线 + 侧翻（或底边撞线）
            if dist_now <= near_lim and (side_flip or bottom_hits) and step >= self.MIN_TRAVEL_FRAC * diag * 0.5:
                crossed = True
            # 大巴：底边已压线 + 足够横向位移
            elif area_ratio >= self.HUGE_AREA_FRAC and bottom_hits:
                travel_x = abs(cx - hist[0][0])
                if travel_x >= 0.04 * w:
                    crossed = True
                    dx = float(cx - hist[0][0])
                    dy = float(cy - hist[0][1])

        # —— 行人：保留近景贴线兜底（近景人一进画就压线）——
        if not crossed and kind == "person":
            bottom_hits = _seg_intersect((x1, y2), (x2, y2), q0, q1)
            box_covers = (
                min(x2, max(ax, bx)) >= max(x1, min(ax, bx))
                and min(y2, max(ay, by)) >= max(y1, min(ay, by))
            )
            travel_x = abs(cx - hist[0][0])
            if area_ratio >= 0.05 and (bottom_hits or (box_covers and dist_now <= 0.06 * max(w, h))):
                if travel_x >= 0.03 * w:
                    crossed = True
                    dx = float(cx - hist[0][0])
                    dy = float(cy - hist[0][1])
            elif side_flip and dist_now <= 0.05 * max(w, h) and step >= self.MIN_TRAVEL_FRAC * diag * 0.4:
                crossed = True

        if not crossed:
            return None
        # 位移过小：框抖动不计数（大巴贴线除外）
        if step < self.MIN_TRAVEL_FRAC * diag * 0.4 and area_ratio < self.HUGE_AREA_FRAC:
            return None
        # 横向几乎不动时方向不可靠，等下一帧
        if abs(dx) < 0.004 * w and area_ratio < self.HUGE_AREA_FRAC:
            return None

        self._counted_cross.add(tid)
        direction = "right" if dx > 0 else "left"
        self.crossed += 1
        if direction == "right":
            self.right += 1
            self.down = self.right
            dir_cn = "向右"
            if kind == "person":
                self.person_right += 1
                self.person_down = self.person_right
                self.person_crossed += 1
            else:
                self.vehicle_right += 1
                self.vehicle_down = self.vehicle_right
                self.vehicle_crossed += 1
        else:
            self.left += 1
            self.up = self.left
            dir_cn = "向左"
            if kind == "person":
                self.person_left += 1
                self.person_up = self.person_left
                self.person_crossed += 1
            else:
                self.vehicle_left += 1
                self.vehicle_up = self.vehicle_left
                self.vehicle_crossed += 1

        if kind == "person":
            who = f"人{tid}号"
            msg = f"最新人：{who}{dir_cn}穿过黄线"
            self.latest_person = msg
        else:
            who = f"车{tid}号"
            msg = f"最新车：{who}{dir_cn}穿过黄线"
            self.latest_vehicle = msg
        self.latest = msg
        return direction

    def summary_text(self, person_now: int | None = None, vehicle_now: int | None = None) -> str:
        """供 CLI / Gradio。"""
        if person_now is None:
            person_now = self._last_person_now
        if vehicle_now is None:
            vehicle_now = self._last_vehicle_now
        lines = [
            f"当前画面：行人 {person_now} / 车辆 {vehicle_now}",
            f"穿过黄线：人数 {self.person_crossed} / 车数 {self.vehicle_crossed}",
            f"方向：向左 {self.left} / 向右 {self.right}",
            self.latest_vehicle or "最新车：暂无",
            self.latest_person or "最新人：暂无",
        ]
        return "\n".join(lines)


def draw_id_box(
    im,
    bbox_xyxy,
    track_id: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label_extra: str = "",
) -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy[:4])
    h = im.shape[0]
    font_scale = max(0.55, min(0.85, h / 720.0 * 0.7))
    thick = max(2, int(round(h / 720.0 * 2)))
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thick)
    tag = label_extra if label_extra else ""
    label = f"ID:{int(track_id)}" + (f" {tag}" if tag else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    pad = 4
    ty = max(0, y1 - th - pad * 2)
    cv2.rectangle(im, (x1, ty), (x1 + tw + pad * 2, y1), color, -1)
    cv2.putText(
        im,
        label,
        (x1 + pad, y1 - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )


def draw_count_line(im, x1: int, y1: int, x2: int, y2: int, thickness: int = 3) -> None:
    """绘制斜向计数线。"""
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    # 夜景：先画黑边再画黄线，避免被霓虹/路面吞掉
    cv2.line(im, p1, p2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.line(im, p1, p2, (0, 255, 255), thickness, cv2.LINE_AA)
    r = max(5, thickness + 2)
    cv2.circle(im, p1, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(im, p2, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(im, p1, r, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(im, p2, r, (0, 255, 255), -1, cv2.LINE_AA)


def _draw_text_stroke(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, stroke=(0, 0, 0)) -> None:
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=stroke)
    draw.text((x, y), text, font=font, fill=fill)


def draw_osd_panel(
    im,
    person_now: int,
    vehicle_now: int,
    person_crossed: int = 0,
    vehicle_crossed: int = 0,
    latest: str = "",
    latest_person: str = "",
    latest_vehicle: str = "",
) -> np.ndarray:
    """左上半透明中文面板：当前人数/车数 + 过线累计 + 最新车/最新人。"""
    lines = [
        f"行人：{person_now}",
        f"车辆：{vehicle_now}",
        f"穿过黄线人数：{person_crossed}",
        f"穿过黄线车数：{vehicle_crossed}",
        latest_vehicle or "最新车：暂无",
        latest_person or "最新人：暂无",
    ]
    # 兼容旧单条 latest
    if not latest_vehicle and not latest_person and latest:
        lines[-2] = latest if "车" in latest else "最新车：暂无"
        lines[-1] = latest if "人" in latest else "最新人：暂无"

    font_size = max(20, int(im.shape[0] / 720 * 22))
    font = _load_font(font_size)
    pad_x, pad_y = 14, 10
    line_h = font_size + 14

    def _text_width(text: str) -> int:
        try:
            bbox = font.getbbox(text)
            return max(1, bbox[2] - bbox[0])
        except Exception:
            try:
                return int(font.getsize(text)[0])
            except Exception:
                return len(text) * font_size

    max_w = max(_text_width(t) for t in lines)
    panel_w = max_w + pad_x * 2
    panel_h = pad_y * 2 + line_h * len(lines)

    overlay = im.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    im = cv2.addWeighted(overlay, 0.62, im, 0.38, 0)

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    y = 8 + pad_y
    for text in lines:
        fill = (255, 90, 90) if text.startswith("最新") else (255, 255, 255)
        _draw_text_stroke(draw, (8 + pad_x, y), text, font, fill)
        y += line_h
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_frame(
    im0: np.ndarray,
    tracks: List,
    names,
    counter: LineCrossingCounter,
    colors_fn,
    draw_line: bool = True,
) -> np.ndarray:
    """
    tracks: iterable of [x1,y1,x2,y2,track_id,cls_id]
    画面叠加 ID 框、斜向计数线、左上 OSD（含过线累计）。
    """
    im = im0.copy()
    h, w = im.shape[:2]
    person_now = 0
    vehicle_now = 0

    for t in tracks:
        x1, y1, x2, y2, tid, cls_id = t[:6]
        c = int(float(cls_id))
        try:
            name = names[c] if not isinstance(names, dict) else names.get(c, names.get(str(c), str(c)))
        except Exception:
            name = str(c)
        counter.update(int(tid), (x1, y1, x2, y2), h, name, cls_id=c, frame_w=w)
        kind = counter._kind(name, cls_id=c)
        if kind == "person":
            person_now += 1
            tag = "P"
            color = (80, 220, 80)
        elif kind == "vehicle":
            vehicle_now += 1
            tag = "C"
            color = (60, 160, 255)
        else:
            tag = "?"
            color = colors_fn(int(tid), True)
        draw_id_box(im, (x1, y1, x2, y2), int(tid), color, label_extra=tag)

    if draw_line:
        ax, ay, bx, by = counter.line_xyxy(w, h)
        draw_count_line(im, ax, ay, bx, by, thickness=max(3, int(round(h / 720.0 * 4))))

    im = draw_osd_panel(
        im,
        person_now,
        vehicle_now,
        person_crossed=counter.person_crossed,
        vehicle_crossed=counter.vehicle_crossed,
        latest=counter.latest,
        latest_person=getattr(counter, "latest_person", "") or "",
        latest_vehicle=getattr(counter, "latest_vehicle", "") or "",
    )
    counter._last_person_now = person_now
    counter._last_vehicle_now = vehicle_now
    return im
