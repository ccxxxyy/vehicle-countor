"""黄线撞线计数 + 左上中文 OSD（对齐参考图），供 main.py / webui.py 共用。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO / 常见类别 -> 中文展示名
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


class LineCrossingCounter:
    """中部黄线 + 方向感知撞线计数。"""

    def __init__(self, line_ratio: float = 0.5):
        self.line_ratio = float(line_ratio)
        self.reset()

    def reset(self) -> None:
        self.total = 0          # 客流总数（独立 track 出现数）
        self.crossed = 0        # 穿过黄线总事件数
        self.up = 0
        self.down = 0
        # 分别累计：出现过的独立 ID / 撞线事件
        self.person_total = 0
        self.vehicle_total = 0
        self.person_crossed = 0
        self.vehicle_crossed = 0
        self.person_up = 0
        self.person_down = 0
        self.vehicle_up = 0
        self.vehicle_down = 0
        self.latest = ""
        self._seen_ids: set = set()
        self._seen_person_ids: set = set()
        self._seen_vehicle_ids: set = set()
        self._prev_cy: Dict[int, float] = {}
        self._track_cls: Dict[int, str] = {}
        self._counted_cross: set = set()  # (track_id, direction) 防重复

    def line_y(self, h: int) -> int:
        return int(h * self.line_ratio)

    @staticmethod
    def _kind(class_name: str) -> str:
        """返回 'person' | 'vehicle' | 'other'。"""
        name = str(class_name).lower()
        if name == "person":
            return "person"
        if is_vehicle(name):
            return "vehicle"
        return "other"

    def observe_track(self, track_id: int, class_name: str) -> None:
        tid = int(track_id)
        kind = self._kind(class_name)
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
    ) -> Optional[str]:
        """若发生跨线返回方向 'up'|'down'，否则 None。"""
        tid = int(track_id)
        x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
        cy = (y1 + y2) / 2.0
        self.observe_track(tid, class_name)

        ly = self.line_y(frame_h)
        prev = self._prev_cy.get(tid)
        self._prev_cy[tid] = cy
        if prev is None:
            return None

        # 图像坐标：y 增大为向下
        crossed_down = prev < ly <= cy
        crossed_up = prev > ly >= cy
        if not (crossed_down or crossed_up):
            return None

        direction = "down" if crossed_down else "up"
        key = (tid, direction)
        if key in self._counted_cross:
            return None
        self._counted_cross.add(key)

        self.crossed += 1
        kind = self._kind(class_name)
        if direction == "up":
            self.up += 1
            dir_cn = "向上"
            if kind == "person":
                self.person_up += 1
                self.person_crossed += 1
            elif kind == "vehicle":
                self.vehicle_up += 1
                self.vehicle_crossed += 1
        else:
            self.down += 1
            dir_cn = "向下"
            if kind == "person":
                self.person_down += 1
                self.person_crossed += 1
            elif kind == "vehicle":
                self.vehicle_down += 1
                self.vehicle_crossed += 1

        cn = class_cn(class_name)
        self.latest = f"最新：{cn} {tid} 号{dir_cn}穿过黄线"
        return direction


def draw_id_box(im, bbox_xyxy, track_id: int, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy[:4])
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness)
    label = f"ID:{int(track_id)}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(0, y1 - th - 4)
    cv2.rectangle(im, (x1, ty), (x1 + tw + 4, y1), color, -1)
    cv2.putText(im, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_yellow_line(im, line_y: int, thickness: int = 2) -> None:
    h, w = im.shape[:2]
    cv2.line(im, (0, int(line_y)), (w - 1, int(line_y)), (0, 255, 255), thickness, cv2.LINE_AA)


def draw_osd_panel(im, counter: LineCrossingCounter) -> np.ndarray:
    """左上半透明中文面板（含行人/车辆分别累计）。"""
    lines = [
        f"客流总数：{counter.total}",
        f"行人：{counter.person_total}  车辆：{counter.vehicle_total}",
        "穿过黄线：",
        f"行人 向上：{counter.person_up}  向下：{counter.person_down}",
        f"车辆 向上：{counter.vehicle_up}  向下：{counter.vehicle_down}",
        f"合计 向上：{counter.up}  向下：{counter.down}",
    ]
    if counter.latest:
        lines.append(counter.latest)

    font = _load_font(22)
    font_small = _load_font(20)
    # 估面板尺寸
    pad_x, pad_y, line_h = 12, 10, 28
    max_w = 0
    for text in lines:
        fnt = font_small if text.startswith("最新") else font
        try:
            bbox = fnt.getbbox(text)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw, _ = fnt.getsize(text)
        max_w = max(max_w, tw)
    panel_w = max_w + pad_x * 2
    panel_h = pad_y * 2 + line_h * len(lines)

    overlay = im.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    im = cv2.addWeighted(overlay, 0.55, im, 0.45, 0)

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    y = 8 + pad_y
    for text in lines:
        if text.startswith("最新"):
            draw.text((8 + pad_x, y), text, font=font_small, fill=(255, 64, 64))
        else:
            draw.text((8 + pad_x, y), text, font=font, fill=(255, 255, 255))
        y += line_h
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_frame(
    im0: np.ndarray,
    tracks: List,
    names,
    counter: LineCrossingCounter,
    colors_fn,
) -> np.ndarray:
    """
    tracks: iterable of [x1,y1,x2,y2,track_id,cls_id]
    """
    im = im0.copy()
    h = im.shape[0]
    ly = counter.line_y(h)

    for t in tracks:
        x1, y1, x2, y2, tid, cls_id = t[:6]
        c = int(cls_id)
        name = names[c] if isinstance(names, (list, dict)) or hasattr(names, "__getitem__") else str(c)
        try:
            name = names[c]
        except Exception:
            name = str(c)
        counter.update(int(tid), (x1, y1, x2, y2), h, name)
        color = colors_fn(int(tid), True)
        draw_id_box(im, (x1, y1, x2, y2), int(tid), color)

    draw_yellow_line(im, ly, thickness=2)
    im = draw_osd_panel(im, counter)
    return im
