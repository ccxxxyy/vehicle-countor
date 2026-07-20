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
        self._last_person_now = 0
        self._last_vehicle_now = 0

    def line_y(self, h: int) -> int:
        return int(h * self.line_ratio)

    @staticmethod
    def _kind(class_name: str, cls_id: int | None = None) -> str:
        """返回 'person' | 'vehicle' | 'other'。自训练 2 类时 0=人 1=车。"""
        name = str(class_name).lower().strip()
        if name in ("person", "行人", "people", "pedestrian"):
            return "person"
        if is_vehicle(name) or name in ("车辆", "汽车"):
            return "vehicle"
        # 类别名异常时按 ID 兜底（自定义模型：0 person / 1 car）
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
    ) -> Optional[str]:
        """若发生跨线返回方向 'up'|'down'，否则 None。"""
        tid = int(track_id)
        x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
        cy = (y1 + y2) / 2.0
        self.observe_track(tid, class_name, cls_id=cls_id)

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
        kind = self._kind(class_name, cls_id=cls_id)
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

    def summary_text(self, person_now: int | None = None, vehicle_now: int | None = None) -> str:
        """供 CLI / Gradio：以当前画面人数/车数为准。"""
        lines = []
        if person_now is not None and vehicle_now is not None:
            lines.append(f"当前画面：行人 {person_now} / 车辆 {vehicle_now}")
        else:
            lines.append(f"当前画面：行人 {self._last_person_now} / 车辆 {self._last_vehicle_now}")
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
    # 中等字号：能看清，又不过度挡住画面
    h = im.shape[0]
    font_scale = max(0.55, min(0.85, h / 720.0 * 0.7))
    thick = max(2, int(round(h / 720.0 * 2)))
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thick)
    # 短标签：ID + 人/车（OpenCV 不画中文，用 P/C）
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


def draw_yellow_line(im, line_y: int, thickness: int = 2) -> None:
    h, w = im.shape[:2]
    cv2.line(im, (0, int(line_y)), (w - 1, int(line_y)), (0, 255, 255), thickness, cv2.LINE_AA)


def draw_osd_panel(im, person_now: int, vehicle_now: int) -> np.ndarray:
    """左上半透明中文面板：当前画面行人 / 车辆数量。"""
    lines = [
        f"行人：{person_now}",
        f"车辆：{vehicle_now}",
    ]

    font_size = max(22, int(im.shape[0] / 720 * 24))
    font = _load_font(font_size)
    pad_x, pad_y = 14, 12
    # 固定行距（字号+余量），中文字体用 getbbox 行高经常偏小会叠字
    line_h = font_size + 20

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
    im = cv2.addWeighted(overlay, 0.55, im, 0.45, 0)

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    y = 8 + pad_y
    for text in lines:
        draw.text((8 + pad_x, y), text, font=font, fill=(255, 255, 255))
        y += line_h
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_frame(
    im0: np.ndarray,
    tracks: List,
    names,
    counter: LineCrossingCounter,
    colors_fn,
    draw_line: bool = False,
) -> np.ndarray:
    """
    tracks: iterable of [x1,y1,x2,y2,track_id,cls_id]
    画面叠加 ID 框；左上显示【当前帧】行人/车辆数量（不是乱跳的累计 ID）。
    """
    im = im0.copy()
    h = im.shape[0]
    person_now = 0
    vehicle_now = 0

    for t in tracks:
        x1, y1, x2, y2, tid, cls_id = t[:6]
        c = int(float(cls_id))
        try:
            name = names[c] if not isinstance(names, dict) else names.get(c, names.get(str(c), str(c)))
        except Exception:
            name = str(c)
        # 累计仍记录（摘要用）；画面数字用当前帧
        counter.update(int(tid), (x1, y1, x2, y2), h, name, cls_id=c)
        kind = counter._kind(name, cls_id=c)
        if kind == "person":
            person_now += 1
            tag = "P"
            color = (80, 220, 80)  # 行人：绿
        elif kind == "vehicle":
            vehicle_now += 1
            tag = "C"
            color = (60, 160, 255)  # 车辆：橙蓝
        else:
            tag = "?"
            color = colors_fn(int(tid), True)
        draw_id_box(im, (x1, y1, x2, y2), int(tid), color, label_extra=tag)

    # 默认不画黄线（用户未要求撞线时不叠加）
    if draw_line:
        draw_yellow_line(im, counter.line_y(h), thickness=2)
    im = draw_osd_panel(im, person_now, vehicle_now)
    # 挂到 counter 上供 Gradio 摘要读取
    counter._last_person_now = person_now
    counter._last_vehicle_now = vehicle_now
    return im
