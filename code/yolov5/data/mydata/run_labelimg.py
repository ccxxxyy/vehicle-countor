"""Launch LabelImg with Python3.10+ crash fixes and project defaults."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
CLASSES = HERE / "predefined_classes.txt"
XML = HERE / "xml"


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                "LabelImg error",
                "".join(traceback.format_exception(exc_type, exc, tb))[-1500:],
            )
        except Exception:
            pass

    sys.excepthook = _hook


def _patch_canvas_ints() -> None:
    """LabelImg crashes on Py3.10+ when Qt draw* gets float coords."""
    from libs import canvas as canvas_mod
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QBrush, QColor

    orig = canvas_mod.Canvas.paintEvent

    def paintEvent(self, event):  # noqa: N802
        try:
            return orig(self, event)
        except TypeError:
            # Fallback path if vendor patch missing / partial
            from PyQt5.QtGui import QPainter
            from libs.shape import Shape

            if self.pixmap is None:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.HighQualityAntialiasing)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.scale(self.scale, self.scale)
            p.translate(self.offset_to_center())
            p.drawPixmap(0, 0, self.pixmap)
            Shape.scale = self.scale
            Shape.label_font_size = self.label_font_size
            for shape in self.shapes:
                if (shape.selected or not self._hide_background) and self.isVisible(shape):
                    shape.fill = shape.selected or shape == self.h_shape
                    shape.paint(p)
            if self.current:
                self.current.paint(p)
                self.line.paint(p)
            if self.selected_shape_copy:
                self.selected_shape_copy.paint(p)
            if self.current is not None and len(self.line) == 2:
                left_top = self.line[0]
                right_bottom = self.line[1]
                rect_width = right_bottom.x() - left_top.x()
                rect_height = right_bottom.y() - left_top.y()
                p.setPen(self.drawing_rect_color)
                p.setBrush(QBrush(Qt.BDiagPattern))
                p.drawRect(
                    int(left_top.x()),
                    int(left_top.y()),
                    int(rect_width),
                    int(rect_height),
                )
            if (
                self.drawing()
                and not self.prev_point.isNull()
                and not self.out_of_pixmap(self.prev_point)
            ):
                p.setPen(QColor(0, 0, 0))
                p.drawLine(
                    int(self.prev_point.x()),
                    0,
                    int(self.prev_point.x()),
                    int(self.pixmap.height()),
                )
                p.drawLine(
                    0,
                    int(self.prev_point.y()),
                    int(self.pixmap.width()),
                    int(self.prev_point.y()),
                )
            self.setAutoFillBackground(True)
            p.end()

    canvas_mod.Canvas.paintEvent = paintEvent


def _patch_create_shortcuts() -> None:
    import labelImg.labelImg as li

    def create_shape(self):
        if not self.beginner():
            self.toggle_advanced_mode(False)
        self.canvas.set_editing(False)
        self.actions.create.setEnabled(False)

    def set_create_mode(self):
        if not self.advanced():
            self.create_shape()
            return
        self.toggle_draw_mode(False)

    def set_edit_mode(self):
        if not self.advanced():
            return
        self.toggle_draw_mode(True)
        self.label_selection_changed()

    def scroll_request(self, delta, orientation):
        units = -delta / (8 * 15)
        bar = self.scroll_bars[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))

    li.MainWindow.create_shape = create_shape
    li.MainWindow.set_create_mode = set_create_mode
    li.MainWindow.set_edit_mode = set_edit_mode
    li.MainWindow.scroll_request = scroll_request


def _prefer_london_file(win) -> None:
    """优先打开待标注的新帧 london_030+（花车重训用）。"""
    files = sorted(IMAGES.glob("london_*.jpg"))
    if not files:
        return
    # 优先未标注的新帧
    unlabeled = []
    for p in files:
        stem = p.stem
        has = (XML / f"{stem}.xml").exists() or (XML / f"{stem}.txt").exists() or (HERE / "labels" / f"{stem}.txt").exists()
        if not has and stem >= "london_030":
            unlabeled.append(p)
    target = unlabeled[0] if unlabeled else files[0]
    try:
        win.load_file(str(target))
        win.set_fit_width(True)
        z = int(win.zoom_widget.value())
        if z < 40:
            win.set_zoom(70)
        n_new = len([p for p in files if p.stem >= "london_030"])
        win.status(
            f"Opened {Path(target).name} | 待标新帧约 {len(unlabeled)}/{n_new} | "
            f"花车务必选 car | W=画框 Ctrl+S=保存 D=下一张"
        )
    except Exception:
        traceback.print_exc()


def main() -> None:
    XML.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    if not CLASSES.exists():
        CLASSES.write_text("person\ncar\n", encoding="utf-8")

    _install_excepthook()
    _patch_canvas_ints()
    _patch_create_shortcuts()

    from labelImg.labelImg import LabelFileFormat, get_main_app
    from PyQt5.QtCore import QTimer
    from libs.constants import FORMAT_PASCALVOC

    argv = [
        "labelImg",
        str(IMAGES),
        str(CLASSES),
        str(XML),
    ]
    app, win = get_main_app(argv)

    # Force project defaults
    win.default_save_dir = str(XML)
    try:
        win.label_file_format = LabelFileFormat.PASCAL_VOC
        if hasattr(win, "set_format"):
            win.set_format(FORMAT_PASCALVOC)
    except Exception:
        pass

    # Force beginner mode (W = create_shape)
    try:
        if win.advanced():
            win.actions.advancedMode.setChecked(False)
            win.toggle_advanced_mode(False)
    except Exception:
        pass

    # Ensure class list has person + car
    for name in ("person", "car"):
        if name not in win.label_hist:
            win.label_hist.append(name)
    try:
        win.label_dialog = type(win.label_dialog)(parent=win, list_item=win.label_hist)
    except Exception:
        pass

    win.setWindowTitle(
        f"LabelImg | images={IMAGES.name} | save={XML} | classes=person,car"
    )
    QTimer.singleShot(300, lambda: _prefer_london_file(win))
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
