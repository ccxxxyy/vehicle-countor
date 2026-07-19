"""从正式测试视频抽帧到 mydata/images，供 LabelImg 标注（阶段 C）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[4]  # vehicle-countor/
DEFAULT_VIDEO = ROOT / "video" / "9663b86299d95875dcdbe231c1d5caba_raw.mp4"
DEFAULT_OUT = Path(__file__).resolve().parent / "images"


def extract(
    video: Path,
    out_dir: Path,
    every: int = 30,
    max_width: int = 1280,
    prefix: str = "london",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            h, w = frame.shape[:2]
            if max_width > 0 and w > max_width:
                scale = max_width / w
                frame = cv2.resize(
                    frame,
                    (max_width, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            path = out_dir / f"{prefix}_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved += 1
        idx += 1
    cap.release()
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="抽帧到 mydata/images")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--every", type=int, default=30, help="每隔 N 帧保存一张")
    p.add_argument("--max-width", type=int, default=1280, help="缩小宽度便于标注；0=原尺寸")
    p.add_argument("--prefix", default="london", help="输出文件名前缀")
    args = p.parse_args()

    n = extract(args.video, args.out, args.every, args.max_width, args.prefix)
    print(f"saved {n} frames -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
