# 模型权重（唯一存放处）

本目录是预训练 / 微调权重的**唯一**目录，请勿再把 `.pt` 复制到 `code/` 下。

| 文件 | 说明 | Git |
|------|------|-----|
| `yolov5n.pt` | 轻量，CPU 冒烟推荐 | 可入库 |
| `yolov5s.pt` | 平衡 | 可入库 |
| `yolov5m.pt` | 中等精度 | 本地（>50MB，gitignore） |
| `yolov5l.pt` | 更大 | 本地（gitignore） |
| `yolov5x.pt` | 最大 | 本地（>100MB，gitignore） |

推理示例：

```bash
cd code
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid
```
