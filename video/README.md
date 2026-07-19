# 测试视频（项目验收核心素材）

本目录是**正式测试与验证**用视频的唯一存放位置。检测、追踪、撞线计数、Gradio 演示均应优先使用此处文件。

## 文件说明

| 文件 | 用途 | 规格 | 备注 |
|------|------|------|------|
| `9663b86299d95875dcdbe231c1d5caba_raw.mp4` | **官方原始测试视频**（验收首选） | 3840×2160，约 30s | 体积较大；本地保留，默认不推 GitHub |
| `smoke_15s.mp4` | 原始视频前 15s 截取 | 4K，约 15s | 调试用；体积大，默认不推远程 |
| `smoke_5s_720p.mp4` | 快速冒烟 / 日常调试 | 1280×720，约 5s | 已入库，方便克隆后立刻跑通 |

## 推荐用法

```bash
cd code

# 日常调试（快）
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 2 5 7 --device cpu

# 正式验收（完整原始视频）
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --classes 0 2 5 7 --device 0
```

结果视频默认写到：`outputs/runs/track/`。

## 注意

- 不要把测试视频再复制到 `code/` 下；`code/VIDEO.md` 仅作路径说明。
- 克隆仓库后若缺少 raw / 15s 文件，请自行放回本目录（本地大文件未进 Git）。
