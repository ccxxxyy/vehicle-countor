# 高速车流量计数器

---

## 目录

0. [使用指南](#0-使用指南)
1. [项目定位与目标效果](#1-项目定位与目标效果)
2. [计数方案（核心）](#2-计数方案核心)
3. [系统架构与目录](#3-系统架构与目录)
4. [环境与依赖](#4-环境与依赖)
5. [四项指标实现说明与验收](#5-四项指标实现说明与验收)
6. [实现计划（分阶段）](#6-实现计划分阶段)
7. [详细操作步骤](#7-详细操作步骤)
8. [关键命令速查](#8-关键命令速查)

---

## 0. 使用指南

### 0.1 安装

```powershell
cd vehicle-countor

python -m venv .venv
.\.venv\Scripts\activate

cd code
pip install -r requirements.txt
pip install gradio
```

标注工具：

```powershell
pip install labelImg "PyQt5==5.15.11" "PyQt5-Qt5==5.15.2" lxml
```

### 0.2 立刻跑通（COCO 预训练 + 冒烟视频）

```powershell
cd code
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 2 5 7 --device cpu --conf-thres 0.25
```

结果视频：`outputs/runs/track/。

```powershell
python webui.py
```

### 0.3 正式验收（有自训练权重 + 完整视频时）

1. 将需要处理的视频放到 `video/`
2. 将 `best.pt` 放到 `outputs/runs/train/phase_d_london3/weights/`，或按训练生成
3. 运行：

```powershell
cd code
python main.py --yolo_model ../outputs/runs/train/phase_d_london3/weights/best.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --classes 0 1 --device cpu --conf-thres 0.2
```
---

## 1. 项目定位与目标效果

### 1.1 定位

- **检测**：YOLOv5
- **追踪**：Gradio 默认 **SimpleIoUTracker**；可选 **DeepSort**
- **计数**：斜向黄线（伦敦街景：灯柱底 ↔ 赌场外 LED 屏下人行道）；方向按穿越时横向位移判 **向左 / 向右**
- **可视化**：`ID:x` + P/C、斜黄线、左上中文 OSD；Gradio 双栏预览与下载视频

### 1.2 画面效果

| 元素 | 说明 |
|------|------|
| 检测框 | 行人绿框 `ID:n P`，车辆橙框 `ID:n C` |
| 黄线 | 斜向；归一化默认 `(0.755, 0.64)→(0.04, 0.92)`，见 `code/viz_count.py` 的 `DEFAULT_LINE_NORM` |
| OSD | 左上半透明中文面板 |
| 最新事件 | **最新车** / **最新人** 分行红字 |

---

## 2. 计数方案

在 `code/viz_count.py`，由 `main.py` 与 `webui.py` **共用**同一套 `LineCrossingCounter` + `render_frame`，保证 CLI 与 Gradio 结果一致。

### 2.1 流水线

```text
每一帧：
  1) YOLOv5 → bbox + class + conf
  2) 追踪器分配稳定 track_id（IoU 或 DeepSort）
  3) LineCrossingCounter.update(id, bbox, …) → 若判定过线：累计人数/车数，记录向左/向右，更新「最新车/最新人」
  4) render_frame：画框、画黄线、画 OSD
```

### 2.2 黄线如何定义

- 用**归一化坐标**存端点，适配任意分辨率：`DEFAULT_LINE_NORM = (x1, y1, x2, y2)`
- 当前默认对齐验收街景手绘线：**右端** ≈ LED 屏下人行道，**左端** ≈ 近景灯柱底部
- 改线：只改 `viz_count.py` 中 `DEFAULT_LINE_NORM`，**重启** Gradio / 重新跑 CLI

### 2.3 过线如何判定

对每个 track：

1. 取检测框**底边中心** `(cx, y2)` 作为脚点（贴近路面；大巴若用框中心会悬在线上方导致永不穿越）
2. 至少观察 `MIN_HITS=3` 帧，抑制新 ID 闪框误计
3. 主规则：上一帧脚点 → 当前脚点 的线段与黄线**相交**，或相对黄线发生**侧翻**且靠近线段
4. 大目标兜底（大巴 / 贴镜头近景人）：底边已压在线上、尚未「穿过」时，用「底边贴线 / 框盖住黄线 + 足够横向位移」计一次
5. **每个 track_id 只计一次过线**（`_counted_cross`）

### 2.4 方向如何判定

- 用穿越当帧（或大目标累计）的横向位移：`dx > 0` → 向右，`dx < 0` → 向左

### 2.5 人 / 车如何区分

| 权重 | 类别 ID | 计数映射 |
|------|---------|----------|
| `phase_d_london3`（自训练） | `0=person`，`1=car` | 人 / 车；大巴等大目标按车逻辑 |

### 2.6 追踪两条路径

| 路径 | 入口 | 特点 |
|------|------|------|
| 快速（默认 Gradio） | `webui.detect_fast` + `SimpleIoUTracker` | CPU 更快；拥挤夜景可能换号 |
| DeepSort | `main.py` 默认 / Gradio 勾选 | 更稳更慢；需 ReID 相关依赖 |

换号会导致「同一辆车计两次」或「漏计」——这是精度上限来源之一，不是功能缺失。

---

## 3. 系统架构与目录

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────────┐
│ 视频输入     │────▶│ YOLOv5 检测   │────▶│ IoU / DeepSort   │────▶│ 斜线撞线计数 │
│ mp4         │     │ bbox+cls+conf│     │ track ID         │     │ 向左/向右    │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────┬──────┘
                                                                         ▼
                                                           viz_count OSD + Gradio
```

| 模块 | 路径 | 职责 |
|------|------|------|
| 撞线 / OSD | `code/viz_count.py` | 斜线、过线、中文面板 |
| CLI | `code/main.py` | 检测 + 追踪 + 计数 + 保存 |
| Gradio | `code/webui.py` | 上传视频；默认 `detect_fast` |
| YOLOv5 | `code/yolov5/` | 训练 / 验证 / 检测 |
| DeepSort | `code/tracker/` | 可选多目标关联 |
| 自训练权重 | `outputs/runs/train/phase_d_london3/weights/best.pt` | 产物验收 |
| 预训练 | `weights/*.pt` | 开箱冒烟 |
| 测试视频 | `video/` | 见 `video/README.md` |
| 标注 | `code/yolov5/data/mydata/` | 抽帧、XML、YOLO TXT |

```text
vehicle-countor/
├── README.md
├── weights/               
├── video/                  
├── outputs/                  
└── code/
    ├── main.py / webui.py / viz_count.py
    ├── requirements.txt
    ├── tracker/
    └── yolov5/data/mydata/   # images / xml / labels / 脚本
```

---

## 4. 环境与依赖

| 项 | 建议 |
|----|------|
| OS | Windows 10/11 |
| Python | 3.8～3.10（与 `code/requirements.txt`、YOLOv5 更匹配） |
| 设备 | **Gradio / 默认推理固定 CPU** |
| 磁盘 | ≥ 5GB |

---

## 5. 四项指标实现说明与验收

### 指标一：程序可运行，测试视频可跑通，模型可训练且不报错

**实现情况**

- [x] `main.py` 能对 `video/` 下视频完成推理并输出结果视频
- [x] `yolov5/train.py` 能启动训练（已完成 `phase_d_london3`）
- [x] 无致命异常（ImportError / 错误路径导致必崩等）

**实现说明**

1. 依赖：`code/requirements.txt` + PyTorch / OpenCV
2. 推理入口：`code/main.py` 加载 YOLO 权重 → DeepSort（CLI）→ 每帧调用 `viz_count.render_frame`
3. 冒烟视频入库：`video/smoke_5s_720p.mp4`；完整 raw 需自备
4. 训练入口：`code/yolov5/train.py` + `data/mydata.yaml` + `weights/yolov5n.pt`

**验收**

```powershell
cd code
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 2 5 7 --device cpu
```

冒烟训练：

```powershell
cd code\yolov5\data\mydata
python voc2yolo_label.py
# 编辑 ..\mydata.yaml 把 train/val 改成本机路径后：
cd ..\..
python train.py --img 640 --batch 4 --epochs 3 --data data/mydata.yaml --weights ../../weights/yolov5n.pt --device cpu --project ../../outputs/runs/train --name smoke_train
```
---

### 指标二：数据标注程序可启动并能正确画框

**实现情况**

- [x] LabelImg 可正常打开
- [x] 能对 `images/` 画矩形框
- [x] 保存为 VOC XML 到 `xml/`，类别 `person` / `car`

**实现说明**

1. 工具：LabelImg + 钉版本 PyQt5
2. 类别文件：`code/yolov5/data/mydata/predefined_classes.txt` → `person` / `car`
3. 一键脚本：`launch_labelimg.ps1`；抽帧：`extract_frames.py`
4. 仓库已含约 **84** 份 `images` + 对应 `xml`（伦敦夜景抽帧 + 样例）

**验收**

```powershell
powershell -ExecutionPolicy Bypass -File code\yolov5\data\mydata\launch_labelimg.ps1
```

在 LabelImg 中：Save Dir → `xml/`，格式 PascalVOC，打开任一张 `images/london_*.jpg` 能看到已有框或可新画框保存。

---

### 指标三：XML→TXT 转换正确，模型训练可正常跑

**实现情况**

- [x] `split_dataset.py` 划分 train/val/test
- [x] `voc2yolo_label.py` 将 XML 转为 YOLO TXT（`labels/*.txt`）
- [x] `mydata.yaml`：`nc: 2`，`names: ["person", "car"]`
- [x] 本机已训练产出 `phase_d_london3/weights/best.pt`

**实现情况**

1. `split_dataset.py`：按 XML 文件名划分到 `dataSet/*.txt`
2. `voc2yolo_label.py`：`classes=["person","car"]` → `labels/*.txt` 每行 `cls cx cy w h`（归一化），并写绝对路径的 `train.txt` / `val.txt` / `test.txt`
3. `data/mydata.yaml` 指向上述列表；`nc: 2`
4. 训练示例产出目录：`outputs/runs/train/phase_d_london3/`

**验收**

```powershell
cd code\yolov5\data\mydata
python split_dataset.py --xml_path xml --txt_path dataSet   # 若需重新划分
python voc2yolo_label.py                                    # 刷新本机路径
# 改 code/yolov5/data/mydata.yaml 中 train/val 为绝对路径
cd ..\..
python train.py --img 640 --batch 8 --epochs 50 --data data/mydata.yaml --cfg models/custom_yolov5s.yaml --weights ../../weights/yolov5n.pt --device cpu --project ../../outputs/runs/train --name phase_d_london3
```

训练结束后应出现：`outputs/runs/train/phase_d_london3/weights/best.pt`。

---

### 指标四：Gradio 前端可视化，可跑通检测与计数

**实现情况**

- [x] `webui.py` 可启动（默认 `http://127.0.0.1:7860`）
- [x] 可上传视频并下载处理后视频
- [x] Detection：斜向黄线、左上 OSD、过线人数/车数、最新车/最新人（向左/向右）
- [x] 可调权重、置信度、stride、最多处理帧数、播放帧率、是否 DeepSort
- [x] 滑块写入 `opt.max_frames` 等

**实现情况**

1. `code/webui.py`：`build_demo` 组装 Gradio；开始后走 `detect_fast`（默认）或 `detect`（DeepSort）
2. 与 CLI 共用 `viz_count.render_frame` / `LineCrossingCounter`，保证计数与画面一致
3. 权重下拉：优先 `phase_d_london3/.../best.pt`，不存在则 `yolov5n`
4. 固定 `device=cpu`；自训练权重强制 `classes=[0,1]`

**验收**

```powershell
cd code
python webui.py
```

1. 上传 `video/smoke_5s_720p.mp4`
2. 预训练：类别 `0,2,5,7`；有 london3：类别 `0,1`
3. 点开始 → 右侧应见斜黄线 + OSD；结束后可下载视频

---

## 6. 实现计划（分阶段）

### Phase A — 环境与基线（指标一）✅

开箱用 §0.2；正式验收用 london3 + raw（§0.3）。

### Phase B — 斜线撞线与 OSD ✅

见 [§2 计数方案](#2-计数方案)。

### Phase C — 数据标注（指标二）✅

约 84 张；`extract_frames.py` + `launch_labelimg.ps1`。

### Phase D — 转换与训练（指标三）✅

产出（本地）：`outputs/runs/train/phase_d_london3/weights/best.pt`。

### Phase E — Gradio（指标四）✅

```powershell
cd code
..\.venv\Scripts\python.exe webui.py
# 或：已 activate .venv 时直接 python webui.py
```
---

## 7. 详细操作步骤

### 7.1 指标一：跑通检测 / 追踪 / 保存结果视频

**开箱：**

```powershell
cd code
python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 2 5 7 --device cpu --conf-thres 0.25
```

**有自训练权重时：**

```powershell
python main.py --yolo_model ../outputs/runs/train/phase_d_london3/weights/best.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --classes 0 1 --device cpu --conf-thres 0.2
```

结果：`outputs/runs/track/<实验名>/`。

### 7.2 指标二：数据标注（LabelImg）

1. 安装
2. （可选）抽帧：`python code/yolov5/data/mydata/extract_frames.py`
3. 启动：`powershell -ExecutionPolicy Bypass -File code\yolov5\data\mydata\launch_labelimg.ps1`
4. **Change Save Dir** → `code/yolov5/data/mydata/xml`
5. 格式 **PascalVOC**；快捷键 `W` / `Ctrl+S` / `D`
6. 类别仅 `person` / `car`（与 `predefined_classes.txt`、`voc2yolo_label.py` 一致）

```text
mydata/
  images/     抽帧 / 待标注
  xml/        LabelImg VOC
  labels/     YOLO TXT（阶段三）
  predefined_classes.txt
  extract_frames.py / launch_labelimg.ps1 / split_dataset.py / voc2yolo_label.py
```

### 7.3 指标三：XML → TXT 与训练

```powershell
cd code\yolov5\data\mydata
python split_dataset.py --xml_path xml --txt_path dataSet
python voc2yolo_label.py
```

编辑 `code/yolov5/data/mydata.yaml`：

```yaml
train: <仓库绝对路径>/code/yolov5/data/mydata/train.txt
val: <仓库绝对路径>/code/yolov5/data/mydata/val.txt

nc: 2
names: ["person", "car"]
```

```powershell
cd code\yolov5
python train.py --img 640 --batch 8 --epochs 50 --data data/mydata.yaml --cfg models/custom_yolov5s.yaml --weights ../../weights/yolov5n.pt --device cpu --project ../../outputs/runs/train --name phase_d_london3
```

用新权重计数：

```powershell
cd code
python main.py --yolo_model ../outputs/runs/train/phase_d_london3/weights/best.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 1 --device cpu
```

### 7.4 指标四：Gradio 前端

```powershell
cd code
python webui.py
```

1. 有 london3 选 london3 + 类别 `0,1`；否则 yolov5n + `0,2,5,7`
2. 默认不勾选 DeepSort；可调 conf / stride / max_frames / playback_fps
3. 上传视频 → 开始 → 看 Detection 与 OSD → 下载结果
---

## 8. 关键命令速查

| 目的 | 命令 |
|------|------|
| 开箱冒烟计数 | `python main.py --yolo_model ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --save-vid --classes 0 2 5 7 --device cpu` |
| 自训练权重计数 | `python main.py --yolo_model ../outputs/runs/train/phase_d_london3/weights/best.pt --source ../video/xxx.mp4 --save-vid --classes 0 1 --device cpu` |
| 纯检测 | `python yolov5/detect.py --weights ../weights/yolov5n.pt --source ../video/smoke_5s_720p.mp4 --classes 0 2 5 7` |
| 刷新本机路径列表 | `python yolov5/data/mydata/voc2yolo_label.py` |
| 划分数据 | `python yolov5/data/mydata/split_dataset.py` |
| 训练 | 见 §7.3 |
| 前端 | `python webui.py` |
| 标注 | `launch_labelimg.ps1` 或 `labelImg` |

---

## 附录 A：推荐顺序

```text
1. §0 开箱：venv + yolov5n + smoke_5s + main.py / webui.py     【指标一】
2. 读 §2 计数方案；按需改 DEFAULT_LINE_NORM                   【业务】
3. LabelImg 查看/增补 mydata 标注                              【指标二】
4. voc2yolo + train → phase_d_london3/best.pt                 【指标三】
5. Gradio 用 london3 跑完整验收视频                            【指标四】
6. 按交付清单整理截图与视频
```