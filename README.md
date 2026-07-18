# 高速车流量计数器（YOLOv5 + DeepSort）

基于 **YOLOv5 目标检测** 与 **DeepSort 多目标追踪** 的离线视频 / 视频流车流量（及行人）计数项目。

系统流程：

```text
离线视频 / 视频流 → 车辆/行人检测(YOLOv5) → 目标追踪(DeepSort) → 撞线计数 → 前端可视化(Gradio)
```

本 README 覆盖：项目结构、环境准备、四项考核指标达成路径、实现计划与详细操作步骤。

---

## 目录

1. [项目定位与目标效果](#1-项目定位与目标效果)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [环境与依赖](#4-环境与依赖)
5. [四项指标与验收标准](#5-四项指标与验收标准)
6. [实现计划（分阶段）](#6-实现计划分阶段)
7. [详细操作步骤](#7-详细操作步骤)
8. [关键命令速查](#8-关键命令速查)
9. [常见问题](#9-常见问题)
10. [交付清单](#10-交付清单)

---

## 1. 项目定位与目标效果

### 1.1 定位

- **检测**：YOLOv5（可用 `pt/` 下预训练权重，或自训练权重）
- **追踪**：DeepSort（外观特征 + 卡尔曼滤波 + 匈牙利匹配）
- **计数**：中部黄线撞线；根据跨线前后中心点 y 判断向上 / 向下；事件文案区分行人 / 车辆
- **可视化**：对齐参考图——`ID:x` 框、黄线、左上中文面板、最新事件；Gradio 展示 Detection 画面

### 1.2 目标可视化效果（对齐最新参考图）

处理后的视频帧以「黄线撞线 + 左上统计面板 + ID 框」为验收外观标准：

| 元素 | 说明 |
|------|------|
| 检测框 | 彩色矩形框包裹行人 / 车辆，颜色可按 track ID 区分 |
| 标签 | 框上方显示 `ID:数字`（如 `ID:1`、`ID:29`） |
| 黄线 | 画面中部画一条水平黄色计数线（撞线 / tripwire） |
| 统计面板 | **左上角**半透明黑底面板，白字为主、最新事件用红字 |
| 最新事件 | 撞线瞬间提示，如：`最新：行人 1 号向下穿过黄线` |

**面板字段（与参考图一致）：**

```text
客流总数：N
穿过黄线人数：
向上：U
向下：D
最新：行人 X 号向上/向下穿过黄线
```

**计数语义（按参考图规划）：**

| 统计项 | 含义 |
|--------|------|
| 客流总数 | 当前画面中已出现过的独立 track 累计（或业务约定的总客流） |
| 穿过黄线人数 | 中心点（或底边）穿越黄线的目标数 |
| 向上 | 由画面下方 → 上方穿越黄线 |
| 向下 | 由画面上方 → 下方穿越黄线 |
| 人 / 车区分 | 检测类别含 person 与 car（及 truck/bus）；事件文案中写「行人 / 车辆」 |

> 说明：当前仓库 `main.py` / `webui.py` 是「左右半屏撞线 + 两侧大号数字」，与参考图不符。Phase B 需改为 **中部黄线 + 上下方向判定 + 左上面板 + `ID:x` 标签**（见下文）。

**视觉验收清单（结果视频每一帧应能看到）：**

- [ ] 彩色检测框 + 框上 `ID:数字`
- [ ] 画面中部一条黄色水平线
- [ ] 左上角半透明统计面板：客流总数 / 穿过黄线人数 / 向上 / 向下
- [ ] 有人过线时出现红色「最新：行人|车辆 X 号向上|向下穿过黄线」
- [ ] Gradio 右侧 Detection 与 CLI 保存视频观感一致

---

## 2. 系统架构

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ 视频输入     │────▶│ YOLOv5 检测   │────▶│ DeepSort 追踪 │────▶│ 黄线撞线计数 │
│ mp4 / 摄像头 │     │ bbox+cls+conf│     │ 稳定 track ID │     │ 向上/向下    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬──────┘
                                                                      │
                                                                      ▼
                                                              ┌─────────────┐
                                                              │ OSD 面板绘制 │
                                                              │ Gradio 展示  │
                                                              └─────────────┘
```

### 2.1 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| 检测推理入口 | `code/main.py` | 命令行跑检测 + 追踪 + 计数 + 保存视频 |
| Gradio 前端 | `code/webui.py` | 上传视频，流式输出原图与 Detection 图 |
| YOLOv5 | `code/yolov5/` | 训练、验证、检测、导出 |
| DeepSort | `code/tracker/` | 多目标关联与 ID 维护 |
| 预训练权重 | `pt/*.pt`、`code/yolov5m.pt` 等 | 直接推理或作微调起点 |
| 测试视频 | `video/*.mp4` | 离线演示与验收 |
| 标注与转换 | `code/yolov5/data/mydata/` | VOC XML → YOLO TXT、划分数据集 |

---

## 3. 目录结构

```text
vehicle-countor/
├── README.md                 # 本说明文档
├── pyproject.toml            # uv / 项目依赖（可选）
├── pt/                       # 官方预训练权重
│   ├── yolov5n.pt
│   ├── yolov5s.pt
│   ├── yolov5m.pt
│   ├── yolov5l.pt
│   └── yolov5x.pt
├── video/                    # 离线测试视频
│   └── 9663b86299d95875dcdbe231c1d5caba_raw.mp4
└── code/                     # 主代码
    ├── main.py               # 检测+追踪+撞线计数（CLI）
    ├── webui.py              # Gradio 可视化
    ├── requirements.txt      # pip 依赖清单
    ├── int8RT.py             # TensorRT INT8 相关（可选加速）
    ├── example.mp4           # 示例视频
    ├── yolov5m.pt / yolov5n.pt
    ├── tracker/              # DeepSort
    │   ├── deep_sort.py
    │   ├── configs/deep_sort.yaml
    │   ├── deep/             # ReID 特征与 checkpoint
    │   └── sort/             # 卡尔曼 / 匹配等
    └── yolov5/
        ├── train.py          # 训练
        ├── detect.py         # 纯检测
        ├── val.py            # 验证
        ├── models/           # 网络结构 yaml
        └── data/
            ├── mydata.yaml   # 自定义数据集配置（需改成本机路径）
            └── mydata/       # 标注数据工作区
                ├── images/   # 图片
                ├── xml/      # VOC 标注
                ├── labels/   # YOLO txt 标注
                ├── dataSet/  # train/val/test 划分列表
                ├── split_dataset.py
                └── voc2yolo_label.py
```

---

## 4. 环境与依赖

### 4.1 建议环境

| 项 | 建议 |
|----|------|
| OS | Windows 10/11（本仓库路径按 Windows 编写） |
| Python | 3.8 ~ 3.10（与 `code/requirements.txt`、YOLOv5 更匹配；`pyproject.toml` 中 `>=3.14` 偏新，若冲突以 `requirements.txt` 为准） |
| GPU | NVIDIA + CUDA（可选，CPU 可跑但较慢） |
| 磁盘 | 预留 ≥ 5GB（权重、视频、训练缓存） |

### 4.2 安装步骤

```bash
# 进入项目
cd d:\PythonProjects\vehicle-countor

# 创建虚拟环境（任选其一）
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖（推荐在 code 目录按官方清单安装）
cd code
pip install -r requirements.txt

# Gradio 前端（若 requirements 未包含）
pip install gradio

# 标注工具（指标二，单独安装）
pip install labelImg
# 或使用 labelImg 可执行包 / git clone 安装
```

### 4.3 权重说明

| 权重 | 用途 |
|------|------|
| `pt/yolov5*.pt` | COCO 预训练，可直接检测 person/car/truck/bus 等 |
| `code/yolov5m.pt` 等 | 运行入口默认可引用的本地权重 |
| DeepSort ReID | 首次运行会按 `tracker` 逻辑下载到 `tracker/deep/checkpoint/` |

快速验收可用 **COCO 预训练权重 + 类别过滤**（person=0, car=2, truck=7, bus=5），无需先完成自训练。

---

## 5. 四项指标与验收标准

### 指标一：程序可运行，测试视频可跑通，模型可训练且不报错

**验收点**

- [ ] `main.py` 能对 `video/` 下视频完成推理并输出结果视频
- [ ] `yolov5/train.py` 能启动训练（可用少量样本 smoke test）
- [ ] 全程无致命异常（ImportError / 路径错误 / CUDA 崩溃等）

**对应能力**：检测 + 追踪 + 撞线流水线可跑通。

---

### 指标二：数据标注程序可启动并能正确画框

**验收点**

- [ ] LabelImg（或同类 VOC 工具）可正常打开
- [ ] 能对 `images/` 中图片画矩形框
- [ ] 保存为 VOC XML，写入 `xml/`，类别名与配置一致（如 `person` / `car`）

**说明**：仓库未内置 LabelImg，需本机安装；标注格式与 `voc2yolo_label.py` 配套。

---

### 指标三：XML→TXT 转换正确，模型训练可正常跑

**验收点**

- [ ] `split_dataset.py` 划分 train/val/test
- [ ] `voc2yolo_label.py` 将 XML 转为 YOLO TXT（`labels/*.txt`）
- [ ] `mydata.yaml` 路径、`nc`、`names` 与标注类别一致
- [ ] `train.py` 能读取数据并开始迭代（至少跑通若干 epoch）

---

### 指标四：Gradio 前端可视化，可跑通检测与计数

**验收点**

- [ ] `webui.py` 启动后浏览器可打开界面
- [ ] 可上传测试视频
- [ ] 输出原视频帧与 Detection 帧（框、ID、计数信息）
- [ ] Detection 画面含黄线、左上统计面板、上下方向计数、最新事件提示

---

## 6. 实现计划（分阶段）

> 建议严格按 Phase 顺序推进：先跑通，再按参考图改造可视化，再标注训练与 Gradio。

### Phase A — 环境与基线跑通（对应指标一）【预计 0.5～1 天】

| 步骤 | 内容 | 产出 |
|------|------|------|
| A1 | 建虚拟环境，安装 `code/requirements.txt` + gradio | 可 import torch / cv2 / yolov5 |
| A2 | 确认权重路径：优先 `../pt/yolov5m.pt` 或 `code/yolov5m.pt` | 模型可加载 |
| A3 | 用 CLI 跑测试视频，开启 `--save-vid` | `runs/track/` 下结果 mp4 |
| A4 | 用 `yolov5/train.py` 对 `mydata` 做短时训练（1～3 epoch） | 训练不报错，生成 `runs/train/` |

**基线命令（示例）**

```bash
cd code
python main.py --yolo_model ../pt/yolov5m.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --classes 0 2 5 7 --device 0
```

> `--classes 0 2 5 7`：COCO 中 person / car / bus / truck。人和车都参与检测与撞线。

**本阶段完成标志**：指标一打勾。

---

### Phase B — 对齐参考图的撞线计数与 OSD（核心业务）【预计 1～2 天】

当前 `count_obj()` 为左右半屏计数，需改造成参考图风格：

| 步骤 | 内容 |
|------|------|
| B1 | 定义黄线位置：`line_y = int(h * 0.5)`（可配置比例） |
| B2 | 维护 `track_history[id] = (cy_prev, cls)`，用跨线前后 y 判断 **向上 / 向下** |
| B3 | 统计量：`total`（客流总数）、`crossed`、`up`、`down`；可选再拆 `person_*` / `vehicle_*` |
| B4 | 标签改为 `ID:{id}`；框颜色可用 `colors(id)` |
| B5 | 绘制黄线：`cv2.line(..., (0, 255, 255), thickness=2)` |
| B6 | 左上半透明面板 + 中文（PIL + 字体如 `msyh.ttc` / `simhei.ttf`） |
| B7 | 更新「最新」事件字符串（类别中文名 + ID + 方向） |
| B8 | 抽出公共模块（建议 `code/viz_count.py`），供 `main.py` / `webui.py` 共用 |

**状态机（防抖动）建议**

```text
on_side = sign(cy - line_y)   # +1 线下, -1 线上
若 prev_side 与 on_side 异号且 id 未在 cooldown：
    记一次穿越，更新方向计数与「最新」
    将该 id 加入已计数集合（或短时 cooldown）
```

**本阶段完成标志**：结果视频视觉接近参考图（黄线 + 左上面板 + 上下计数 + `ID:x` + 最新事件）。

---

### Phase C — 数据标注流程（对应指标二）【预计 0.5～1 天】

| 步骤 | 内容 |
|------|------|
| C1 | 安装并启动 LabelImg |
| C2 | 设置格式为 PascalVOC，目录指向 `mydata/images`，标注保存到 `mydata/xml` |
| C3 | 预定义类别：至少 `person`、`car`（可按需加 `truck`） |
| C4 | 对抽帧图片认真画框并保存 XML |

**抽帧建议（可选）**

```bash
# 从测试视频每隔 N 帧抽一张图到 mydata/images
python -c "import cv2,os; p=r'..\video\9663b86299d95875dcdbe231c1d5caba_raw.mp4'; out=r'yolov5\data\mydata\images'; os.makedirs(out,exist_ok=True); cap=cv2.VideoCapture(p); i=n=0
while True:
 r,f=cap.read()
 if not r: break
 if i%30==0: cv2.imwrite(os.path.join(out,f'{n}.jpg'),f); n+=1
 i+=1
print('saved',n)"
```

**本阶段完成标志**：指标二打勾。

---

### Phase D — 格式转换与训练（对应指标三）【预计 0.5～1 天】

| 步骤 | 内容 |
|------|------|
| D1 | 修改 `voc2yolo_label.py` 中 `classes = ["person", "car", ...]` 与标注一致 |
| D2 | 运行 `split_dataset.py` 生成 `dataSet/*.txt` |
| D3 | 运行 `voc2yolo_label.py` 生成 `labels/*.txt` 与 `train.txt` / `val.txt` |
| D4 | 修改 `data/mydata.yaml`：**改成本机绝对路径**，更新 `nc` 与 `names` |
| D5 | 修改 `models/yolov5s.yaml` 或 `custom_yolov5s.yaml` 的 `nc` 与类别数一致 |
| D6 | 启动训练，检查 loss 下降、权重写出 |

**本阶段完成标志**：指标三打勾；得到 `best.pt` / `last.pt`。

---

### Phase E — Gradio 前端完善（对应指标四）【预计 0.5～1 天】

| 步骤 | 内容 |
|------|------|
| E1 | 整理 `webui.py`：上传视频 → 调用改造后的 `detect` → 流式输出 origin / Detection |
| E2 | 界面文案：标题、说明、计数摘要（可用额外 Text组件显示最终人数/车数） |
| E3 | 默认权重、`--classes`、设备（cpu/cuda）可配置 |
| E4 | 关闭无头环境下的 `cv2.imshow`，避免服务器/无显示器报错 |
| E5 | 用 `video/` 测试视频完整跑通并录屏/截图验收 |

**本阶段完成标志**：四项指标全部满足。

---

### Phase F — 可选增强

- 速度区间统计（示例图中 `<90 / 90~110 / >110`）：需标定像素尺度或已知焦距与车道长度
- TensorRT / `int8RT.py` 加速
- 摄像头 / RTSP 实时流（`source=0` 或 rtsp URL）

---

## 7. 详细操作步骤

### 7.1 指标一：跑通检测 / 追踪 / 保存结果视频

```bash
cd d:\PythonProjects\vehicle-countor\code

# 推荐：保存结果视频，过滤人与常见车辆类
python main.py ^
  --yolo_model ../pt/yolov5m.pt ^
  --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 ^
  --save-vid ^
  --classes 0 2 5 7 ^
  --conf-thres 0.4 ^
  --device 0

# 仅 CPU
python main.py --yolo_model ../pt/yolov5n.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --classes 0 2 5 7 --device cpu
```

结果默认在：`code/runs/track/<实验名>/`。

**冒烟训练（验证 train.py 可跑）**

```bash
cd yolov5
# 先确保 mydata.yaml 路径已改成本机；可用极小 epoch
python train.py --img 640 --batch 4 --epochs 3 --data data/mydata.yaml --weights ../../pt/yolov5n.pt --device 0
```

---

### 7.2 指标二：数据标注（LabelImg）

1. 安装：`pip install labelImg` 后执行 `labelImg`
2. Open Dir → 选择 `code/yolov5/data/mydata/images`
3. Change Save Dir → 选择 `code/yolov5/data/mydata/xml`
4. 格式选 **PascalVOC**（生成 `.xml`）
5. 快捷键：`W` 画框，`Ctrl+S` 保存，`D` 下一张
6. 类别名使用英文小写，且后续与 `voc2yolo_label.py` 中 `classes` 列表完全一致

**目录约定**

```text
mydata/
  images/xxx.jpg
  xml/xxx.xml      ← LabelImg 输出
  labels/xxx.txt   ← 转换脚本输出
```

---

### 7.3 指标三：XML → TXT 与训练

#### （1）划分数据集

```bash
cd code\yolov5\data\mydata
python split_dataset.py --xml_path xml --txt_path dataSet
```

生成：`dataSet/train.txt`、`val.txt`、`test.txt`、`trainval.txt`（内容为不含扩展名的文件名列表）。

#### （2）修改类别并转换

编辑 `voc2yolo_label.py`：

```python
classes = ["person", "car"]  # 与 LabelImg 中类别顺序、名称一致
```

然后执行：

```bash
python voc2yolo_label.py
```

生成：

- `labels/*.txt`：每行 `class_id x_center y_center w h`（归一化）
- `train.txt` / `val.txt` / `test.txt`：图片绝对路径列表

#### （3）配置 `mydata.yaml`

将其中路径改成本机，例如：

```yaml
train: D:/PythonProjects/vehicle-countor/code/yolov5/data/mydata/train.txt
val: D:/PythonProjects/vehicle-countor/code/yolov5/data/mydata/val.txt

nc: 2
names: ["person", "car"]
```

> 仓库内现有 `mydata.yaml` 指向其他机器绝对路径，**必须修改**，否则训练读不到数据。

#### （4）开始训练

```bash
cd code\yolov5
python train.py --img 640 --batch 8 --epochs 50 --data data/mydata.yaml --cfg models/yolov5s.yaml --weights ../../pt/yolov5s.pt --name vehicle_person --device 0
```

权重输出示例：`runs/train/vehicle_person/weights/best.pt`。

用自训练权重做计数：

```bash
cd code
python main.py --yolo_model yolov5/runs/train/vehicle_person/weights/best.pt --source ../video/9663b86299d95875dcdbe231c1d5caba_raw.mp4 --save-vid --device 0
```

---

### 7.4 指标四：Gradio 前端

```bash
cd code
python webui.py
```

浏览器打开终端提示的本地地址（一般为 `http://127.0.0.1:7860`）：

1. 上传 `video/` 中测试视频  
2. 等待推理流式刷新  
3. 左侧为原图（origin），右侧为 Detection（框 + ID + 计数）

**前端改造建议（实现时）**

- `detect(..., grstatus=True)` 已支持 `yield` 双图；补齐按类别计数面板
- 无显示器时去掉 / 条件化 `cv2.imshow`
- 可增加：模型路径、置信度、类别过滤、设备选择控件

---

## 8. 关键命令速查

| 目的 | 命令 |
|------|------|
| 离线计数出视频 | `python main.py --yolo_model ../pt/yolov5m.pt --source ../video/xxx.mp4 --save-vid --classes 0 2 5 7` |
| 纯检测 | `python yolov5/detect.py --weights ../pt/yolov5m.pt --source ../video/xxx.mp4 --classes 0 2 5 7` |
| 划分数据 | `python yolov5/data/mydata/split_dataset.py` |
| XML→TXT | `python yolov5/data/mydata/voc2yolo_label.py` |
| 训练 | `python yolov5/train.py --data data/mydata.yaml --weights ../../pt/yolov5s.pt --epochs 50` |
| 前端 | `python webui.py` |
| 标注 | `labelImg` |

---

## 9. 常见问题

### 9.1 训练报错找不到图片

- `train.txt` / `mydata.yaml` 仍是旧机器路径 → 重新跑 `voc2yolo_label.py` 或手动替换路径前缀。

### 9.2 DeepSort / torchreid 下载失败

- 检查网络；可手动将 ReID 权重放到 `code/tracker/deep/checkpoint/`。
- 确认已安装 `torchreid`、`gdown`、`yacs`、`easydict`。

### 9.3 Gradio 打开后黑屏或卡死

- 首次加载模型较慢，等待终端日志。
- CPU 模式下长视频极慢，建议用 `yolov5n.pt` 或缩短测试片段。
- 无 GUI 环境禁用 `cv2.imshow`。

### 9.4 中文面板乱码

- OpenCV 默认字体不支持中文，需用 PIL + 中文字体绘制，或面板改用英文 `person/car/truck`。

### 9.5 计数不准

- 调整撞线高度（当前约 `h - 350`，需按分辨率适配）。
- 提高 `--conf-thres` 减少误检；保证 DeepSort ID 稳定（遮挡严重时可调 `MAX_AGE` / `N_INIT`）。
- 同一目标只计一次依赖 `data` 列表存 ID，重启程序会清零。

### 9.6 `classes` 过滤与自训练权重

- COCO 预训练：使用数字类别 ID（0 person, 2 car…）。
- 自训练权重：类别 ID 以你的 `names` 为准，一般不必再传 COCO 的 `--classes`，或按新映射过滤。

---

## 10. 交付清单

验收或提交前建议准备：

| 序号 | 交付物 | 对应指标 |
|------|--------|----------|
| 1 | 可运行环境说明 + 依赖安装成功截图 | 一 |
| 2 | 测试视频处理后的结果视频（含框与计数） | 一 / 四 |
| 3 | LabelImg 标注过程截图 + 示例 XML | 二 |
| 4 | 转换后的 `labels/*.txt` + 训练日志 / `best.pt` | 三 |
| 5 | Gradio 界面运行截图（origin + Detection） | 四 |
| 6 | 本 README 与关键运行命令 | 文档 |

---

## 附录 A：COCO 常用类别 ID（预训练权重）

| ID | 名称 |
|----|------|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

## 附录 B：当前代码与目标差距（实现时优先改）

| 项 | 现状 | 目标（对齐最新参考图） |
|----|------|------------------------|
| 撞线 | 底部左右两段绿/蓝线 | 画面中部 **一条黄线** |
| 计数维度 | 左右车道 `count` / `count2` | **客流总数 + 向上 / 向下**（事件区分行人/车辆） |
| 统计面板 | 大号数字简单 `putText` | **左上角**半透明中文面板 + 「最新」事件 |
| 标签样式 | `{id} {name} {conf}` | `ID:{id}` |
| 路径配置 | `mydata.yaml` 为他人机器路径 | 本机路径 |
| 类别配置 | 训练侧多为单类 `car` | 至少 `person` + `car` |
| Gradio | 已有双图 yield | 输出画面与 CLI 一致（黄线+面板） |

---

## 附录 C：推荐实施顺序（一页纸）

```text
1. 装环境 → 用预训练权重跑通 main.py 出结果视频              【指标一】
2. 改造撞线为黄线 + 上下方向计数 + 左上 OSD + ID:标签       【对齐参考图】
3. 安装 LabelImg，标注一批 person/car                       【指标二】
4. split + voc2yolo + 改 yaml → train.py 训练               【指标三】
5. 完善 webui.py，Gradio 输出与参考图一致的 Detection 画面  【指标四】
6. 整理截图、结果视频、权重，按交付清单验收
```

---

**维护说明**：实现代码改造时，优先保证 `main.py`（CLI 出视频）与 `webui.py`（Gradio）共用同一检测/计数函数，避免两套逻辑漂移。
