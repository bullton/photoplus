# 智能本地照片筛选与管理 Web 系统 - 开发需求与架构文档

## 1. 项目概述与核心目标
本项目旨在开发一套完全本地化运行的 B/S 架构照片初筛系统。核心目标是解决海量摄影照片（尤其是风光、星空天文、街拍及家庭记录）的快速筛选、去重与质量打分问题。
*   **硬件约束**：系统需在具有 **4GB VRAM 的 NVIDIA RTX 3050 Laptop GPU** 上稳定运行，严禁显存溢出（OOM）。
*   **网络约束**：离线优先，零外部网络 API 依赖。
*   **工作流集成**：分析结果最终需生成标准 `.xmp` Sidecar 文件，确保与 DxO PhotoLab 等后期软件的无缝衔接。
*   **部署环境**：支持 Docker 容器化部署（未来将运行于 Proxmox VE 虚拟化环境，挂载 NAS 存储）。

## 2. 技术栈选型
*   **后端开发语言**：Python 3.10+
*   **Web 框架**：FastAPI (利用 AsyncIO 处理高并发 I/O 请求)
*   **数据库**：SQLite3 (使用 SQLAlchemy ORM 或直接配合 aiosqlite)
*   **计算机视觉与 AI**：
    *   `OpenCV` (cv2)：图像基础操作与拉普拉斯方差计算。
    *   `PyTorch` + `Transformers`：运行 OpenAI CLIP (ViT-B/32) 进行零样本分类。
    *   `torchvision`：运行 MobileNetV2 进行特征提取与去重。
*   **元数据处理**：`PyExifTool` (调用 ExifTool 提取 EXIF 及 RAW 内嵌预览图)
*   **前端框架**：Vue 3 (Composition API) + Vite + Tailwind CSS

## 3. 核心后端模块详细设计

### 模块 A：极速摄入与文件监视 (Ingestion & Watcher)
*   **功能**：使用 `watchdog` 库监听指定挂载目录（Inbox），当有新照片（支持 JPG, ARW, CR3 等 RAW 格式）写入时触发处理。
*   **RAW 处理策略（关键）**：严禁使用 Python 库直接渲染 RAW 文件！必须通过命令 `exiftool -b -PreviewImage [文件路径]` 提取 RAW 内嵌的 JPG 预览图加载进内存（使用 `io.BytesIO` 和 `cv2.imdecode`），以保障摄入速度。

### 模块 B：物理质量评估引擎 (Focus & Exposure Check)
*   **功能**：剔除绝对废片。
*   **曝光检测**：转灰度图，计算直方图。如果过曝像素或欠曝像素占比超过设定阈值，直接标记为 `REJECT`。
*   **对焦/锐度评分**：实现类似 `FocusCore` 的算法：
    1.  计算全图的 4 邻域拉普拉斯方差 (Laplacian Variance)。
    2.  计算中心 62% 区域的拉普拉斯方差，取两者最大值作为基础锐度分。
    3.  结合批次数据，使用对数函数 (`Math.log1p`) 进行分数归一化 (0-100分)。

### 模块 C：AI 语义分类引擎 (Semantic Tagger)
*   **模型约束**：加载 `openai/clip-vit-base-patch32`。必须使用 `torch.no_grad()` 和 `model.half()` 开启 FP16 精度，确保 4GB VRAM 足够运行。
*   **功能**：预设标签库（如：`["landscape", "street photography", "astrophotography", "macro", "family portrait", "low light"]`）。将提取的图像进行推理，计算余弦相似度，将得分最高的 Top-3 词汇作为照片的 Tags。

### 模块 D：连拍去重与优选引擎 (Deduplication Engine)
*   **聚类逻辑**：读取 EXIF 中的精确拍摄时间，将间隔小于 3 秒的连续照片划归为同一“连拍组”。
*   **特征提取**：加载预训练的 `MobileNetV2`，抽取展平的特征向量。
*   **优选逻辑**：若连拍组内照片特征余弦相似度 > 0.88，则认为构图相同。系统自动比惹该组内照片在“模块 B”中的锐度得分，最高分标记为 `KEEP`，其余标记为 `DUPLICATE`。

### 模块 E：XMP 导出引擎 (Sidecar Generator)
*   生成与 RAW 文件同名的 `.xmp` 文件。
*   将系统的归一化评分 (0-100) 映射为星级 (1-5)，写入 `<xmp:Rating>` 节点。
*   将 AI 提取的标签写入 `<dc:subject>` 节点。
*   将 `REJECT` 或 `DUPLICATE` 状态写入颜色标签或特定标记中，方便后续筛选。

## 4. 数据库 Schema 设计 (SQLite)

请使用 ORM 定义以下核心表：
*   **Table: Photos**
    *   `id` (PK)
    *   `filename` (String)
    *   `filepath` (String, Unique)
    *   `capture_time` (DateTime)
    *   `camera_model` (String)
    *   `focal_length`, `aperture`, `iso`, `shutter_speed` (String/Float)
    *   `sharpness_score` (Float, 原始方差)
    *   `normalized_score` (Float, 0-100)
    *   `status` (Enum: PENDING, KEEP, DUPLICATE, REJECT)
    *   `tags` (JSON String)

## 5. API 接口定义 (FastAPI)

1.  `GET /api/photos`：支持分页 (skip, limit)、状态筛选 (status) 的照片流接口。
2.  `GET /api/photos/{id}/thumbnail`：返回从源文件提取的高速 Web 预览图 (需在后端缓存)。
3.  `POST /api/photos/{id}/status`：手动覆盖系统判定的状态（接受 KEEP, REJECT 等参数）。
4.  `POST /api/export`：触发系统将数据库中的状态与标签批量生成 XMP 侧写文件。

## 6. 前端 UI/UX 需求 (给前端大模型的提示)
*   **主界面**：类似相册瀑布流（Masonry Grid），使用懒加载 (Lazy Load) 展示图片。
*   **图片卡片**：缩略图左上角显示 1-5 星评分，右上角用红/绿圆点直观显示系统状态（绿=保留，灰=重复，红=废片）。下方展示 AI Tags。
*   **快捷键支持**：点击大图预览时，支持键盘 `1-5` 打星，`X` 标废片，`P` 标保留，左右方向键切换。

## 7. 给 AI 代码生成的执行建议
请 LLM 按照以下顺序逐步生成代码，不要一次性输出所有文件：
1.  **步骤一**：先给出 `requirements.txt` 和 `models.py` (数据库定义)。
2.  **步骤二**：编写图像提取与 OpenCV 物理检测模块。
3.  **步骤三**：编写基于 PyTorch 的 CLIP 和 MobileNetV2 引擎（注意 4GB 显存优化的代码细节）。
4.  **步骤四**：编写 FastAPI 路由与 XMP 导出逻辑。
5.  **步骤五**：编写基于 Vue 3 的前端基础代码。