# 智能本地照片筛选与管理系统 — 技术可行性分析

> 分析对象：`code_artifact.md`（开发需求与架构文档 v1）
> 分析日期：2026-11-13
> 分析范围：需求可行性、设计缺陷、实现方案、环境验证、交付路线

---

## 1. 结论摘要

| 维度 | 判断 |
|---|---|
| 整体可行性 | **可行**，无技术死路 |
| 硬件约束（RTX 3050 Laptop 4GB） | **成立**，模型侧约需 1.4–2GB 显存，余量充足 |
| 离线约束（零外部 API） | **成立**，但模型权重需一次性预置进镜像/卷 |
| DxO PhotoLab 互操作 | **成立**，但有明确边界（见 §5） |
| 原文档设计 | **需修正 3 处**，否则会出现"能跑但不可用"的结果 |
| 本机当前状态 | **不能直接开工**，GPU 与 Python 环境均未就绪（见 §8） |

**核心判断**：四个引擎（预览提取、OpenCV 物理指标、CLIP 零样本分类、连拍去重）全部可实现。真正的风险不在"能不能做出来"，而在"做出来准不准"——即阈值标定和标签语义质量。因此本方案把"效果验证"前置到写代码之前。

---

## 2. 需求拆解与可行性矩阵

| 需求项 | 可行 | 依赖 | 备注 |
|---|---|---|---|
| B/S 架构本地部署 | 是 | FastAPI + Vue3 | 常规 |
| 4GB 显存不 OOM | 是 | fp16 + 单消费者队列 | 需严格串行 GPU |
| 离线运行 | 是 | 权重预置 | 首次构建需联网下载一次 |
| RAW 快速读取 | 是 | ExifTool | 禁用 Python RAW 渲染的思路正确 |
| 废片剔除 | 有条件 | 阈值 + 场景分类 | 见 §3.1，原阈值必然误杀星空 |
| AI 语义标签 | 有条件 | CLIP ViT-B/32 | 英文短句标签区分度一般，需 Prompt 工程 |
| 连拍去重优选 | 是 | EXIF 时间 + 特征相似度 | 见 §3.3，需改特征来源与阈值 |
| XMP 侧写输出 | 是 | 手写 XMP + 合并写 | 见 §5，命名与字段有硬约束 |
| 瀑布流相册 + 快捷键 | 是 | Vue3 | 常规，需补队列进度与批量操作 |
| Docker / Proxmox 部署 | 有条件 | NVIDIA Container Toolkit | 笔记本 dGPU 直通是最大不确定项，见 §9 |

---

## 3. 三个必须修正的设计问题

### 3.1 曝光/锐度阈值会集体误杀星空与夜景（最高优先级）

**问题**：原文档模块 B 用固定百分比阈值判定过曝/欠曝，超阈值直接 `REJECT`。银河与夜景照片天然有 70% 以上像素接近纯黑，拉普拉斯方差也极低（大面积平滑渐变 + 点状星点）。结果是星空片几乎全部被判废，而这恰恰是目标题材之一。

**修正**：

1. 物理引擎**只产出特征，不直接下终局结论**。输出 `sharpness_raw`、`clip_high_ratio`、`clip_low_ratio`、`mid_tone_ratio`、`histogram_entropy`。
2. 判定顺序改为：先跑 CLIP 粗分类 → 按 `scene_class` 取对应阈值组 → 再决定状态。
3. `REJECT` 收紧为强判据，例如：`高光溢出 > 15%` **且** `暗部死黑 > 60%` **且** `中心区锐度 < 同批次 P10`。其余情况一律落 `PENDING` 交人工。
4. 首次上线建议 `REJECT` 只做"提示"（`status=PENDING` + `flag=likely_reject`），积累 2–3 批人工反馈后再开启自动判废。

**原则**：宁可漏杀，不可错杀。初筛工具一旦误杀用户最在意的作品，用户就会永久弃用它。

### 3.2 锐度分在 RAW 内嵌预览上不可直接比较

**问题**：ExifTool 提取的预览分辨率随机身和机内设置变化（Canon CR3 的 `JpgFromRaw` 常见约 1620×1080；Sony ARW 的 `PreviewImage` 更小），而拉普拉斯方差对分辨率、压缩噪声、缩放算法极其敏感。跨机身、跨预览档位直接比较方差，会得出与真实对焦无关的排序。

**修正**：

1. 所有预览先统一缩放到固定长边（建议 1280），再做锐度计算，保证量纲一致。
2. 分批校准：以"同一机身 + 同一批次"为基线做相对排序，不跨批次写绝对分。
3. 原文档的"批次内 `log1p` 归一化"必须**降级为展示用的相对分值**，不能直接映射星级。星级映射应基于绝对指标 + 可配置阈值，否则同一张照片今天 4 星、下周导入新批次变 3 星。

```python
def sharpness_score(bgr, long_edge: int = 1280) -> tuple[float, float]:
    """返回 (全图方差, 中心 62% 区域方差)，已做尺寸归一化。"""
    h, w = bgr.shape[:2]
    scale = long_edge / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    full = cv2.Laplacian(gray, cv2.CV_32F).var()
    h, w = gray.shape
    mx, my = int(w * 0.19), int(h * 0.19)          # 保留中心 62%
    center = gray[my:h - my, mx:w - mx]
    return float(full), float(cv2.Laplacian(center, cv2.CV_32F).var())
```

### 3.3 去重不需要 MobileNetV2，且 0.88 阈值不适配

**问题一**：模块 C 已经产出 512 维归一化 CLIP 图像向量，模块 D 再加载 MobileNetV2 抽一遍特征属于重复计算：多一个模型、多一次前向、多占显存；且 MobileNetV2 特征对白平衡/曝光差异更敏感，连拍优选效果反而更差。

**修正**：复用 CLIP image embedding（存 BLOB，float32 × 512 = 2KB/张）。

**问题二**：CLIP 余弦相似度分布偏窄。同构图连拍通常落在 0.93–0.98，**不同构图也常有 0.85+**，因此 0.88 会把整个连拍组之外的内容也判为重复，或反过来组内判别失效。

**修正**：

1. 阈值改为实测标定（建议 0.93 起），配置化。
2. 组内两两比较前先用 pHash（汉明距离 ≤ 8）做廉价预筛。
3. 连拍分组键用 `camera_serial + burst_group`，时间间隔用 `SubSecTimeOriginal` 精确到毫秒，避免多机位同拍被错误合并。
4. RAW+JPG 同时存在时先归并为一个逻辑照片（`group_key = 相对路径去扩展名`），避免重复入库与重复评分。

---

## 4. 逐模块实现方案

### 4.1 模块 A：摄入与文件监视

**结论**：可行，但需三处工程化处理。

1. **预览提取标签顺序**：`-JpgFromRaw` 优先（质量最高），空则回退 `-PreviewImage`，再空回退 `-ThumbnailImage`。不同厂商标签名不一致，必须按序探测并缓存探测结果到配置（按机型记录）。
2. **ExifTool 必须批处理**：进程冷启动开销为数十毫秒级，逐文件调用在万张量级会成为主要瓶颈。使用 `-stay_open True -@ -` 常驻模式，实测可提升数十倍（官方文档明确说明 `-stay_open` 用于规避启动开销）。
3. **网络存储上禁用 inotify 依赖**：SMB/NFS 上的 inotify 事件会丢失或延迟，watchdog 只能作为加速手段。主路径改为"启动扫描 + 定时轮询（mtime/size 比对）+ 手动触发扫描"。
4. **增量识别**：以 `(相对路径, 文件大小, mtime, 可选 xxHash)` 作为指纹，避免重复处理。

**性能基线**：CR3 在 SD 卡上实测约 0.4s/张（ExifTool issue #384：158 张耗时 60s），本地 SSD 显著更快。NAS 场景建议 worker 侧先把文件复制到本地临时目录再处理，或接受该速率并做进度反馈。

### 4.2 模块 B：物理质量评估

见 §3.1、§3.2。补充实现细节：

- 灰度直方图分箱后计算 `clip_low = mean(gray < 4)`、`clip_high = mean(gray > 250)`、`mid_ratio = mean(50 < gray < 200)`。
- 增加 `histogram_entropy` 作为"信息量"辅助特征，可有效区分"死黑纯色"与"星空暗场"。
- 锐度取 `max(full, center)` 的思路保留，但两个原始值分别入库，便于后续调参回溯。
- 所有阈值集中到 `config/scoring.yaml`，按 `scene_class` 分组。

### 4.3 模块 C：CLIP 语义分类

**显存与吞吐（4GB 结论明确可行）**：

| 项 | 数值 | 来源 |
|---|---|---|
| ViT-B/32 峰值显存 | ≈1.4 GB（batch 32） | Jina CLIP-as-service 基准 |
| 权重规模（fp16） | ≈289 MB | HF model-sizer |
| 单图吞吐（T4 级 GPU） | 200–290 img/s（batch 32） | 同上，3050 Laptop 约为其 40–60% |

**实现要点**：

```python
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=torch.float16,      # 显存减半
    local_files_only=True,          # 离线约束
).to("cuda").eval()

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    img_emb = model.get_image_features(pixel_values).float()
    txt_emb = model.get_text_features(input_ids, attention_mask).float()
img_emb = F.normalize(img_emb, dim=-1)     # 归一化后余弦 = 点积
txt_emb = F.normalize(txt_emb, dim=-1)
```

- 文本侧**只算一次**并缓存（标签库固定），图像侧逐批计算，避免每张图重复编码 prompt。
- `batch_size` 从 8 起测，显存富余再升到 16/32；显存峰值记录到日志。
- **标签体系是关键风险**：CLIP 对 `street photography` / `family portrait` / `documentary` 这类近义短语区分度一般。建议：
  - 用句式模板 `"a photo of {label}"`，保持英文；
  - 每类配 2–3 个同义 prompt 取平均向量；
  - 设置最低置信阈值（如 0.24），低于阈值不写标签，宁缺毋滥；
  - 后续用人工标注 200 张做一次 prompt/阈值搜索，这一步的收益远大于换模型。

### 4.4 模块 D：连拍去重与优选

```python
# 分组：同机身 + 间隔 < 3s
photos.sort(key=lambda p: (p.camera_serial, p.capture_dt))
for group in groupby_burst(photos, gap_seconds=3.0):
    embs = np.stack([p.embedding for p in group])         # 已 L2 归一化
    sim = embs @ embs.T
    clusters = union_find(sim > SIM_THRESHOLD)            # 默认 0.93，可配置
    for cluster in clusters:
        best = max(cluster, key=lambda p: p.sharpness_score)
        for p in cluster:
            p.status = "KEEP" if p is best else "DUPLICATE"
        best.status = "KEEP"
```

- 单组内若全部低于阈值，视为"构图变化"，不做 DUPLICATE 标记。
- 用户手动改过状态的记录加 `manual_override=True`，重跑时跳过。
- pHash（`imagehash` 库，DCT 版 64bit）作为预筛：汉明距离 ≤ 8 才进入余弦比较。

### 4.5 模块 E：XMP 侧写生成

见 §5 专门章节。

### 4.6 模块 F：前端

- 瀑布流：CSS `column-count` 或 `grid-auto-rows: 8px` 的 JS 行跨度方案均可，**不要**用固定高度 + `object-fit: cover` 展示照片（会裁掉构图，用户无法判断），卡片应按原图比例自适应。
- 缩略图由后端预生成 512px WebP 落缓存目录，前端只请求静态路径，禁止每屏现算。
- 快捷键：`1-5` 星级、`X` 废片、`P` 保留、`←/→` 切换、`Space` 对比、`Z` 放大到 100%、`U` 撤销。
- 必须有的状态反馈：扫描/分析队列进度条、当前处理文件、失败文件列表与重试按钮。

---

## 5. XMP 与 DxO PhotoLab 互操作规范

依据 DxO 官方用户指南与支持文档，约束如下。

### 5.1 命名与位置

- RAW 文件 `IMG_1234.CR3` 的侧写文件名为 **`IMG_1234.xmp`**（同名、换扩展名），**不是** `IMG_1234.CR3.xmp`。
- 必须与 RAW 同目录。
- 仅 RAW 使用 `.xmp`；JPEG/TIFF/DNG 的元数据写在文件头内（系统只处理 RAW 时不受影响）。

### 5.2 可被 DxO 识别的字段

| 字段 | XMP 标签 | DxO 行为 |
|---|---|---|
| 星级 | `xmp:Rating`（0–5） | 读取，需启用"与 XMP 侧写同步元数据"或手动 `文件 > 元数据 > 从图像读取` |
| 关键词 | `dc:subject`（rdf:Bag） | 读取（启用关键词层级写入时兼容 Dublin Core） |
| 标题/说明 | `dc:title` / `dc:description` | 读取 |
| IPTC | `photoshop:*` / `Iptc4xmpCore:*` | 读取 |
| 颜色标签 | `xmp:Label` | **不识别**（DxO 用自有选片标记，存在 `.dop` 中） |
| 拒绝标记 | `xmp:Rating = -1` | **不支持**（DxO 只显示 0–5） |

### 5.3 状态映射（关键决策）

| 系统状态 | `xmp:Rating` | `dc:subject` 追加 | `xmp:Label` |
|---|---|---|---|
| KEEP（85–100） | 5 | `AI/{scene}` | `Green` |
| KEEP（70–84） | 4 | `AI/{scene}` | — |
| KEEP（55–69） | 3 | `AI/{scene}` | — |
| PENDING（40–54） | 2 | `AI/{scene}` | — |
| DUPLICATE | 1 | `AI/Duplicate` | `Yellow` |
| REJECT | 0 | `AI/Reject` | `Red` |

> 用 `0` 星而非 `-1` 表达"废片"，因为 DxO 不识别 -1。Bridge/Lightroom 侧可读 label 做进一步区分。

### 5.4 写入策略（不可省略）

1. **禁止整文件覆盖**：用户可能已有 Lightroom/DxO 生成的 `.xmp`，含关键词树、IPTC、修图参数。
2. 流程固定为：读取现有 → 解析 → 仅更新目标字段（`xmp:Rating`、`dc:subject` 中 `AI/` 前缀项、`xmp:Label`）→ 写临时文件 → `os.replace` 原子替换 → 保留 `.xmp.bak`。
3. 幂等：重复导出不产生重复关键词，不覆盖用户手工改过的星级（`manual_override` 时跳过）。
4. 首次写入前打印"将修改 N 个文件"确认，或提供 `--dry-run`。

### 5.5 参考模板

```xml
<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="PhotoPlus 0.1">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/">
   <xmp:Rating>4</xmp:Rating>
   <xmp:Label>Green</xmp:Label>
   <xmp:MetadataDate>2026-11-13T10:00:00+08:00</xmp:MetadataDate>
   <dc:subject>
    <rdf:Bag>
     <rdf:li>AI/astrophotography</rdf:li>
     <rdf:li>AI/keep</rdf:li>
    </rdf:Bag>
   </dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
```

### 5.6 用户侧操作要求

DxO 会优先使用数据库中的元数据。写入 XMP 后必须：

- 偏好设置 → 元数据 → 勾选"与 XMP 侧写文件同步元数据"，或
- 手动执行 `文件 > 元数据 > 从图像读取`。

此条必须写进 README，否则用户会认为系统没生效。

---

## 6. 架构调整建议

### 6.1 进程模型（重要）

**不要把 CUDA 推理放进 FastAPI 进程**。CUDA 上下文与 GIL、异步事件循环叠加会导致：

- 长任务阻塞事件循环（即使 `run_in_executor`，CUDA 上下文也会被多线程争用）；
- 一次 OOM 直接拖垮整个 Web 服务；
- 无法优雅地限制并发，显存不可预测。

调整为三进程：

```
api (FastAPI, 只读为主)  ──┐
                           ├── SQLite (WAL)
worker (单进程单消费者) ───┘   queue 表 + 轮询
```

- worker 独占 GPU，任务严格串行，天然避免显存竞争；
- 队列用数据库表实现（`status`: `PENDING/PROCESSING/DONE/FAILED` + `retry_count`），重启可恢复；
- 每个文件处理完立即 commit，避免大批量事务。

### 6.2 数据库

- 启用 `PRAGMA journal_mode=WAL`、`PRAGMA busy_timeout=5000`、`PRAGMA synchronous=NORMAL`。
- 跨进程不要复用 aiosqlite 连接；API 侧短连接 + 连接池，worker 侧单连接。
- embedding 存 `BLOB`（`np.float32.tobytes()`），查询时反序列化；不要存 JSON 数组（体积 4 倍、解析慢）。

### 6.3 缩略图

- 生成 512px（长边）WebP（质量 80），落 `cache/thumbs/{xx}/{hash}.webp`，两级目录散列避免单目录文件过多。
- 由 `StaticFiles` 直接托管，API 不参与字节传输。
- 原图预览（全屏查看）另存 2048px JPEG，按需生成 + 缓存。

---

## 7. 修订后的数据模型与 API

### 7.1 `photos` 表（相对原文档的增补）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INTEGER PK | |
| `rel_path` | TEXT UNIQUE | 相对根目录路径（避免根目录变更导致失效） |
| `abs_path` | TEXT | 运行期拼接，不参与唯一约束 |
| `group_key` | TEXT | 去扩展名路径，用于 RAW+JPG 归并 |
| `file_size` | INTEGER | 增量识别 |
| `file_mtime` | REAL | 增量识别 |
| `content_hash` | TEXT | 可选 xxHash，重命名识别 |
| `capture_time` | DATETIME | `DateTimeOriginal` |
| `subsec_time` | TEXT | `SubSecTimeOriginal`，连拍分组必需 |
| `camera_model` / `camera_serial` | TEXT | 分组与批次校准 |
| `lens` / `focal_length` / `aperture` / `iso` / `shutter_speed` | 混合 | 原设计保留 |
| `image_width` / `image_height` | INTEGER | 布局与比例计算 |
| `orientation` | INTEGER | 前端旋转 |
| `preview_path` | TEXT | 提取的预览缓存路径 |
| `sharpness_full` / `sharpness_center` | REAL | 原始方差 |
| `sharpness_norm` | REAL | 归一化分（仅用于同批次排序） |
| `clip_high` / `clip_low` / `mid_ratio` / `hist_entropy` | REAL | 曝光特征 |
| `normalized_score` | REAL | 0–100 |
| `scene_class` | TEXT | CLIP 主类 |
| `embedding` | BLOB | 512 × float32 |
| `status` | TEXT | PENDING / KEEP / DUPLICATE / REJECT / ERROR |
| `manual_override` | INTEGER | 人工干预标记，重跑跳过 |
| `xmp_synced_at` | DATETIME | 导出追踪 |
| `error_message` / `retry_count` | TEXT/INT | 失败可追溯 |

新增 `photo_tags` 关联表（`photo_id`, `tag`, `score`, `source`），替代 JSON 字符串，便于按标签筛选与聚合。

### 7.2 API 增补

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/scan` | 触发扫描（指定目录、是否强制重扫） |
| GET | `/api/queue/status` | 队列进度：总量/完成/失败/当前文件 |
| GET | `/api/photos` | 分页 + 状态/星级/标签/相机/日期筛选 + 排序 |
| GET | `/api/photos/{id}` | 详情，含全部特征与 EXIF |
| GET | `/api/photos/{id}/thumb` | 302 到静态缓存路径 |
| GET | `/api/groups` | 连拍组列表（组内最优置顶） |
| POST | `/api/photos/{id}/status` | 单张覆盖（写 `manual_override`） |
| POST | `/api/photos/batch` | 批量改状态（前端多选） |
| POST | `/api/export` | 导出 XMP（支持 `dry_run`、`only_changed`） |
| GET | `/api/export/report` | 导出结果：成功/跳过/冲突数 |

---

## 8. 本机环境实测（2026-11-13）

| 项目 | 实测结果 | 影响 |
|---|---|---|
| GPU | NVIDIA RTX 3050 Laptop，显存 4,293,918,720 B（≈4GB）；另有 AMD 集显 | 满足需求，但需显式选择 NVIDIA 设备 |
| CUDA 状态 | `nvidia-smi` 报 **"Failed to initialize NVML: Unknown Error"**，驱动 32.0.16.1088 | **阻塞项**：CUDA 当前不可用，需更新/重装驱动 |
| Python | 仅 3.14.6 | **阻塞项**：PyTorch 暂无 3.14 wheel，需安装 3.11 或 3.12 |
| Docker | 未安装 | 容器化阶段需先装 Docker Desktop + WSL2 |
| Git | 2.54.0 + Credential Manager 可用 | 可直接推送 |
| ExifTool | 未验证是否已安装 | 需确认 `exiftool -ver` |

**结论**：写代码之前必须先打通 §8 前两项，否则任何 GPU 相关代码都无法验证。

---

## 9. 部署（Proxmox + NAS）方案与风险

### 9.1 推荐分层

```
Proxmox VE
 ├─ LXC/VM: photoplus-web   → FastAPI + Vue 静态 + SQLite（无 GPU 依赖，7×24）
 ├─ LXC/VM: photoplus-worker→ 推理 worker（需 GPU）
 └─ NAS 挂载: /mnt/photos   → 只读挂载相册根目录
```

- Web 与 Worker 通过数据库队列表解耦，可部署在不同物理机。
- 若 GPU 直通失败，Worker 可先跑在笔记本本地（Windows + CUDA），Web 跑在 PVE，通过共享目录或 HTTP 拉任务，功能不受损。

### 9.2 风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 笔记本 dGPU 直通到 VM 失败（vBIOS/复位问题） | 中高 | 高 | 优先方案：Web 在 PVE、Worker 在物理机；GPU 直通作为优化项后置 |
| NAS 上 ExifTool 读取极慢 | 高 | 中 | worker 先本地缓存，或只对新增文件处理 |
| SMB inotify 丢事件 | 高 | 中 | 轮询扫描为主，watchdog 为辅 |
| 数据量大（>10 万张）后 SQLite 查询变慢 | 中 | 中 | 索引 + 分页游标；必要时迁移 PostgreSQL |

---

## 10. Day-0 验证清单（写代码前必做，约 1 天）

用 200–300 张真实素材（必须包含星空、夜景、连拍、逆光），逐项记录数值：

| # | 验证项 | 方法 | 通过标准 |
|---|---|---|---|
| 1 | 预览提取 | 对每张 RAW 依次试 `JpgFromRaw`/`PreviewImage`/`ThumbnailImage`，记录标签、分辨率、耗时（本地盘 vs NAS） | 覆盖率 100%，单张 < 1s（本地） |
| 2 | 锐度排序有效性 | 人工排出 20 张连拍的真实清晰度顺序，与方差排序做 Spearman 相关 | ρ ≥ 0.6，否则锐度不能作主判据 |
| 3 | 曝光误杀率 | 星空/夜景样本套用拟用阈值 | 误杀率 0% |
| 4 | CLIP 标签质量 | 50 张分类样本，统计 top-1/top-3 命中 | top-1 ≥ 70% 可用；否则调整标签库 |
| 5 | 连拍相似度分布 | 打印同组内与跨组的余弦分布 | 找到可分离阈值（如 0.93） |
| 6 | DxO 往返 | 手写一个 xmp，在 PhotoLab 中验证星级/关键词是否显示，验证 0 星与 -1 的表现 | 星级与关键词均可见 |
| 7 | 显存峰值 | fp16，batch 8/16，记录 `torch.cuda.max_memory_allocated()` | < 3.0 GB |

这 7 项产出的是**阈值与结论**，而不是代码。之后的实现基本是"按数据施工"。

---

## 11. 实施路线与工作量

| 阶段 | 内容 | 估时 |
|---|---|---|
| M0 | 环境打通（驱动/Python 3.12/CUDA 自检）+ Day-0 验证脚本 | 1.5–2 天 |
| M1 | 后端骨架：模型定义、ExifTool stay_open 封装、扫描与队列、缩略图 | 3–4 天 |
| M2 | 物理指标引擎 + 场景阈值配置 | 1–2 天 |
| M3 | CLIP 引擎（fp16、标签缓存、批量推理） | 2 天 |
| M4 | 连拍分组与去重优选 | 1–2 天 |
| M5 | XMP 合并导出 + dry-run 报告 | 1.5–2 天 |
| M6 | FastAPI 路由完整实现 | 2 天 |
| M7 | Vue3 前端（瀑布流、大图、快捷键、批量、进度） | 3–5 天 |
| M8 | Docker 化 + PVE/NAS 部署文档 | 2 天 |
| — | **合计（不含调参迭代）** | **约 2.5–3 周** |

不确定性主要在 M2/M3 的效果调优，属经验性工作，建议留出 3–5 天迭代窗口。

---

## 12. 风险登记

| ID | 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|---|
| R1 | 星空/夜景被误判为废片 | 高（原设计必然发生） | 高 | §3.1 场景化阈值 + REJECT 后置 |
| R2 | 锐度排序与真实对焦不相关 | 中 | 高 | Day-0 第 2 项先验证；不通过则改为辅助提示 |
| R3 | CLIP 标签噪声大 | 中高 | 中 | 阈值门控 + 标签库调优 + 人工可编辑 |
| R4 | 覆盖用户已有 XMP | 中 | 高 | 只合并、原子写、保留 .bak |
| R5 | 4GB 显存 OOM | 低 | 中 | 单消费者队列 + fp16 + batch 自适配 + OOM 自动降 batch |
| R6 | GPU 环境不可用（当前已发生） | 已发生 | 高 | M0 前置解决 |
| R7 | 网络存储性能瓶颈 | 高 | 中 | 本地缓存 + 轮询扫描 |
| R8 | 星级随批次漂移 | 高（原设计必然发生） | 中 | 绝对阈值映射，不做批次归一 |

---

## 13. 待确认事项

1. 主力机身与格式构成（Sony ARW / Canon CR3 / 其他比例），决定预览标签与分辨率基线。
2. DxO PhotoLab 具体版本（v5 起读 xmp 星级），以及是否愿意开启"同步元数据"。
3. NAS 协议（SMB/NFS）与相册根目录规模（当前张数、年增量）。
4. 是否需要多用户/多库（家庭共享），影响鉴权与数据库设计。
5. 星级映射阈值是否需在 UI 中可调（建议：是，且可导出配置）。
6. 是否需要"人工复核工作流"（大图对比、AB 二选一），这会是后续最大增值功能。

---

## 14. 对原文档的结论

`code_artifact.md` 的技术选型**方向正确**（ExifTool 提预览、OpenCV 算物理指标、CLIP 零样本、XMP 侧写），主要问题是：

1. 用固定阈值判废 → 必然误杀星空（§3.1）；
2. 未做预览尺寸归一化 → 锐度不可比（§3.2）；
3. 重复引入 MobileNetV2 且阈值凭经验 → 冗余且不准（§3.3）；
4. 未考虑 XMP 合并与 DxO 的字段限制（§5）；
5. 未考虑进程/显存/队列的工程约束（§6）；
6. 数据库与 API 字段不足（§7）。

按本文修正后，项目在 4GB 显存的笔记本上可稳定运行，离线可用，产出可直接进入 DxO PhotoLab 工作流。
