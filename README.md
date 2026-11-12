# PhotoPlus

本地化照片初筛与管理系统：批量摄入 RAW/JPG，自动完成物理质量评估、AI 语义打标、连拍去重优选，并输出可被 DxO PhotoLab 读取的 XMP 侧写文件。

- 架构：B/S（FastAPI + Vue 3），完全本地运行，无外部网络依赖
- 目标硬件：NVIDIA RTX 3050 Laptop / 4GB 显存（不 OOM）
- 后期衔接：`.xmp` sidecar（星级 + 关键词）

## 当前状态

**需求调研与可行性分析阶段，尚无代码实现。**

## 文档索引

| 文档 | 内容 |
|---|---|
| [code_artifact.md](code_artifact.md) | 原始开发需求与架构文档 |
| [ANALYSIS.md](ANALYSIS.md) | 技术可行性分析：设计缺陷、实现方案、环境实测、部署风险、验证清单、实施路线 |

## 结论速览

| 维度 | 判断 |
|---|---|
| 整体可行性 | 可行 |
| 4GB 显存约束 | 成立（CLIP ViT-B/32 fp16 约 1.4–2GB） |
| 离线运行 | 成立（权重需预置） |
| DxO 互操作 | 成立，仅支持星级/关键词/IPTC，不支持 `Rating=-1` |
| 原设计需修正 | 3 处：曝光阈值误杀星空、锐度未做尺寸归一化、去重特征与阈值不当 |

详见 [ANALYSIS.md](ANALYSIS.md)。

## 环境要求

| 项 | 要求 | 本机现状 |
|---|---|---|
| GPU | NVIDIA，≥4GB 显存，CUDA 可用 | RTX 3050 Laptop 4GB，但 `nvidia-smi` 报 NVML 错误 |
| Python | 3.11 / 3.12（PyTorch 暂无 3.14 wheel） | 仅 3.14.6 |
| ExifTool | 任意较新版本，需在 PATH 中 | 待确认 |
| Docker | 可选，用于容器化部署 | 未安装 |

## 计划中的技术栈

- 后端：Python 3.12 / FastAPI / SQLAlchemy + SQLite(WAL)
- 视觉：ExifTool（预览提取）、OpenCV（物理指标）、PyTorch + Transformers（CLIP ViT-B/32，fp16）
- 前端：Vue 3 + Vite + Tailwind CSS

## 计划中的目录结构

```
photoplus/
├─ backend/
│  ├─ app/            # FastAPI 路由与依赖
│  ├─ engines/        # exif / quality / clip / dedup / xmp
│  ├─ worker/         # GPU 推理消费者
│  └─ models.py       # ORM 定义
├─ frontend/          # Vue 3 + Vite
├─ config/            # scoring.yaml 等可调阈值
├─ cache/             # 缩略图与预览（git 忽略）
└─ docs/
```

## 开发顺序

1. 打通环境（驱动 / Python 3.12 / CUDA 自检）
2. 完成 Day-0 验证（预览提取、锐度排序、曝光误杀率、CLIP 标签、相似度分布、DxO 往返、显存峰值）
3. 按 `ANALYSIS.md` §11 的 M1–M8 推进

## 已知限制

- DxO PhotoLab 将星级存于数据库，写入 XMP 后需开启"与 XMP 侧写同步元数据"或手动"从图像读取"。
- 废片不使用 `xmp:Rating=-1`（DxO 不支持），改用 0 星 + 关键词 `AI/Reject` + `xmp:Label=Red`。
- 锐度分仅在"同机身同批次"内具有比较意义。
