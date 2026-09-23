# Local VL-Jev — 用视觉语言模型本地复现 Jev 式视觉判别

> 一个开源项目，把 Jev 式固定答案打分扩展到视觉领域：输入图片+问题，不生成文本，只读取第一个 token 的 logits，在已知候选选项间做 softmax 归一化，返回概率分布。

**作者：贾承斌 (shengzing@163.com)**

[English](./README_EN.md) | 中文

## 这是什么

[local-jev](https://github.com/shengzing/local-jev) 证明了：用纯文本 LLM 做固定答案打分，比生成文本快 3-30 倍。本项目把同样的思路扩展到**视觉语言模型（VLM）**——

当应用需要根据图片做分类、判别、筛选时，不需要让 VLM "描述图片再写一个答案"，而是直接读取第一个 token 的 logits，在候选选项间做 softmax。

基于 `mlx-vlm` + Qwen2.5-VL-3B-Instruct，在 Apple Silicon 上原生运行。

### 典型场景

| 场景 | 输入 | 候选选项 | 当前做法 | VL-Jev 做法 |
|------|------|---------|---------|------------|
| 内容审核 | 商品截图 | 正常/违规/存疑 | VLM 生成描述→判断 | 直接打分返回分布 |
| 工单分类 | 错误截图 | UI/后端/网络 | VLM 写一段分析 | 直接打分返回分布 |
| 文档分类 | 文档照片 | 发票/合同/报表/信件 | VLM 生成文本→解析 | 直接打分返回分布 |
| 质量检测 | 产品照片 | 合格/不合格 | VLM 写描述→判断 | 直接打分返回分布 |
| 场景识别 | 风景照 | 室内/户外/夜景 | VLM 生成描述 | 直接打分返回分布 |

## 快速开始

```bash
# 1. 安装依赖
pip install mlx-vlm Pillow

# 2. 运行视觉判别（首次自动下载 Qwen2.5-VL-3B-4bit 模型）
python src/decide_vl.py --image assets/sample.jpg
```

输出：

```json
{
  "decision": "invoice",
  "probabilities": {
    "invoice": 0.9978,
    "contract": 0.0016,
    "report": 0.0002,
    "letter": 0.0004
  },
  "threshold_passed": true,
  "latency_ms": 2775.2
}
```

### 运行基准测试

```bash
python benchmark/run_benchmark_vl.py --dataset scene_recognition
python benchmark/run_benchmark_vl.py --dataset document_classification
python benchmark/run_benchmark_vl.py --dataset content_moderation
```

### 重新生成测试图片

`datasets/` 与 `assets/` 下的图片由脚本合成，可复现：

```bash
python scripts/generate_dataset_images.py
```

## 核心原理

### 文本 Jev vs 视觉 Jev

| 特性 | local-jev (文本) | local-vl-jev (视觉) |
|------|-----------------|---------------------|
| 输入 | 文本 prompt | 图片 + 文本 prompt |
| 模型 | Qwen2.5-0.5B-Instruct | Qwen2.5-VL-3B-Instruct |
| 推理路径 | tokenize → 前向传播 → 取 logits | 图片编码+tokenize → 前向传播 → 取 logits |
| 打分逻辑 | 读取 A/B/C logits → softmax | 完全相同 |
| 生成路径 | 自回归循环 | 完全相同（更慢，VLM 更大） |
| 速度优势 | 3x | 1.0-2.6x（实测；两条路径共享视觉编码成本，见实测结果） |

### 为什么视觉场景适合打分

1. **概率分布更有价值**：0.91 vs 0.46 的置信度差异，直接决定自动通过还是人工复核
2. **输出确定且结构化**：打分天然返回 `选项 → 概率` 的映射；生成路径需要解析自由文本，实测说明式回答的 label 解析失败率很高
3. **长输出场景显著提速**：输出长度不可控的生成（详细描述、多步推理）耗时 2-3 倍于打分

### 运行流程

```
用户输入（图片+问题+候选选项）
         │
    构建 multimodal prompt（含图片占位符 + 文字标签 A/B/C/D）
         │
    tokenize 验证标签是单 token
         │
    图片编码 + 文本 tokenize → 前向传播
         │
    读取第一个 token 的 logits（A/B/C/D 位置）
         │
    受限 softmax → 概率分布
         │
    映射回语义选项 → 返回决策
```

## 实测结果（Apple M4, Qwen2.5-VL-3B-Instruct-4bit, MLX）

### 模型加载

| 指标 | 数值 |
|------|------|
| 模型 | mlx-community/Qwen2.5-VL-3B-Instruct-4bit |
| 加载时间 | ~1s（模型已在本地缓存） |
| 词表大小 | 151,643 |
| 内存占用 | ~3.3GB (RSS) |

### 标签 Token 验证

| 标签 | Token ID | 状态 |
|------|----------|------|
| A (invoice) | [32] | ✓ 单 token |
| B (contract) | [33] | ✓ 单 token |
| C (report) | [34] | ✓ 单 token |
| D (letter) | [35] | ✓ 单 token |

与文本 Jev 完全一致——VLM 的标签 token 不受图片输入影响。

### 数据集打分结果（`python benchmark/run_benchmark_vl.py`）

仓库自带三个数据集（14 张 PIL 合成测试图，位于 `datasets/`）：

| 数据集 | 案例 | 打分准确率 | 生成准确率 | 打分延迟 | 生成延迟 |
|--------|------|-----------|-----------|---------|---------|
| scene_recognition | 5 | 5/5 (100%) | 5/5 (100%) | 1557 ms | 1464 ms |
| document_classification | 5 | 5/5 (100%) | 5/5 (100%) | 2473 ms | 2438 ms |
| content_moderation | 4 | 3/4 (75%) | 3/4 (75%) | 1480 ms | 1483 ms |

逐图打分明细（document_classification，打分概率均为 0.99+）：

| 图片 | 预期 | 决策 | 正确 | Top 概率 |
|------|------|------|------|---------|
| sample_invoice.jpg | invoice | invoice | ✓ | 0.999 |
| sample_contract.jpg | contract | contract | ✓ | 0.999 |
| sample_report.jpg | report | report | ✓ | 0.998 |
| sample_letter.jpg | letter | letter | ✓ | 0.996 |
| sample_receipt.jpg | invoice | invoice | ✓ | 0.997 |

### 打分 vs 生成：延迟构成分析

视觉判别的延迟大头是**视觉编码 + 一次 723-token 前向传播**（~2.7s，640×800 图片），生成路径同样要付这笔成本，差别只在自回归生成的 token 数：

| 生成形式 | 平均延迟 | 相对打分 (2726 ms) |
|---------|---------|-------------------|
| Jev 式打分（0 token 生成） | 2726 ms | 1.0x |
| 短回答生成（1-5 tokens） | 2642 ms | ≈1.0x |
| 说明式生成（~25 tokens） | 3029 ms | 1.11x |
| 长描述生成（150-250 tokens） | 7207 ms | 2.64x |

> **诚实结论**：在"模型只需输出一两个 token"的理想情况下，打分没有速度优势（两条路径共享同样的视觉编码和 prefill 成本）。打分的优势在于：① 输出长度不可控的长回答场景（2-3x）；② 天然返回结构化概率分布（生成路径需要脆弱的文本解析，实测说明式回答的 label 解析失败率很高）；③ 输出确定性。

### verify_mlx_vlm.py 验证输出（4 张 PIL 文字图）

```
  Image                     Expected     Decision      OK  Score(ms)    Gen(ms)
  invoice_test.png          invoice      invoice        ✓      742.3      793.9
  contract_test.png         contract     contract       ✓      733.8      785.4
  report_test.png           report       report         ✓      737.1      777.1
  letter_test.png           letter       letter         ✓      735.6      745.3

  指标                     视觉打分     视觉生成
  准确率                   4/4          4/4
  平均延迟 (ms)            737.2        775.4
```

### 重要修复记录

初版代码在 `apply_chat_template` 渲染时**漏传 `num_images=1`**，导致渲染结果不含 `<|image_pad|>` 占位符，图片从未真正进入模型——初版 README 中"16.7 ms 打分延迟、模型总是输出 report"的现象即是此 bug 的症状（纯文本前向 + 语言先验）。修复后图片真正参与推理：延迟升至真实的 ~1.5-2.7s（取决于图片分辨率），准确率从 25% 恢复到 100%。详见 `docs/architecture.md`。

## 项目结构

```
local-vl-jev/
├── README.md                    ← 中文文档
├── README_EN.md                 ← English docs
├── LICENSE                      ← MIT
├── pyproject.toml               ← 包元数据
├── requirements.txt             ← 依赖声明
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── decide_vl.py             ← 核心视觉决策客户端
│   ├── engine_vl.py             ← 双路径引擎（scoring vs generation）
│   └── utils_vl.py              ← 工具函数
├── benchmark/
│   └── run_benchmark_vl.py      ← 视觉打分 vs 生成对比
├── datasets/
│   ├── document_classification/ ← 文档分类数据集（含合成测试图）
│   ├── content_moderation/       ← 内容审核数据集（含合成测试图）
│   └── scene_recognition/        ← 场景识别数据集（含合成测试图）
├── verify_mlx_vlm.py            ← MLX-VLM 验证脚本
├── scripts/
│   └── generate_dataset_images.py ← 重新合成全部测试图片
├── assets/
│   └── sample.jpg                ← 快速开始示例图片
├── article/
│   └── .gitkeep                  ← 技术文章
└── docs/
    ├── architecture.md          ← 架构详解
    └── calibration.md           ← 校准与评估
```

> **注意**：模型参数文件（Qwen2.5-VL-3B-Instruct-4bit，约 3GB）不会包含在项目中。首次运行时由 mlx-vlm 自动从 HuggingFace 下载到本地缓存目录。

## 何时用视觉打分，何时用生成

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 图片分类、内容审核 | 打分 | 候选有限，只需选择 |
| 质量检测、缺陷判别 | 打分 | 枚举类别，概率分布有业务价值 |
| 图片描述、OCR | 生成 | 输出内容不可预知 |
| 视觉问答（开放） | 生成 | 需要自然语言回答 |
| 目标检测 | 生成 | 需要坐标输出 |

**核心判据**：如果候选答案可事先枚举，用打分；如果输出内容不可预知，用生成。

## 模型选择

| 模型 | 参数量 | 内存占用 | 精度 | 推荐场景 |
|------|--------|---------|------|---------|
| mlx-community/Qwen2.5-VL-3B-Instruct-4bit | 3B | ~1.5GB | 4bit | 默认推荐，平衡速度和效果 |
| mlx-community/Qwen2.5-VL-3B-Instruct-8bit | 3B | ~3GB | 8bit | 需要更高精度 |
| mlx-community/Qwen2.5-VL-7B-Instruct-4bit | 7B | ~4GB | 4bit | 16GB+ Mac，效果最好 |

## 致谢

- [local-jev](https://github.com/shengzing/local-jev) 文本版项目
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) MLX 视觉语言模型库
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 阿里通义千问视觉语言模型
- [mlx-community](https://huggingface.co/mlx-community) MLX 格式模型转换

## License

MIT — Copyright (c) 2026 贾承斌 (shengzing@163.com)
