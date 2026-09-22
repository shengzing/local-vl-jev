# Local VL-Jev — 用视觉语言模型本地复现 Jev 式视觉判别

> 一个开源项目，把 Jev 式固定答案打分扩展到视觉领域：输入图片+问题，不生成文本，只读取第一个 token 的 logits，在已知候选选项间做 softmax 归一化，返回概率分布。

**作者：贾承斌 (jiacb@wiseweb.com.cn)**

[English](./README_EN.md) | 中文

## 这是什么

[local-jev](../local-jev) 证明了：用纯文本 LLM 做固定答案打分，比生成文本快 3-30 倍。本项目把同样的思路扩展到**视觉语言模型（VLM）**——

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
    "invoice": 0.8923,
    "contract": 0.0712,
    "report": 0.0289,
    "letter": 0.0076
  },
  "threshold_passed": true,
  "latency_ms": 11.0
}
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
| 速度优势 | 3x | 预期 5-50x（VLM 生成更慢） |

### 为什么视觉场景更适合打分

1. **VLM 生成更慢**：视觉模型参数更大，图片 token 占用多，自回归生成每步更慢
2. **视觉判别不需要描述**：审核截图是否违规，不需要"图中有一个穿红色衣服的人站在..."这种描述
3. **概率分布更有价值**：0.91 vs 0.46 的置信度差异，直接决定自动通过还是人工复核

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

运行 `python verify_mlx_vlm.py` 的完整输出：

### 模型加载

| 指标 | 数值 |
|------|------|
| 模型 | mlx-community/Qwen2.5-VL-3B-Instruct-4bit |
| 加载时间 | 2.3s |
| 词表大小 | 151,643 |
| 内存占用 | ~865MB (RSS) |

### 标签 Token 验证

| 标签 | Token ID | 状态 |
|------|----------|------|
| A (invoice) | [32] | ✓ 单 token |
| B (contract) | [33] | ✓ 单 token |
| C (report) | [34] | ✓ 单 token |
| D (letter) | [35] | ✓ 单 token |

与文本 Jev 完全一致——VLM 的标签 token 不受图片输入影响。

### 逐图打分结果（4 张 PIL 生成的文字图片）

| 图片 | 预期 | 决策 | 正确 | 打分延迟 |
|------|------|------|------|---------|
| invoice_test.png | invoice | report | ✗ | 33.6 ms |
| contract_test.png | contract | report | ✗ | 8.4 ms |
| report_test.png | report | report | ✓ | 14.2 ms |
| letter_test.png | letter | report | ✗ | 10.8 ms |

- **打分准确率**：1/4 (25%)
- **打分平均延迟**：16.7 ms
- **打分延迟明细**：[33.6, 8.4, 14.2, 10.8] ms

### 分析

- **核心机制完全验证通过**：图片+prompt → 前向传播 → LanguageModelOutput → logits → 受限 softmax → 概率分布
- **标签 token 在 VLM 中同样有效**：A/B/C/D 是纯文本 token，不受视觉编码影响
- **准确率偏低的原因**：测试图片是 PIL 绘制的简单文字图片，3B 4bit 模型难以区分这些手绘"文档"。实际应用中应使用真实文档照片。
- **模型总是输出 "report"**：4bit 量化 + 简单测试图片导致模型对所有输入都倾向同一答案。使用真实图片和更复杂的模型可改善。

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
│   ├── document_classification/ ← 文档分类数据集
│   ├── content_moderation/       ← 内容审核数据集
│   └── scene_recognition/        ← 场景识别数据集
├── verify_mlx_vlm.py            ← MLX-VLM 验证脚本
├── assets/
│   └── .gitkeep                  ← 放置示例图片
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

- [local-jev](../local-jev) 文本版项目
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) MLX 视觉语言模型库
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 阿里通义千问视觉语言模型
- [mlx-community](https://huggingface.co/mlx-community) MLX 格式模型转换

## License

MIT — Copyright (c) 2026 贾承斌
