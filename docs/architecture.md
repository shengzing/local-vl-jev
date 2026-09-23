# 架构详解

## 整体架构

```
                    ┌─────────────────┐
                    │  用户输入        │
                    │  (图片+问题+选项) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  构建 Prompt     │
                    │  A = invoice     │
                    │  B = contract    │
                    │  C = report      │
                    │  D = letter      │
                    │  ...Label:       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  /tokenize       │  验证标签是单 token
                    │  验证标签        │  拒绝非单 token
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  图片编码        │  ViT 提取视觉特征
                    │  + 文本 tokenize │  合并视觉+文本嵌入
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  前向传播         │  生成 vocabulary-sized logits
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  读取 logits     │  只取 A/B/C/D 位置
                    │  忽略其余         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  受限 Softmax    │  在选定 logits 间归一化
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  映射回选项       │  A→invoice, B→contract...
                    │  返回分布        │  + 阈值判断
                    └─────────────────┘
```

## 与文本 Jev 的关键差异

### 1. 输入处理

文本 Jev：`tokenizer(prompt) → input_ids`

视觉 Jev：
```
图片 → ViT 编码 → 视觉特征嵌入
文本 → tokenizer → 文本嵌入
合并 → 统一嵌入序列 → 前向传播
```

### 2. 视觉 token 开销

一张图片经 ViT 编码后，通常占用 256-1024+ 个 token 位置（取决于图片分辨率和 patch size）。这意味着：

- 前向传播的计算量更大（序列更长）
- 打分路径与生成路径**共享**这笔视觉编码 + 长 prefill 成本
- 生成路径在此之上还要自回归生成 token；生成 token 越多，打分的相对优势越大（实测：短回答 ≈1.0x，长描述 2-3x，详见 README 实测结果）

### 3. 图片真正进入模型的关键：`num_images=1`

mlx-vlm 0.7.x 的 `apply_chat_template` 渲染 multimodal 消息时，需要显式传 `num_images=1`，渲染结果才会包含 `<|vision_start|><|image_pad|><|vision_end|>` 占位符。漏传时模板静默退化为纯文本——模型收不到任何视觉信息，打分结果只反映语言先验（症状：所有图片得到相同分布）。本项目初版即踩过此坑，已修复。

### 4. 标签验证

与文本 Jev 完全相同。标签 A/B/C/D 的 token ID 不受图片输入影响——它们是纯文本 token，在词表中的位置固定。

## Qwen2.5-VL 架构

```
输入图片 → Patch Embedding → ViT Blocks (窗口注意力) → Patch Merger → 视觉嵌入
                                                                     ↓
输入文本 → Text Embedding ←─────────────────────────────────────────合并
                ↓
        Qwen2.5 LLM (因果注意力)
                ↓
        Logits (词表大小向量)
```

- ViT 处理图片，输出视觉嵌入
- 视觉嵌入替换 input_ids 中的 `<|image|>` 占位符
- 合并后的序列送入 Qwen2.5 LLM 做因果注意力
- 最后一个位置的 logits 包含了视觉+文本的完整理解

## 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 视觉决策客户端 | `src/decide_vl.py` | 图片输入、prompt 构建、打分、返回决策 |
| 双路径引擎 | `src/engine_vl.py` | 封装 scoring + generation 对比 |
| 工具函数 | `src/utils_vl.py` | prompt 构建、softmax、格式化 |
| 基准测试 | `benchmark/run_benchmark_vl.py` | 串行对比延迟和准确率 |

## 数据流

```
应用代码                  mlx-vlm 服务
    │                        │
    │── image+question+choices ─→ │
    │                        │── tokenize labels
    │←── token_ids ───────── │
    │                        │── 图片编码
    │                        │── 文本 tokenize
    │                        │── 前向传播
    │                        │── 读取 logits
    │                        │── softmax
    │←── probabilities ───── │
    │                        │
    │── 阈值判断 → 决策      │
```
