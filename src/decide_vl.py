#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VL-Jev — 视觉决策客户端
用 mlx-vlm + Qwen2.5-VL 做固定答案视觉打分：
输入图片+问题+候选选项，不生成文本，只读第一个 token 的 logits 做 softmax。

用法:
    python decide_vl.py --image assets/sample.jpg
    python decide_vl.py --image photo.png --question "这是什么类型的文档？"
    python decide_vl.py --image screenshot.png --choices invoice contract report letter

需要先安装:
    pip install mlx-vlm
"""

import argparse
import json
import math
import sys
import time
from typing import Optional

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template

# ── 配置 ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

# 默认选项：文档分类
DEFAULT_CHOICES = {
    "A": "invoice",
    "B": "contract",
    "C": "report",
    "D": "letter",
}

DEFAULT_QUESTION = "What type of document is this? Choose exactly one label."


# ── Prompt 构建 ────────────────────────────────────────────────────────

def build_vl_prompt(question: str, choices: dict) -> str:
    """构建视觉决策 prompt，末尾以 'Label: ' 结束。"""
    choice_lines = "\n".join(f"{label} = {meaning}" for label, meaning in choices.items())
    return f"""Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """


# ── 标签验证 ───────────────────────────────────────────────────────────

def resolve_label_token_ids(choices: dict, tokenizer) -> list:
    """用 tokenizer 验证每个标签是否恰好是单 token。"""
    token_ids = []
    for label in choices:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label {label!r} is not a single token: {ids}. "
                f"Use a different label that maps to exactly one token."
            )
        print(f"  {label!r} -> token_id {ids[0]}")
        token_ids.append(ids[0])
    return token_ids


# ── 打分核心 ───────────────────────────────────────────────────────────

def score_vl(
    model,
    processor,
    image_path: str,
    prompt: str,
    label_token_ids: list,
    config,
) -> tuple:
    """
    视觉打分：图片+prompt → 前向传播 → 取 logits → softmax。
    返回 (logits, probabilities, latency_ms)
    """
    from PIL import Image as PILImage

    # 构建 multimodal 消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 用 chat template 渲染
    # num_images=1 至关重要：缺少它时渲染结果不含 <|image_pad|> 占位符，
    # 图片不会被送进模型（mlx-vlm 0.7.x 行为）
    rendered = apply_chat_template(processor, config, messages, tokenize=False, num_images=1)

    # 处理输入 — mlx-vlm 0.7.x: 用 processor(text=..., images=[PIL.Image])
    pil_img = PILImage.open(image_path)
    inputs = processor(text=rendered, images=[pil_img], return_tensors='mlx')

    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")

    # 前向传播
    t0 = time.perf_counter()
    output = model(
        input_ids,
        pixel_values=pixel_values,
        mask=attention_mask,
        image_grid_thw=inputs.get("image_grid_thw"),
    )
    # LanguageModelOutput — 需要取 .logits 属性
    logits = output.logits if hasattr(output, 'logits') else output
    # MLX 是惰性求值：显式 eval 才能把真实计算时间计入延迟
    mx.eval(logits)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 取最后一个位置的 logits
    last_logits = logits[0, -1, :]

    # 提取标签位置的 logits
    selected_logits = [float(last_logits[tid]) for tid in label_token_ids]

    # 受限 softmax
    max_logit = max(selected_logits)
    exp_vals = [math.exp(l - max_logit) for l in selected_logits]
    sum_exp = sum(exp_vals)
    probs = [v / sum_exp for v in exp_vals]

    return selected_logits, probs, latency_ms


# ── 决策 ───────────────────────────────────────────────────────────────

def decide_vl(
    image_path: str,
    question: str = DEFAULT_QUESTION,
    choices: Optional[dict] = None,
    model_path: str = DEFAULT_MODEL,
) -> dict:
    """完整视觉决策流程：加载模型 → 构建prompt → 验证标签 → 打分 → 返回决策。"""
    choices = choices or DEFAULT_CHOICES

    print(f"[1/4] 加载模型 {model_path}...")
    model, processor = load(model_path)
    config = model.config

    print("[2/4] 构建视觉 prompt...")
    prompt = build_vl_prompt(question, choices)
    print(f"  问题: {question}")
    print(f"  图片: {image_path}")
    print(f"  选项: {list(choices.values())}")

    print("[3/4] 验证标签 token...")
    text_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    label_token_ids = resolve_label_token_ids(choices, text_tokenizer)

    print("[4/4] 视觉打分...")
    logits, probs, latency_ms = score_vl(
        model, processor, image_path, prompt, label_token_ids, config
    )

    probabilities = {
        choices[label]: prob
        for label, prob in zip(choices, probs)
    }
    decision = max(probabilities, key=probabilities.get)
    top_prob = probabilities[decision]

    THRESHOLD = 0.70
    threshold_passed = top_prob >= THRESHOLD

    result = {
        "decision": decision,
        "probabilities": probabilities,
        "threshold_passed": threshold_passed,
        "latency_ms": latency_ms,
    }
    if not threshold_passed:
        result["warning"] = (
            f"Top probability {top_prob:.4f} < threshold {THRESHOLD}. "
            f"Consider sending for human review."
        )

    return result


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local VL-Jev — 用 VLM 做固定答案视觉打分"
    )
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="问题文本")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型路径")
    parser.add_argument(
        "--choices",
        nargs="+",
        default=["invoice", "contract", "report", "letter"],
        help="候选选项（按顺序对应 A/B/C/D）",
    )
    args = parser.parse_args()

    labels = [chr(ord("A") + i) for i in range(len(args.choices))]
    choices = dict(zip(labels, args.choices))

    print(f"模型: {args.model}")
    print(f"图片: {args.image}")
    print(f"选项: {choices}\n")

    result = decide_vl(args.image, args.question, choices, args.model)
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
