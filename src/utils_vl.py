#!/usr/bin/env python3
"""
Local VL-Jev — 工具函数
"""

import json
import math
from typing import Optional


def build_decision_prompt(question: str, choices: dict[str, str]) -> str:
    """构建视觉决策 prompt"""
    choice_lines = "\n".join(f"{k} = {v}" for k, v in choices.items())
    return f"""Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """


def validate_label_tokens(
    choices: dict[str, str],
    tokenizer,
) -> dict[str, int]:
    """验证标签是否单 token"""
    result = {}
    for label in choices:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {label!r} is not single token: {ids}")
        result[label] = ids[0]
    return result


def restricted_softmax(logits: list[float]) -> list[float]:
    """受限 softmax：只在选定 logits 间归一化"""
    max_logit = max(logits)
    exp_vals = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_vals)
    return [v / sum_exp for v in exp_vals]


def format_result(
    decision: str,
    probabilities: dict[str, float],
    threshold: float = 0.70,
    latency_ms: float = 0,
) -> str:
    """格式化输出结果"""
    top_prob = probabilities[decision]
    status = "✓ AUTO" if top_prob >= threshold else "⚠ REVIEW"

    lines = [
        f"\n{'='*50}",
        f"Decision: {decision}",
        f"Confidence: {top_prob:.4f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Status: {status}",
        f"{'='*50}",
        "\nProbabilities:",
    ]
    for choice, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        lines.append(f"  {choice:25s} {prob:.4f} {bar}")

    if top_prob < threshold:
        lines.append(f"\n⚠ Top probability {top_prob:.4f} < threshold {threshold}")
        lines.append("  Recommend human review.")

    return "\n".join(lines)


def load_dataset(path: str) -> list[dict]:
    """加载 JSONL 数据集"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
