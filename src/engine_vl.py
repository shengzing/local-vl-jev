#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VL-Jev — 双路径引擎
封装 visual scoring（Jev 式打分）和 visual generation（常规视觉文本生成）。
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template


@dataclass
class VLDecisionResult:
    """视觉打分路径返回结果"""
    decision: str
    probabilities: dict
    latency_ms: float


@dataclass
class VLGenerationResult:
    """视觉生成路径返回结果"""
    response: str
    parsed_choice: Optional[str]
    latency_ms: float


class LocalVLJevEngine:
    """视觉双路径推理引擎"""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    ):
        print(f"Loading model: {model_path}")
        self.model, self.processor = load(model_path)
        self.config = self.model.config
        self._label_cache: dict = {}

    def _build_prompt(self, question: str, choices: dict) -> str:
        choice_lines = "\n".join(f"{k} = {v}" for k, v in choices.items())
        return f"""Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """

    def _resolve_token_ids(self, choices: dict) -> list:
        """验证并缓存标签 token IDs"""
        cache_key = "".join(choices.keys())
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        token_ids = []
        for label in choices:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Label {label!r} is not single token: {ids}")
            token_ids.append(ids[0])

        self._label_cache[cache_key] = token_ids
        return token_ids

    def decide(
        self,
        image_path: str,
        question: str,
        choices: dict,
    ) -> VLDecisionResult:
        """
        Jev 式视觉打分：图片+prompt → 前向传播 → 取 logits → softmax。
        不生成任何 token。
        """
        from PIL import Image as PILImage

        prompt = self._build_prompt(question, choices)
        token_ids = self._resolve_token_ids(choices)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # num_images=1 至关重要：缺少它时渲染结果不含 <|image_pad|> 占位符，
        # 图片不会被送进模型（mlx-vlm 0.7.x 行为）
        rendered = apply_chat_template(
            self.processor, self.config, messages, tokenize=False, num_images=1
        )
        pil_img = PILImage.open(image_path)
        inputs = self.processor(text=rendered, images=[pil_img], return_tensors='mlx')

        t0 = time.perf_counter()
        output = self.model(
            inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            mask=inputs.get("attention_mask"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = output.logits if hasattr(output, 'logits') else output
        # MLX 是惰性求值：显式 eval 才能把真实计算时间计入延迟
        mx.eval(logits)
        latency_ms = (time.perf_counter() - t0) * 1000

        last_logits = logits[0, -1, :]
        selected = [float(last_logits[tid]) for tid in token_ids]

        max_logit = max(selected)
        exp_vals = [math.exp(l - max_logit) for l in selected]
        sum_exp = sum(exp_vals)
        probs = [v / sum_exp for v in exp_vals]

        probabilities = {
            choices[label]: prob
            for label, prob in zip(choices, probs)
        }
        decision = max(probabilities, key=probabilities.get)

        return VLDecisionResult(
            decision=decision,
            probabilities=probabilities,
            latency_ms=latency_ms,
        )

    def generate_response(
        self,
        image_path: str,
        question: str,
        choices: list,
        max_tokens: int = 100,
    ) -> VLGenerationResult:
        """常规视觉生成路径：让 VLM 生成文本，后解析选项。"""
        from PIL import Image as PILImage

        choice_lines = "\n".join(f"{k} = {v}" for k, v in choices.items())
        prompt = f"Question: {question}\n\nLabels:\n{choice_lines}\n\nReply with only the label letter."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # 同 decide()：num_images=1 保证图片真正进入模型
        rendered = apply_chat_template(
            self.processor, self.config, messages, tokenize=False, num_images=1
        )

        t0 = time.perf_counter()
        output = generate(
            self.model, self.processor,
            prompt=rendered,
            image=[image_path],
            max_tokens=max_tokens,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # mlx-vlm 0.7.x 返回 GenerationResult 对象而非 str
        response = output.text if hasattr(output, "text") else str(output)

        parsed = None
        search_text = response[:200].lower()
        for label, meaning in choices.items():
            if label.lower() in search_text or meaning.lower() in search_text:
                parsed = meaning
                break

        return VLGenerationResult(
            response=response,
            parsed_choice=parsed,
            latency_ms=latency_ms,
        )
