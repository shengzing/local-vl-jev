#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VL-Jev — MLX-VLM 验证脚本
用 mlx-vlm + Qwen2.5-VL-3B 在 Mac M4 上验证视觉决策打分。
"""

import json
import math
import os
import sys
import time
import traceback

import mlx.core as mx
from mlx_vlm import load, generate, apply_chat_template

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

print("=" * 60)
print("Local VL-Jev — MLX-VLM 验证")
print("=" * 60)

# ── 加载模型 ─────────────────────────────────────────────
print(f"\n[1/5] 加载模型 {MODEL}...")
t0 = time.perf_counter()
model, processor = load(MODEL)
config = model.config
load_time = time.perf_counter() - t0
print(f"  加载时间: {load_time:.1f}s")

# 获取 tokenizer
text_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
print(f"  词表大小: {text_tokenizer.vocab_size}")

# ── 验证标签 token ────────────────────────────────────────
print("\n[2/5] 验证标签 token...")
choices = {
    "A": "invoice",
    "B": "contract",
    "C": "report",
    "D": "letter",
}
label_token_ids = []
for label in choices:
    ids = text_tokenizer.encode(label, add_special_tokens=False)
    ok = "✓" if len(ids) == 1 else "✗"
    print(f"  {ok} '{label}' -> {ids}")
    if len(ids) != 1:
        raise ValueError(f"Label {label!r} not single token: {ids}")
    label_token_ids.append(ids[0])
print(f"  打分位置: {label_token_ids}")

# ── 创建测试图片 ──────────────────────────────────────────
print("\n[3/5] 创建测试图片...")
from PIL import Image, ImageDraw, ImageFont
import tempfile

tmpdir = tempfile.mkdtemp(prefix="vl_jev_test_")
test_images = []
for name, label_text, bg_color in [
    ("invoice", "INVOICE\nAmount: $250\nDate: 2025-01-15", (255, 255, 240)),
    ("contract", "CONTRACT\nParty A agrees to...", (230, 240, 255)),
    ("report", "QUARTERLY REPORT\nRevenue: $1.2M\nGrowth: +15%", (240, 255, 240)),
    ("letter", "Dear Sir/Madam,\nWe are pleased to...", (255, 240, 230)),
]:
    img = Image.new("RGB", (400, 300), bg_color)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(label_text.split("\n")):
        draw.text((20, 20 + i * 30), line, fill="black")
    path = os.path.join(tmpdir, f"{name}_test.png")
    img.save(path)
    test_images.append((path, name))
print(f"  创建了 {len(test_images)} 张测试图片")

# ── 打分函数 ────────────────────────────────────────────────
def score_one_image(image_path, question, choices):
    """Jev 式视觉打分路径"""
    from PIL import Image as PILImage

    choice_lines = "\n".join(f"{label} = {meaning}" for label, meaning in choices.items())
    prompt = f"""Question:
{question}

Allowed labels:
{choice_lines}

Return only the label.

Label: """

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
    rendered = apply_chat_template(processor, config, messages, tokenize=False, num_images=1)

    pil_img = PILImage.open(image_path)
    inputs = processor(text=rendered, images=[pil_img], return_tensors='mlx')

    t0 = time.perf_counter()
    output = model(
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
    selected = [float(last_logits[tid]) for tid in label_token_ids]

    max_logit = max(selected)
    exp_vals = [math.exp(l - max_logit) for l in selected]
    sum_exp = sum(exp_vals)
    probs = [v / sum_exp for v in exp_vals]

    return selected, probs, latency_ms

# ── 生成函数 ────────────────────────────────────────────────
def generate_one_image(image_path, question, choices, max_tokens=50):
    """常规视觉生成路径"""
    from PIL import Image as PILImage

    choice_lines = "\n".join(f"{label} = {meaning}" for label, meaning in choices.items())
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
    # num_images=1：同打分路径，保证图片进入模型
    rendered = apply_chat_template(processor, config, messages, tokenize=False, num_images=1)

    pil_img = PILImage.open(image_path)
    t0 = time.perf_counter()
    output = generate(
        model, processor,
        prompt=rendered,
        image=[image_path],
        max_tokens=max_tokens,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    # mlx-vlm 0.7.x 返回 GenerationResult 对象而非 str
    response = output.text if hasattr(output, "text") else str(output)
    return response, latency_ms

# ── 测试每个图片 ─────────────────────────────────────────
print("\n[4/5] 逐图测试（打分 vs 生成）...\n")

question = "What type of document is this? Choose exactly one label."

print(f"  {'Image':<25} {'Expected':<12} {'Decision':<12} {'OK':>3} {'Score(ms)':>10} {'Gen(ms)':>10}")
print(f"  {'-'*82}")

score_correct = 0
gen_correct = 0
score_latencies = []
gen_latencies = []
gen_results = []

for image_path, expected in test_images:
    img_name = os.path.basename(image_path)
    # 打分
    try:
        logits, probs, s_lat = score_one_image(image_path, question, choices)
        prob_map = dict(zip(choices.values(), probs))
        decision = max(prob_map, key=prob_map.get)
        ok = "✓" if decision == expected else "✗"
        if ok == "✓":
            score_correct += 1
    except Exception as e:
        decision = f"ERR:{str(e)[:8]}"
        ok = "✗"
        s_lat = 0
    score_latencies.append(s_lat)

    # 生成
    try:
        gen_text, g_lat = generate_one_image(image_path, question, choices)
        search = gen_text[:200].lower()
        gen_decision = "unknown"
        for label, meaning in choices.items():
            if label.lower() in search or meaning.lower() in search:
                gen_decision = meaning
                break
        gen_ok = "✓" if gen_decision == expected else "✗"
        if gen_ok == "✓":
            gen_correct += 1
    except Exception as e:
        gen_text = f"ERR:{str(e)[:20]}"
        gen_decision = "error"
        gen_ok = "✗"
        g_lat = 0
    gen_latencies.append(g_lat)
    gen_results.append((img_name, gen_text[:50]))

    print(f"  {img_name:<25} {expected:<12} {decision:<12} {ok:>3} {s_lat:>10.1f} {g_lat:>10.1f}")

# ── 汇总 ─────────────────────────────────────────────────
print("\n[5/5] 汇总结果")
print("=" * 60)

print(f"\n  {'指标':<35} {'视觉打分':<15} {'视觉生成':<15}")
print(f"  {'-'*65}")
print(f"  {'准确率':.<35} {score_correct}/{len(test_images):<13} {gen_correct}/{len(test_images)}")

avg_s = sum(score_latencies) / len(score_latencies) if score_latencies else 0
avg_g = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0
print(f"  {'平均延迟 (ms)':.<35} {avg_s:<15.1f} {avg_g:<15.1f}")

if avg_s > 0:
    print(f"  {'速度倍数':.<35} {avg_g / avg_s:.1f}x")

print(f"\n  打分延迟明细: {[round(l, 1) for l in score_latencies]}")
print(f"  生成延迟明细: {[round(l, 1) for l in gen_latencies]}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("""
  1. VL-Jev 核心机制验证：图片+prompt → 前向传播 → logits → softmax
  2. 标签 token IDs 在 VLM 中同样有效（A/B/C/D 是纯文本 token）
  3. 视觉打分只需一次前向传播，不生成 token
  4. 视觉生成需要自回归循环，延迟更高
  5. 速度优势在 VLM 场景下预期比纯文本更大

  注意：4bit 量化 + 3B 模型 + 简单测试图片可能导致准确率偏低。
  实际应用中应使用真实图片并选择适当大小的模型。
""")

print("=" * 60)

# 清理测试图片
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
print("\n测试图片已清理。")
