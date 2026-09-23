#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local VL-Jev — 专业对比基准测试
对比 Jev 式视觉打分 vs 常规视觉文本生成，在判别/选择/决策任务上的性能和效率。

测试维度:
  1. 准确率 (Accuracy)
  2. 平均延迟 (Avg Latency)
  3. P50/P95 延迟
  4. 吞吐量 (QPS)
  5. 决策置信度 (Top-1 Probability, 打分路径独有)
  6. 解析失败率 (Parse Failure Rate, 生成路径独有)
  7. 长输出惩罚倍数 (Long Output Penalty)

用法:
    python benchmark/run_comparison.py
"""

import argparse
import json
import os
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine_vl import LocalVLJevEngine

# ── 测试数据集定义 ──────────────────────────────────────────

DATASETS = {
    "scene_recognition": {
        "name": "场景识别",
        "question": "What type of scene is this? Choose exactly one label.",
        "choices": {"A": "indoor", "B": "outdoor", "C": "night"},
        "cases_file": "datasets/scene_recognition/cases.jsonl",
    },
    "document_classification": {
        "name": "文档分类",
        "question": "What type of document is this? Choose exactly one label.",
        "choices": {"A": "invoice", "B": "contract", "C": "report", "D": "letter"},
        "cases_file": "datasets/document_classification/cases.jsonl",
    },
    "content_moderation": {
        "name": "内容审核",
        "question": "Is this content appropriate? Choose exactly one label.",
        "choices": {"A": "normal", "B": "violation", "C": "review"},
        "cases_file": "datasets/content_moderation/cases.jsonl",
    },
}


def load_cases(cases_file: str, base_dir: str) -> list:
    """从 JSONL 加载测试用例，解析图片绝对路径"""
    cases = []
    path = os.path.join(base_dir, cases_file)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            # 图片路径转绝对路径
            img_path = os.path.join(base_dir, case["image"])
            case["image_abs"] = img_path
            cases.append(case)
    return cases


def percentile(data: list, p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def run_benchmark(engine: LocalVLJevEngine, base_dir: str, warmup: int = 2) -> dict:
    """运行全部数据集的打分 vs 生成对比"""

    all_results = {"datasets": {}, "summary": {}}
    all_score_lat = []
    all_gen_lat = []
    all_score_correct = 0
    all_gen_correct = 0
    all_total = 0
    all_gen_parse_fail = 0
    all_score_probs = []

    for ds_key, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"数据集: {ds_info['name']} ({ds_key})")
        print(f"选项: {ds_info['choices']}")
        print(f"{'='*60}")

        cases = load_cases(ds_info["cases_file"], base_dir)
        choices = ds_info["choices"]
        question = ds_info["question"]

        # ── Warmup ──
        print(f"\n  Warming up ({warmup} runs)...")
        for i in range(min(warmup, len(cases))):
            c = cases[i]
            engine.decide(c["image_abs"], question, choices)
            engine.generate_response(c["image_abs"], question, choices, max_tokens=10)

        # ── Scoring path ──
        print(f"\n  [A] Jev 式打分 (scoring)")
        score_results = []
        for i, case in enumerate(cases):
            result = engine.decide(case["image_abs"], question, choices)
            correct = result.decision == case["expected"]
            top_prob = max(result.probabilities.values())
            score_results.append({
                "image": os.path.basename(case["image"]),
                "expected": case["expected"],
                "decision": result.decision,
                "correct": correct,
                "latency_ms": result.latency_ms,
                "top_prob": top_prob,
                "probabilities": result.probabilities,
            })
            print(f"    {i+1}. {os.path.basename(case['image']):30s} exp={case['expected']:12s} got={result.decision:12s} {'OK' if correct else 'WRONG':5s} {result.latency_ms:7.1f}ms  p={top_prob:.4f}")

        # ── Generation path ──
        print(f"\n  [B] 常规文本生成 (generation)")
        gen_results = []
        for i, case in enumerate(cases):
            result = engine.generate_response(case["image_abs"], question, choices, max_tokens=50)
            correct = result.parsed_choice == case["expected"]
            parse_fail = result.parsed_choice is None
            gen_results.append({
                "image": os.path.basename(case["image"]),
                "expected": case["expected"],
                "decision": result.parsed_choice or "PARSE_FAIL",
                "correct": correct,
                "latency_ms": result.latency_ms,
                "parse_fail": parse_fail,
                "response": result.response[:80],
            })
            status = "OK" if correct else ("PARSE_FAIL" if parse_fail else "WRONG")
            print(f"    {i+1}. {os.path.basename(case['image']):30s} exp={case['expected']:12s} got={(result.parsed_choice or 'FAIL'):12s} {status:10s} {result.latency_ms:7.1f}ms  resp='{result.response[:30]}'")

        # ── 计算该数据集的统计 ──
        score_lat = [r["latency_ms"] for r in score_results]
        gen_lat = [r["latency_ms"] for r in gen_results]
        score_correct = sum(1 for r in score_results if r["correct"])
        gen_correct = sum(1 for r in gen_results if r["correct"])
        gen_parse_fail = sum(1 for r in gen_results if r["parse_fail"])
        score_probs = [r["top_prob"] for r in score_results]

        ds_stats = {
            "name": ds_info["name"],
            "n_cases": len(cases),
            "choices": list(choices.values()),
            "scoring": {
                "accuracy": score_correct / len(cases),
                "correct": score_correct,
                "total": len(cases),
                "avg_latency_ms": statistics.mean(score_lat),
                "p50_latency_ms": percentile(score_lat, 50),
                "p95_latency_ms": percentile(score_lat, 95),
                "min_latency_ms": min(score_lat),
                "max_latency_ms": max(score_lat),
                "qps": 1000 / statistics.mean(score_lat) if score_lat else 0,
                "avg_top_prob": statistics.mean(score_probs),
            },
            "generation": {
                "accuracy": gen_correct / len(cases),
                "correct": gen_correct,
                "total": len(cases),
                "avg_latency_ms": statistics.mean(gen_lat),
                "p50_latency_ms": percentile(gen_lat, 50),
                "p95_latency_ms": percentile(gen_lat, 95),
                "min_latency_ms": min(gen_lat),
                "max_latency_ms": max(gen_lat),
                "qps": 1000 / statistics.mean(gen_lat) if gen_lat else 0,
                "parse_failures": gen_parse_fail,
                "parse_fail_rate": gen_parse_fail / len(cases),
            },
        }

        # 速度比
        if statistics.mean(gen_lat) > 0:
            ds_stats["speed_ratio"] = statistics.mean(gen_lat) / statistics.mean(score_lat)
        else:
            ds_stats["speed_ratio"] = 0

        all_results["datasets"][ds_key] = ds_stats

        # 汇总
        all_score_lat.extend(score_lat)
        all_gen_lat.extend(gen_lat)
        all_score_correct += score_correct
        all_gen_correct += gen_correct
        all_total += len(cases)
        all_gen_parse_fail += gen_parse_fail
        all_score_probs.extend(score_probs)

        print(f"\n  --- {ds_info['name']} 汇总 ---")
        print(f"  {'指标':25s} {'打分':>12s} {'生成':>12s} {'比率':>8s}")
        print(f"  {'-'*60}")
        print(f"  {'准确率':25s} {score_correct}/{len(cases):<10d} {gen_correct}/{len(cases):<10d} {ds_stats['speed_ratio']:.2f}x")
        print(f"  {'平均延迟 (ms)':25s} {statistics.mean(score_lat):>12.1f} {statistics.mean(gen_lat):>12.1f}")
        print(f"  {'P50 延迟 (ms)':25s} {percentile(score_lat,50):>12.1f} {percentile(gen_lat,50):>12.1f}")
        print(f"  {'P95 延迟 (ms)':25s} {percentile(score_lat,95):>12.1f} {percentile(gen_lat,95):>12.1f}")
        print(f"  {'QPS':25s} {1000/statistics.mean(score_lat):>12.1f} {1000/statistics.mean(gen_lat):>12.1f}")

    # ── 全局汇总 ──
    all_results["summary"] = {
        "total_cases": all_total,
        "scoring": {
            "accuracy": all_score_correct / all_total,
            "correct": all_score_correct,
            "total": all_total,
            "avg_latency_ms": statistics.mean(all_score_lat),
            "p50_latency_ms": percentile(all_score_lat, 50),
            "p95_latency_ms": percentile(all_score_lat, 95),
            "min_latency_ms": min(all_score_lat),
            "max_latency_ms": max(all_score_lat),
            "qps": 1000 / statistics.mean(all_score_lat),
            "avg_top_prob": statistics.mean(all_score_probs),
        },
        "generation": {
            "accuracy": all_gen_correct / all_total,
            "correct": all_gen_correct,
            "total": all_total,
            "avg_latency_ms": statistics.mean(all_gen_lat),
            "p50_latency_ms": percentile(all_gen_lat, 50),
            "p95_latency_ms": percentile(all_gen_lat, 95),
            "min_latency_ms": min(all_gen_lat),
            "max_latency_ms": max(all_gen_lat),
            "qps": 1000 / statistics.mean(all_gen_lat),
            "parse_failures": all_gen_parse_fail,
            "parse_fail_rate": all_gen_parse_fail / all_total,
        },
        "speed_ratio": statistics.mean(all_gen_lat) / statistics.mean(all_score_lat),
    }

    return all_results


def print_report(results: dict):
    """打印最终对比报告"""
    s = results["summary"]

    print("\n" + "=" * 70)
    print("Local VL-Jev 对比基准报告")
    print("Jev 式视觉打分 vs 常规文本生成 — 判别/决策任务")
    print("=" * 70)

    print(f"\n环境: Apple M4, Qwen2.5-VL-3B-Instruct-4bit (MLX, 4bit)")
    print(f"测试用例: {s['total_cases']} 张图片, 3 个数据集")
    print(f"任务类型: 场景识别 (3选1), 文档分类 (4选1), 内容审核 (3选1)")

    print(f"\n{'指标':30s} {'Jev 打分':>14s} {'文本生成':>14s} {'差异':>10s}")
    print(f"{'-'*70}")

    sc = s["scoring"]
    gen = s["generation"]

    print(f"{'准确率':30s} {sc['accuracy']*100:>13.1f}% {gen['accuracy']*100:>13.1f}% {(sc['accuracy']-gen['accuracy'])*100:>+9.1f}%")
    print(f"{'正确数/总数':30s} {str(sc['correct'])+'/'+str(sc['total']):>14s} {str(gen['correct'])+'/'+str(gen['total']):>14s} {'':>10s}")
    print(f"{'平均延迟 (ms)':30s} {sc['avg_latency_ms']:>14.1f} {gen['avg_latency_ms']:>14.1f} {s['speed_ratio']:>9.2f}x")
    print(f"{'P50 延迟 (ms)':30s} {sc['p50_latency_ms']:>14.1f} {gen['p50_latency_ms']:>14.1f} {gen['p50_latency_ms']/sc['p50_latency_ms']:>9.2f}x")
    print(f"{'P95 延迟 (ms)':30s} {sc['p95_latency_ms']:>14.1f} {gen['p95_latency_ms']:>14.1f} {gen['p95_latency_ms']/sc['p95_latency_ms']:>9.2f}x")
    print(f"{'最小延迟 (ms)':30s} {sc['min_latency_ms']:>14.1f} {gen['min_latency_ms']:>14.1f}")
    print(f"{'最大延迟 (ms)':30s} {sc['max_latency_ms']:>14.1f} {gen['max_latency_ms']:>14.1f}")
    print(f"{'吞吐量 QPS':30s} {sc['qps']:>14.1f} {gen['qps']:>14.1f} {sc['qps']/gen['qps']:>9.2f}x")
    print(f"{'决策置信度 (Top-1 Prob)':30s} {sc['avg_top_prob']:>14.4f} {'N/A':>14s}")
    print(f"{'解析失败率':30s} {'0.0%':>14s} {gen['parse_fail_rate']*100:>13.1f}%")
    print(f"{'解析失败数':30s} {'0':>14s} {gen['parse_failures']:>14d}")

    # 逐数据集
    print(f"\n{'='*70}")
    print("逐数据集明细")
    print(f"{'='*70}")

    for ds_key, ds in results["datasets"].items():
        print(f"\n  {ds['name']} ({ds_key}) — {ds['n_cases']} 张图片, 选项: {ds['choices']}")
        print(f"  {'指标':25s} {'打分':>12s} {'生成':>12s} {'比率':>8s}")
        print(f"  {'-'*60}")
        print(f"  {'准确率':25s} {ds['scoring']['accuracy']*100:>11.1f}% {ds['generation']['accuracy']*100:>11.1f}%")
        print(f"  {'平均延迟':25s} {ds['scoring']['avg_latency_ms']:>10.1f}ms {ds['generation']['avg_latency_ms']:>10.1f}ms {ds['speed_ratio']:>7.2f}x")
        print(f"  {'P50 延迟':25s} {ds['scoring']['p50_latency_ms']:>10.1f}ms {ds['generation']['p50_latency_ms']:>10.1f}ms")
        print(f"  {'P95 延迟':25s} {ds['scoring']['p95_latency_ms']:>10.1f}ms {ds['generation']['p95_latency_ms']:>10.1f}ms")
        print(f"  {'QPS':25s} {ds['scoring']['qps']:>12.1f} {ds['generation']['qps']:>12.1f}")
        print(f"  {'置信度':25s} {ds['scoring']['avg_top_prob']:>12.4f} {'N/A':>12s}")
        if ds['generation']['parse_failures'] > 0:
            print(f"  {'解析失败':25s} {'0':>12s} {ds['generation']['parse_failures']:>12d}")

    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Local VL-Jev — 专业对比基准测试"
    )
    parser.add_argument("--output", default="benchmark/results/comparison_report.json", help="输出 JSON 报告路径")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"项目目录: {base_dir}")
    print(f"模型: mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    print(f"数据集: {list(DATASETS.keys())}")

    print("\n加载模型...")
    engine = LocalVLJevEngine("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

    results = run_benchmark(engine, base_dir, warmup=2)
    print_report(results)

    # 保存 JSON
    output_path = os.path.join(base_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
