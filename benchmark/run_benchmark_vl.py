#!/usr/bin/env python3
"""
Local VL-Jev — 视觉基准测试
对比 Jev 式视觉打分 vs 常规视觉生成的延迟和准确率。

用法:
    python benchmark/run_benchmark_vl.py
    python benchmark/run_benchmark_vl.py --dataset document_classification
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine_vl import LocalVLJevEngine


def run_benchmark(
    dataset: list[dict],
    engine: LocalVLJevEngine,
    choices: dict[str, str],
) -> dict:
    """双路径串行基准测试（视觉模型不适合并行，内存太大）"""
    results = {"scoring": [], "generation": []}

    for i, case in enumerate(dataset):
        image_path = case["image"]
        question = case["question"]
        expected = case["expected"]

        if not os.path.exists(image_path):
            print(f"  跳过 {image_path} (文件不存在)")
            continue

        # 打分路径
        try:
            score_result = engine.decide(image_path, question, choices)
            correct_s = (score_result.decision == expected)
            results["scoring"].append({
                "case_id": i,
                "image": image_path,
                "decision": score_result.decision,
                "expected": expected,
                "correct": correct_s,
                "latency_ms": score_result.latency_ms,
                "probabilities": score_result.probabilities,
            })
        except Exception as e:
            results["scoring"].append({
                "case_id": i, "image": image_path, "error": str(e),
                "correct": False, "latency_ms": 0,
            })

        # 生成路径
        try:
            gen_result = engine.generate_response(
                image_path, question, list(choices.items()), max_tokens=100
            )
            correct_g = (gen_result.parsed_choice == expected)
            results["generation"].append({
                "case_id": i,
                "image": image_path,
                "parsed_choice": gen_result.parsed_choice,
                "expected": expected,
                "correct": correct_g,
                "latency_ms": gen_result.latency_ms,
                "raw_response": gen_result.response,
            })
        except Exception as e:
            results["generation"].append({
                "case_id": i, "image": image_path, "error": str(e),
                "correct": False, "latency_ms": 0,
            })

    # 汇总
    summary = {}
    for lane_name, lane_results in results.items():
        latencies = [r["latency_ms"] for r in lane_results if r.get("latency_ms", 0) > 0]
        correct_count = sum(1 for r in lane_results if r.get("correct"))
        total = len(lane_results)
        summary[lane_name] = {
            "total_cases": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
        }

    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Local VL-Jev 视觉基准测试")
    parser.add_argument("--dataset", default="document_classification")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    args = parser.parse_args()

    dataset_path = f"datasets/{args.dataset}/cases.jsonl"
    if not os.path.exists(dataset_path):
        print(f"数据集不存在: {dataset_path}")
        print(f"请在 {dataset_path} 创建数据集文件")
        sys.exit(1)

    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(f"数据集: {args.dataset} ({len(dataset)} cases)")
    print(f"模型: {args.model}\n")

    # 每个数据集对应的候选选项
    DATASET_CHOICES = {
        "document_classification": {
            "A": "invoice",
            "B": "contract",
            "C": "report",
            "D": "letter",
        },
        "scene_recognition": {
            "A": "indoor",
            "B": "outdoor",
            "C": "night",
        },
        "content_moderation": {
            "A": "normal",
            "B": "violation",
            "C": "review",
        },
    }
    choices = DATASET_CHOICES.get(args.dataset)
    if choices is None:
        print(f"未知数据集 {args.dataset!r}，请在 DATASET_CHOICES 中登记候选选项")
        sys.exit(1)

    engine = LocalVLJevEngine(model_path=args.model)
    result = run_benchmark(dataset, engine, choices)

    print("\n" + "=" * 60)
    print("视觉基准测试结果")
    print("=" * 60)

    for lane, stats in result["summary"].items():
        lane_label = "Jev 式视觉打分" if lane == "scoring" else "常规视觉生成"
        print(f"\n【{lane_label}】")
        print(f"  准确率:   {stats['accuracy']:.2%} ({stats['correct']}/{stats['total_cases']})")
        print(f"  平均延迟: {stats['avg_latency_ms']:.1f} ms")
        print(f"  P50 延迟: {stats['p50_latency_ms']:.1f} ms")
        print(f"  最小延迟: {stats['min_latency_ms']:.1f} ms")
        print(f"  最大延迟: {stats['max_latency_ms']:.1f} ms")

    out_path = f"benchmark/results_{args.dataset}_{int(time.time())}.json"
    os.makedirs("benchmark", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
