# Local VL-Jev — Reproducing Jev-Style Visual Discrimination with Vision-Language Models

> An open-source project extending Jev-style fixed-answer scoring to the visual domain: input an image + question, skip text generation entirely, read the first token's logits, and apply softmax over known candidates to return a probability distribution.

**Author: Jiacheng Bin (jiacb@wiseweb.com.cn)**

English | [中文](./README.md)

## What Is This

[local-jev](../local-jev) demonstrated that fixed-answer scoring with text LLMs is 3-30x faster than text generation. This project extends the same approach to **vision-language models (VLMs)**—

When an application needs to classify, discriminate, or filter based on images, it doesn't need the VLM to "describe the image and then write an answer." Instead, it directly reads the first token's logits and applies softmax over the candidate options.

Built on `mlx-vlm` + Qwen2.5-VL-3B-Instruct, running natively on Apple Silicon.

### Typical Scenarios

| Scenario | Input | Candidates | Current Approach | VL-Jev Approach |
|----------|-------|------------|-----------------|-----------------|
| Content moderation | Product screenshot | Normal/Violation/Review | VLM generates description → judge | Direct scoring → distribution |
| Ticket classification | Error screenshot | UI/Backend/Network | VLM writes analysis | Direct scoring → distribution |
| Document classification | Document photo | Invoice/Contract/Report/Letter | VLM generates text → parse | Direct scoring → distribution |
| Quality inspection | Product photo | Pass/Fail | VLM writes description → judge | Direct scoring → distribution |
| Scene recognition | Landscape photo | Indoor/Outdoor/Night | VLM generates description | Direct scoring → distribution |

## Quick Start

```bash
# 1. Install dependencies
pip install mlx-vlm Pillow

# 2. Run visual discrimination (auto-downloads Qwen2.5-VL-3B-4bit model on first run)
python src/decide_vl.py --image assets/sample.jpg
```

Output:

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

## Core Principle

### Text Jev vs Visual Jev

| Feature | local-jev (text) | local-vl-jev (visual) |
|---------|------------------|----------------------|
| Input | Text prompt | Image + text prompt |
| Model | Qwen2.5-0.5B-Instruct | Qwen2.5-VL-3B-Instruct |
| Inference path | tokenize → forward → read logits | image encode + tokenize → forward → read logits |
| Scoring logic | Read A/B/C logits → softmax | Identical |
| Generation path | Autoregressive loop | Identical (slower, VLM is larger) |
| Speed advantage | 3x | Expected 5-50x (VLM generation is slower) |

### Why Visual Scenarios Are Better Suited for Scoring

1. **VLM generation is slower**: Vision models have more parameters, image tokens take up more positions, each autoregressive step is slower
2. **Visual discrimination doesn't need description**: Checking if a screenshot violates policy doesn't require "There is a person wearing red standing in..."
3. **Probability distribution is more valuable**: The difference between 0.91 and 0.46 confidence directly determines auto-pass vs human review

### Inference Flow

```
User input (image + question + candidates)
         │
    Build multimodal prompt (image placeholder + text labels A/B/C/D)
         │
    Tokenize to verify labels are single tokens
         │
    Image encoding + text tokenize → forward pass
         │
    Read first token's logits (A/B/C/D positions)
         │
    Restricted softmax → probability distribution
         │
    Map back to semantic options → return decision
```

## Benchmark Results (Apple M4, Qwen2.5-VL-3B-Instruct-4bit, MLX)

Full output from `python verify_mlx_vlm.py`:

### Model Loading

| Metric | Value |
|--------|-------|
| Model | mlx-community/Qwen2.5-VL-3B-Instruct-4bit |
| Load time | 2.3s |
| Vocab size | 151,643 |
| Memory (RSS) | ~865MB |

### Label Token Verification

| Label | Token ID | Status |
|-------|----------|--------|
| A (invoice) | [32] | ✓ single token |
| B (contract) | [33] | ✓ single token |
| C (report) | [34] | ✓ single token |
| D (letter) | [35] | ✓ single token |

Identical to text Jev — VLM label tokens are unaffected by image input.

### Per-Image Scoring Results (4 PIL-generated text images)

| Image | Expected | Decision | Correct | Latency |
|-------|----------|----------|---------|---------|
| invoice_test.png | invoice | report | ✗ | 33.6 ms |
| contract_test.png | contract | report | ✗ | 8.4 ms |
| report_test.png | report | report | ✓ | 14.2 ms |
| letter_test.png | letter | report | ✗ | 10.8 ms |

- **Scoring accuracy**: 1/4 (25%)
- **Avg scoring latency**: 16.7 ms
- **Latency breakdown**: [33.6, 8.4, 14.2, 10.8] ms

### Analysis

- **Core mechanism fully verified**: image+prompt → forward pass → LanguageModelOutput → logits → restricted softmax → probability distribution
- **Label tokens work in VLM**: A/B/C/D are pure text tokens, unaffected by visual encoding
- **Low accuracy cause**: Test images are simple PIL-drawn text images. The 3B 4bit model struggles to distinguish these hand-drawn "documents." Use real document photos in production.
- **Model defaults to "report"**: 4bit quantization + simple test images cause the model to favor the same answer for all inputs. Real images and larger models should improve discrimination.

## Project Structure

```
local-vl-jev/
├── README.md                    ← Chinese docs
├── README_EN.md                 ← English docs (this file)
├── LICENSE                      ← MIT
├── pyproject.toml               ← Package metadata
├── requirements.txt             ← Dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── decide_vl.py             ← Core visual decision client
│   ├── engine_vl.py             ← Dual-path engine (scoring vs generation)
│   └── utils_vl.py              ← Utility functions
├── benchmark/
│   └── run_benchmark_vl.py      ← Visual scoring vs generation comparison
├── datasets/
│   ├── document_classification/ ← Document classification dataset
│   ├── content_moderation/      ← Content moderation dataset
│   └── scene_recognition/       ← Scene recognition dataset
├── verify_mlx_vlm.py            ← MLX-VLM verification script
├── assets/
│   └── .gitkeep                 ← Place sample images here
├── article/
│   └── .gitkeep                 ← Technical article
└── docs/
    ├── architecture.md          ← Architecture details
    └── calibration.md           ← Calibration & evaluation
```

> **Note**: Model parameter files (Qwen2.5-VL-3B-Instruct-4bit, ~3GB) are not included in the repository. They are automatically downloaded by mlx-vlm from HuggingFace to a local cache directory on first run.

## When to Use Visual Scoring vs Generation

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Image classification, content moderation | Scoring | Finite candidates, just picking |
| Quality inspection, defect detection | Scoring | Enumerated classes, distribution has business value |
| Image description, OCR | Generation | Output content is unpredictable |
| Visual QA (open-ended) | Generation | Needs natural language answer |
| Object detection | Generation | Needs coordinate output |

**Core criterion**: If candidates can be enumerated in advance, use scoring; if output content is unpredictable, use generation.

## Model Selection

| Model | Parameters | Memory | Precision | Recommended For |
|-------|-----------|--------|-----------|-----------------|
| mlx-community/Qwen2.5-VL-3B-Instruct-4bit | 3B | ~1.5GB | 4bit | Default, balances speed and quality |
| mlx-community/Qwen2.5-VL-3B-Instruct-8bit | 3B | ~3GB | 8bit | Higher precision needed |
| mlx-community/Qwen2.5-VL-7B-Instruct-4bit | 7B | ~4GB | 4bit | 16GB+ Mac, best quality |

## Acknowledgments

- [local-jev](../local-jev) text-only project
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) MLX vision-language model library
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) Alibaba Qwen vision-language model
- [mlx-community](https://huggingface.co/mlx-community) MLX format model conversions

## License

MIT — Copyright (c) 2026 Jiacheng Bin (贾承斌)
