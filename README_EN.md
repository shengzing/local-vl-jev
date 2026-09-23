# Local VL-Jev — Reproducing Jev-Style Visual Discrimination with Vision-Language Models

> An open-source project extending Jev-style fixed-answer scoring to the visual domain: input an image + question, skip text generation entirely, read the first token's logits, and apply softmax over known candidates to return a probability distribution.

**Author: Chengbin Jia (shengzing@163.com)**

English | [中文](./README.md)

## What Is This

[local-jev](https://github.com/shengzing/local-jev) demonstrated that fixed-answer scoring with text LLMs is 3-30x faster than text generation. This project extends the same approach to **vision-language models (VLMs)**—

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
    "invoice": 0.9978,
    "contract": 0.0016,
    "report": 0.0002,
    "letter": 0.0004
  },
  "threshold_passed": true,
  "latency_ms": 2775.2
}
```

### Run the benchmark

```bash
python benchmark/run_benchmark_vl.py --dataset scene_recognition
python benchmark/run_benchmark_vl.py --dataset document_classification
python benchmark/run_benchmark_vl.py --dataset content_moderation
```

### Regenerate test images

The images under `datasets/` and `assets/` are script-synthesized and reproducible:

```bash
python scripts/generate_dataset_images.py
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
| Speed advantage | 3x | 1.0-2.6x (measured; both paths share vision-encoding cost — see Benchmark Results) |

### Why Visual Scenarios Suit Scoring

1. **The probability distribution is more valuable**: the difference between 0.91 and 0.46 confidence directly determines auto-pass vs human review
2. **Deterministic, structured output**: scoring returns a `choice → probability` mapping for free; the generation path must parse free text, and our measured label-parse failure rate on explanatory answers was high
3. **Real speedup on long outputs**: uncontrolled-length generation (detailed descriptions, multi-step reasoning) takes 2-3x longer than scoring

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

### Model Loading

| Metric | Value |
|--------|-------|
| Model | mlx-community/Qwen2.5-VL-3B-Instruct-4bit |
| Load time | ~1s (model cached locally) |
| Vocab size | 151,643 |
| Memory (RSS) | ~3.3GB |

### Label Token Verification

| Label | Token ID | Status |
|-------|----------|--------|
| A (invoice) | [32] | ✓ single token |
| B (contract) | [33] | ✓ single token |
| C (report) | [34] | ✓ single token |
| D (letter) | [35] | ✓ single token |

Identical to text Jev — VLM label tokens are unaffected by image input.

### Dataset Scoring Results (`python benchmark/run_benchmark_vl.py`)

The repository ships three datasets (14 PIL-synthesized test images under `datasets/`):

| Dataset | Cases | Scoring acc | Generation acc | Scoring latency | Generation latency |
|---------|-------|-------------|----------------|-----------------|--------------------|
| scene_recognition | 5 | 5/5 (100%) | 5/5 (100%) | 1557 ms | 1464 ms |
| document_classification | 5 | 5/5 (100%) | 5/5 (100%) | 2473 ms | 2438 ms |
| content_moderation | 4 | 3/4 (75%) | 3/4 (75%) | 1480 ms | 1483 ms |

Per-image scoring detail (document_classification, all top probabilities 0.99+):

| Image | Expected | Decision | Correct | Top prob |
|-------|----------|----------|---------|----------|
| sample_invoice.jpg | invoice | invoice | ✓ | 0.999 |
| sample_contract.jpg | contract | contract | ✓ | 0.999 |
| sample_report.jpg | report | report | ✓ | 0.998 |
| sample_letter.jpg | letter | letter | ✓ | 0.996 |
| sample_receipt.jpg | invoice | invoice | ✓ | 0.997 |

### Scoring vs Generation: latency anatomy

The dominant cost of visual discrimination is **vision encoding + one 723-token forward pass** (~2.7s for a 640×800 image). The generation path pays the same cost; the only difference is the number of autoregressively generated tokens:

| Output form | Avg latency | vs scoring (2726 ms) |
|------------|-------------|----------------------|
| Jev-style scoring (0 generated tokens) | 2726 ms | 1.0x |
| Short answer generation (1-5 tokens) | 2642 ms | ≈1.0x |
| Explanatory generation (~25 tokens) | 3029 ms | 1.11x |
| Long description (150-250 tokens) | 7207 ms | 2.64x |

> **Honest conclusion**: when the model only needs to emit one or two tokens, scoring has no speed advantage (both paths share the same vision encoding and prefill cost). Scoring's advantages are: ① uncontrolled long-output scenarios (2-3x); ② a structured probability distribution for free (the generation path needs fragile text parsing — our measured label-parse failure rate on explanatory answers was high); ③ deterministic output.

### verify_mlx_vlm.py output (4 PIL text images)

```
  Image                     Expected     Decision      OK  Score(ms)    Gen(ms)
  invoice_test.png          invoice      invoice        ✓      742.3      793.9
  contract_test.png         contract     contract       ✓      733.8      785.4
  report_test.png           report       report         ✓      737.1      777.1
  letter_test.png           letter       letter         ✓      735.6      745.3

  Metric                    Scoring      Generation
  Accuracy                  4/4          4/4
  Avg latency (ms)          737.2        775.4
```

### Important fix record

The initial code **omitted `num_images=1`** when rendering with `apply_chat_template`, so the rendered prompt contained no `<|image_pad|>` placeholder and the image never actually reached the model — the initial README's "16.7 ms scoring latency, model always answers report" was a symptom of this bug (text-only forward + language prior). After the fix, images genuinely participate in inference: latency rises to a real ~1.5-2.7s (resolution-dependent) and accuracy recovers from 25% to 100%. See `docs/architecture.md`.

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
│   ├── document_classification/ ← Document classification dataset (with synthetic test images)
│   ├── content_moderation/      ← Content moderation dataset (with synthetic test images)
│   └── scene_recognition/       ← Scene recognition dataset (with synthetic test images)
├── verify_mlx_vlm.py            ← MLX-VLM verification script
├── scripts/
│   └── generate_dataset_images.py ← Regenerate all test images
├── assets/
│   └── sample.jpg                ← Quick Start sample image
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

- [local-jev](https://github.com/shengzing/local-jev) text-only project
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) MLX vision-language model library
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) Alibaba Qwen vision-language model
- [mlx-community](https://huggingface.co/mlx-community) MLX format model conversions

## License

MIT — Copyright (c) 2026 Chengbin Jia (贾承斌) (shengzing@163.com)
